"""E271 — does ANY receptor-similarity metric predict offset b(R) transfer? (redo, OOF, Ram's metrics)

Fixes two flaws Ram flagged:
  (1) e270 used IN-SAMPLE residuals to estimate b(R) -> shrunk, unreliable. Here: OUT-OF-FOLD (GroupKFold
      by receptor) residuals = honest b(R).
  (2) e268/e270 "sequence similarity" used PPIKB protein_seq, which is a FIXED 50-residue N-TERMINAL
      truncation (signal-peptide junk), NOT the pocket. Here we test the metrics that actually matter:
        M1 nterm50  : k-mer Jaccard of the N-term-50 (the OLD, bad metric — baseline)
        M2 pocketseq: k-mer Jaccard of the POCKET residue sequence (Ram idea #1 / #3: pocket = the
                      receptor residues the peptide interacts with)
        M3 pocketcomp: cosine of pocket residue COMPOSITION (charge/hydrophobic/aromatic/polar fractions)
        M4 pocketpkf: ProtDCal-3D pocket descriptor distance (Ram idea #2: 3D structure of the pocket)
DECISIVE TEST: corr(metric-similarity, -|Δb|) over multi-peptide-receptor pairs. A metric that predicts
small |Δb| (strongly positive corr) ENABLES cross-receptor anchoring by that metric. If all ~0, b(R) is
idiosyncratic by every metric and cross-receptor transfer is impossible regardless of similarity choice.
Run: OMP_NUM_THREADS=1 python experiments/e271_offset_transfer_metrics.py
"""
from __future__ import annotations
import json, importlib.util, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

spec = importlib.util.spec_from_file_location("e158", "scripts/e158_overfit_failure_analysis.py")
e158 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(e158)
except Exception:
    pass

KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
      "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
      "N": -3.5, "K": -3.9, "R": -4.5}
AROM = set("FWY"); POS = set("KR"); NEG = set("DE"); POL = set("STNQHY")


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


def comp(seq):
    n = max(len(seq), 1)
    return np.array([sum(c in POS for c in seq) / n, sum(c in NEG for c in seq) / n,
                     sum(c in AROM for c in seq) / n, sum(c in POL for c in seq) / n,
                     np.mean([KD.get(c, 0) for c in seq]) if seq else 0.0])


rows = [json.loads(l) for l in open("data/ppikb_features.jsonl")]
data = []
for r in rows:
    if not r.get("desc3d"):
        continue
    try:
        y = pf(r["y"]); d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"])
    except Exception:
        continue
    if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
        continue
    psq = e158.pocket_seq(r["pdb"].lower())
    data.append({"nterm": r["protein_seq"], "pep": r["seq"], "y": float(y), "d3": d3, "pk": pk,
                 "len": int(pf(r["length"])), "nc": float(pf(r["net_charge"])), "pocket": psq})
Ld = max(len(d["d3"]) for d in data); Lp = max(len(d["pk"]) for d in data)
data = [d for d in data if len(d["d3"]) == Ld and len(d["pk"]) == Lp]
X = np.array([d["d3"] + d["pk"] + [d["len"], d["nc"]] for d in data])
y = np.array([d["y"] for d in data])
recid = np.array([d["nterm"] for d in data])
print(f"PPIKB rows {len(data)} | with pocket_seq {sum(d['pocket'] is not None for d in data)}", flush=True)

# OOF absolute via GroupKFold by receptor (no same-receptor leakage into its own prediction)
uniq = {s: i for i, s in enumerate(sorted(set(recid)))}
groups = np.array([uniq[s] for s in recid])
S = np.full(len(y), np.nan)
gkf = GroupKFold(n_splits=8)
for tr, te in gkf.split(X, y, groups):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
    S[te] = m.predict(X[te])
resid = y - S
print(f"OOF absolute r={pearsonr(y, S)[0]:+.3f} MAE={np.mean(np.abs(resid)):.2f}", flush=True)

# b(R) per multi-peptide receptor (OOF residual mean); pocket info from the receptor's first pocket
by = defaultdict(list)
for i, s in enumerate(recid):
    by[s].append(i)
mrec = {s: idxs for s, idxs in by.items() if len({data[i]["pep"] for i in idxs}) >= 2}
bR, ntk, pseq, pcomp, ppkf = {}, {}, {}, {}, {}
PKz = (np.array([d["pk"] for d in data]) - np.array([d["pk"] for d in data]).mean(0)) / \
      (np.array([d["pk"] for d in data]).std(0) + 1e-9)
for s, idxs in mrec.items():
    bR[s] = float(np.mean([resid[i] for i in idxs]))
    ntk[s] = s
    ps = next((data[i]["pocket"] for i in idxs if data[i]["pocket"]), None)
    pseq[s] = ps
    pcomp[s] = comp(ps) if ps else None
    ppkf[s] = PKz[idxs].mean(0)
print(f"multi-peptide receptors: {len(mrec)} | with pocket_seq: {sum(v is not None for v in pseq.values())}")
print(f"TRUE OOF b(R) std = {np.std(list(bR.values())):.2f} kcal/mol "
      f"(vs e270 in-sample 0.61 — this is the honest magnitude)")


def km(s, k=3):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if s and len(s) >= k else set()
def jac(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0


keys = list(mrec.keys())
pairs = defaultdict(lambda: ([], []))   # metric -> (sim, -|db|)
for a in range(len(keys)):
    for b in range(a + 1, len(keys)):
        ka, kb = keys[a], keys[b]
        ndb = -abs(bR[ka] - bR[kb])
        pairs["M1_nterm50"][0].append(jac(km(ntk[ka], 4), km(ntk[kb], 4)))
        pairs["M1_nterm50"][1].append(ndb)
        if pseq[ka] and pseq[kb]:
            pairs["M2_pocketseq"][0].append(jac(km(pseq[ka]), km(pseq[kb])))
            pairs["M2_pocketseq"][1].append(ndb)
            ca, cb = pcomp[ka], pcomp[kb]
            cos = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))
            pairs["M3_pocketcomp"][0].append(cos); pairs["M3_pocketcomp"][1].append(ndb)
        pairs["M4_pocketpkf"][0].append(-float(np.linalg.norm(ppkf[ka] - ppkf[kb])))
        pairs["M4_pocketpkf"][1].append(ndb)

print("\nDECISIVE: corr(metric-similarity, -|Δb|)  [positive & large => predicts shared offset]")
print(f"  {'metric':16s} {'n_pairs':>8s} {'corr':>8s} {'p':>10s}")
res = {}
for mname in ["M1_nterm50", "M2_pocketseq", "M3_pocketcomp", "M4_pocketpkf"]:
    sim, ndb = np.array(pairs[mname][0]), np.array(pairs[mname][1])
    if len(sim) > 10 and np.std(sim) > 0:
        r, p = pearsonr(sim, ndb)
        print(f"  {mname:16s} {len(sim):>8d} {r:>+8.3f} {p:>10.1e}")
        res[mname] = dict(n=int(len(sim)), corr=float(r), p=float(p))
json.dump(dict(oof_b_std=float(np.std(list(bR.values()))), n_multirec=len(mrec), metrics=res),
          open("data/e271_offset_transfer.json", "w"), indent=1)
print("\nsaved data/e271_offset_transfer.json")
print("VERDICT: if even pocket-seq/pocket-comp corr ~0, b(R) is idiosyncratic by EVERY metric ->")
print("cross-receptor anchoring impossible regardless of similarity choice. If pocket metric >> nterm50,")
print("Ram is right: pocket similarity is the correct key and we re-run anchoring with it.")
