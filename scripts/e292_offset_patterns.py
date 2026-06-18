"""E292 — decompose per-complex offsets b(receptor) & c(peptide), correlate with EVERY descriptor, find patterns.

Pipeline:
  1. OOF model S, residual e = S − y (the gap b(y)+c(x)+eta).
  2. Ridge two-way decomposition (alternating): e(x,y) ≈ c(x) + b(y). Gauge-free correlations.
  3. For peptides: correlate c(x) with 37 ProtDCal-3D desc + physical (length, net charge, |q|, hydrophobicity,
     aromatic/charged/polar fraction, etc.). For receptors: correlate b(y) with 22 pocket ProtDCal-3D.
  4. Report ALL correlations sorted by |r|, with p-values + Bonferroni threshold; validate top hits by split-half.
HONEST: S already TRAINED on desc3d+pocket_pkf, so correlations with those should be ~0 BY CONSTRUCTION
(the model extracted the linear signal). Real news = (a) any survivor of Bonferroni, (b) correlations with
DERIVED physical quantities the model may underuse (charge, size). 60 tests -> expect ~3 false positives at
p<0.05; only Bonferroni/split-half survivors are real.
Run: OMP_NUM_THREADS=1 python scripts/e292_offset_patterns.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
      "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
      "N": -3.5, "K": -3.9, "R": -4.5}
POS = set("KR"); NEG = set("DE"); AROM = set("FWY"); POL = set("STNQHY")
def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


def pep_phys(s):
    n = max(len(s), 1)
    return {"length": len(s), "net_charge": sum(c in POS for c in s) - sum(c in NEG for c in s),
            "abs_net_charge": abs(sum(c in POS for c in s) - sum(c in NEG for c in s)),
            "pos_frac": sum(c in POS for c in s) / n, "neg_frac": sum(c in NEG for c in s) / n,
            "charged_frac": sum(c in POS | NEG for c in s) / n, "arom_frac": sum(c in AROM for c in s) / n,
            "polar_frac": sum(c in POL for c in s) / n, "hydrophobicity": np.mean([KD.get(c, 0) for c in s]) if s else 0,
            "n_pos": sum(c in POS for c in s), "n_neg": sum(c in NEG for c in s),
            "n_KR": sum(c in "KR" for c in s), "n_DE": sum(c in "DE" for c in s),
            "n_FWY": sum(c in AROM for c in s), "n_cys": s.count("C"), "n_pro": s.count("P")}


recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD") or not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and len(pk) == 22 and np.isfinite(y):
        recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y), "d3": d3, "pk": pk,
                     "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
recname = [r["rec"] for r in recs]; pepname = [r["pep"] for r in recs]
grp = np.array([hash(s) % (10**9) for s in recname])
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
e = S - y
print(f"complexes {len(recs)} | residual e std {e.std():.2f}", flush=True)

# ridge two-way decomposition
ridx = {s: i for i, s in enumerate(sorted(set(recname)))}
pidx = {s: i for i, s in enumerate(sorted(set(pepname)))}
ri = np.array([ridx[s] for s in recname]); pi = np.array([pidx[s] for s in pepname])
rec_cells = defaultdict(list); pep_cells = defaultdict(list)
for i in range(len(e)):
    rec_cells[ri[i]].append(i); pep_cells[pi[i]].append(i)
b = np.zeros(len(ridx)); c = np.zeros(len(pidx)); lam = 3.0
for _ in range(60):
    for rr, cc in rec_cells.items():
        b[rr] = np.sum([e[i] - c[pi[i]] for i in cc]) / (len(cc) + lam)
    for pp, cc in pep_cells.items():
        c[pp] = np.sum([e[i] - b[ri[i]] for i in cc]) / (len(cc) + lam)
# per-complex
cx = np.array([c[pi[i]] for i in range(len(e))])
by = np.array([b[ri[i]] for i in range(len(e))])
print(f"  fit residual after b+c: {np.std(e - cx - by):.2f} (was {e.std():.2f}) | "
      f"b std {b.std():.2f} c std {c.std():.2f}")

# save full offset table
out = [{"pep": pepname[i], "rec_hash": int(grp[i]), "y": float(y[i]), "S": float(S[i]),
        "offset_total": float(e[i]), "c_peptide": float(cx[i]), "b_receptor": float(by[i])}
       for i in range(len(e))]
json.dump(out, open("data/e292_offsets.json", "w"))

# unique-level offsets for correlation (one per peptide / receptor, reliable = >=2 cells)
def corr_report(title, level_offset, level_descs, descnames, n_tests):
    print(f"\n=== {title}: correlations (sorted |r|), Bonferroni p<{0.05/n_tests:.1e} ===")
    res = []
    for k in descnames:
        v = np.array([level_descs[j][k] for j in range(len(level_offset))], float)
        if np.std(v) < 1e-9:
            continue
        r, p = pearsonr(level_offset, v)
        res.append((k, r, p))
    res.sort(key=lambda t: -abs(t[1]))
    for k, r, p in res[:18]:
        flag = "***BONF***" if p < 0.05 / n_tests else ("*" if p < 0.05 else "")
        print(f"  {k:22s} r={r:+.3f} p={p:.1e} {flag}")
    return res


# peptide level
pep_ids = [s for s, cc in pep_cells.items() if len(cc) >= 2]
pep_off = np.array([c[s] for s in pep_ids])
pep_seq = {pi[cc[0]]: pepname[cc[0]] for s, cc in pep_cells.items() for _ in [0]}  # map id->seq
id2seq = {}
for i in range(len(recs)):
    id2seq[pi[i]] = pepname[i]
pdescs = []
for s in pep_ids:
    seq = id2seq[s]; ph = pep_phys(seq)
    # add the 37 ProtDCal desc (mean over this peptide's cells)
    cc = pep_cells[s]; d3m = np.mean([recs[i]["d3"] for i in cc], axis=0)
    for j in range(37):
        ph[f"protdcal_{j}"] = d3m[j]
    pdescs.append(ph)
names_pep = list(pdescs[0].keys())
n_tests = len(names_pep) + 22
corr_report(f"PEPTIDE offset c(x), n={len(pep_ids)} peptides (>=2 receptors)", pep_off, pdescs, names_pep, n_tests)

# receptor level
rec_ids = [s for s, cc in rec_cells.items() if len(cc) >= 2]
rec_off = np.array([b[s] for s in rec_ids])
rdescs = []
for s in rec_ids:
    cc = rec_cells[s]; pkm = np.mean([recs[i]["pk"] for i in cc], axis=0)
    rdescs.append({f"pocket_pkf_{j}": pkm[j] for j in range(22)})
names_rec = list(rdescs[0].keys())
corr_report(f"RECEPTOR offset b(y), n={len(rec_ids)} receptors (>=2 peptides)", rec_off, rdescs, names_rec, n_tests)

# split-half validation of the single strongest peptide-charge correlation
print("\n=== split-half validation: peptide |net charge| vs c(x) (the charged-floor hypothesis) ===")
q = np.array([pep_phys(id2seq[s])["abs_net_charge"] for s in pep_ids], float)
rng = np.random.default_rng(0); perm = rng.permutation(len(pep_ids)); half = len(perm) // 2
r1 = pearsonr(pep_off[perm[:half]], q[perm[:half]])[0]
r2 = pearsonr(pep_off[perm[half:]], q[perm[half:]])[0]
print(f"  half1 r={r1:+.3f} | half2 r={r2:+.3f} (consistent sign+magnitude => real)")
print("\nsaved data/e292_offsets.json")
