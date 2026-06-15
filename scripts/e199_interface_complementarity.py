"""E199 — proper INTER-CHAIN interface complementarity (Ram's match-score, done physically).

PPI's wNc is INTRA-peptide contacts. The physically right complementarity for BINDING is INTER-chain:
for each peptide residue in contact with a pocket residue (Cb-Cb < cutoff), sum the property MATCH across
the interface. We compute, over interface contacts:
   hyd_compl   = Σ  hyd(pep_i) · hyd(pock_j)            (like-likes-like; hydrophobic patch on hydrophobic patch)
   hyd_mismatch= Σ  (hyd(pep_i) − hyd(pock_j))^2         (penalises hydrophilic-in-hydrophobic)
   shape_compl = Σ  vol(pep_i) · vol(pock_j)             (bulky-fills-bulky)
   arom_compl  = Σ  arom(pep_i) · arom(pock_j)           (π-stacking)
   n_contacts  = interface contact count
Cached to data/e199_compl.jsonl. Then: does adding these to the pooled model help NEUTRAL (and overall),
clustered-CV crystal-925?
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
import e180_protdcal_925 as e180  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "scripts/e150_protdcal_descriptors.py").loader.exec_module(e150)
SD, SCALES, POS, NEG = e150.seq_descriptors, e150.SCALES, e150.POS, e150.NEG
SN = list(SCALES.keys())
T2O = e180.T2O
PDBDIR = ROOT / "data" / "rcsb_full"
_parser = PDBParser(QUIET=True)
OUT = ROOT / "data" / "e199_compl.jsonl"
HYD, VOL, AROM = SCALES["kd"], SCALES["vol"], SCALES["arom"]


def chain_res(st, ch):
    out = []
    for r in st[0][ch]:
        if r.id[0] != " ":
            continue
        aa = T2O.get(r.resname)
        if aa is None:
            continue
        a = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
        if a is not None:
            out.append((aa, a.coord))
    return out


def complementarity(pdb, want_seq, cutoff=8.0):
    f = PDBDIR / f"{pdb}.pdb"
    if not f.exists():
        f = e180.fetch(pdb)
    if f is None:
        return None
    try:
        st = _parser.get_structure(pdb, str(f))
    except Exception:  # noqa: BLE001
        return None
    want = want_seq.upper(); L = len(want)
    pep, pep_ch = None, None
    for ch in st[0]:
        res = chain_res(st, ch.id)
        seq = "".join(a for a, _ in res)
        if (2 <= len(res) <= 60) and (want in seq or (seq and seq in want)) and abs(len(res) - L) <= max(3, 0.4 * L):
            pep, pep_ch = res, ch.id; break
    if pep is None:
        return None
    pock = []
    pxyz = np.array([c for _, c in pep])
    for ch in st[0]:
        if ch.id == pep_ch:
            continue
        for a, c in chain_res(st, ch.id):
            if ((np.asarray(c)[None, :] - pxyz) ** 2).sum(-1).min() <= cutoff ** 2:
                pock.append((a, c))
    if not pock:
        return None
    hc = hm = sc = ac = nc = 0.0
    for ai, ci in pep:
        for aj, cj in pock:
            if ((np.asarray(ci) - np.asarray(cj)) ** 2).sum() <= cutoff ** 2:
                hi, hj = HYD.get(ai, 0), HYD.get(aj, 0)
                hc += hi * hj; hm += (hi - hj) ** 2
                sc += VOL.get(ai, 0) * VOL.get(aj, 0)
                ac += AROM.get(ai, 0) * AROM.get(aj, 0); nc += 1
    if nc == 0:
        return None
    return [hc / nc, hm / nc, sc / nc, ac / nc, nc]  # per-contact normalised + count


def build_cache():
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    rows = [json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")]
    todo = [r for r in rows if r["pdb"] not in done]
    if not todo:
        return
    print(f"computing complementarity for {len(todo)}...", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo):
        c = complementarity(r["pdb"], r["seq"])
        with open(OUT, "a") as fh:
            fh.write(json.dumps({"pdb": r["pdb"], "compl": c}) + "\n")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} {(time.time()-t0)/(i+1):.2f}s", flush=True)


def main():
    build_cache()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    compl = {json.loads(l)["pdb"].lower(): json.loads(l)["compl"]
             for l in open(OUT) if json.loads(l).get("compl")}
    GEO = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
           "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac", "mean_burial"]
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        pid = r["pdb"].lower()
        if pid not in compl:
            continue
        ps = e158.pocket_seq(pid)
        if ps is None:
            continue
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        rows.append({"seq": r["seq"], "y": float(r["y"]), "q": abs(pq), "L": r["length"], "pn": float(r["poc_net"]),
                     "geo": [float(r.get(k, 0)) for k in GEO], "compl": compl[pid], "ps": ps,
                     "pkf": [float(np.mean([SCALES[s2].get(c, 0) for c in ps])) for s2 in SN]})
    print(f"\ncrystal with complementarity: n={len(rows)}", flush=True)
    y = np.array([r["y"] for r in rows]); q = np.array([r["q"] for r in rows]); L = np.array([r["L"] for r in rows])
    grp, _ = e158.greedy_cluster([r["ps"] for r in rows], 0.7)

    def feat(r, withc):
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        f = SD(r["seq"]) + r["pkf"] + r["geo"] + [pq * r["pn"], abs(pq) * abs(r["pn"]), abs(pq + r["pn"]), float(len(r["seq"]))]
        return f + (r["compl"] if withc else [])

    def cv(withc):
        X = np.nan_to_num([feat(r, withc) for r in rows]); pred = np.full(len(rows), np.nan)
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        return pred

    pb, pc = cv(False), cv(True)

    def met(p, mask):
        return float(np.corrcoef(p[mask], y[mask])[0, 1])
    # single-feature corr of complementarity terms within neutral
    neu = q <= 1
    C = np.array([r["compl"] for r in rows])
    print("\n=== complementarity single-feature |r| with affinity (NEUTRAL slice) ===")
    for j, nm in enumerate(["hyd_compl", "hyd_mismatch", "shape_compl", "arom_compl", "n_contacts"]):
        print(f"  {nm:<14} r={np.corrcoef(C[neu, j], y[neu])[0,1]:+.3f}")
    print("\n=== clustered-CV: base vs +interface-complementarity ===")
    for nm, mk in [("OVERALL", np.ones(len(rows), bool)), ("neutral|q|<=1", q <= 1),
                   ("charged|q|>=2", q >= 2), ("long13-16", (L >= 13) & (L <= 16))]:
        print(f"  {nm:<14} n={mk.sum():<4} base={met(pb, mk):+.3f}  +compl={met(pc, mk):+.3f}  Δ={met(pc,mk)-met(pb,mk):+.3f}")


if __name__ == "__main__":
    main()
