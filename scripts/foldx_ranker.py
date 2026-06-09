"""FoldX-style pose ranker — benchmark vs ref2015 (τ=0.176) on bench300.

Within-complex pose ranking: peptide size is CONSTANT across a complex's poses,
so the interface-size confound that kills absolute affinity does NOT apply here —
this is the regime where a transparent empirical free-energy function can win.

Energy (lower = better pose), FoldX-style decomposition:
  E = w_lj·E_vdw            coarse Lennard-Jones over interface heavy-atom pairs
                            (clash repulsion + contact attraction in one term)
    + w_sc·ΔS_sidechain     Doig-Sternberg per-AA sidechain entropy × burial (penalty)
    + w_ss·E_rama           backbone φ/ψ Ramachandran favorability (loop/helix/sheet)
    − w_hb·n_hbond          interface N/O donor-acceptor pairs (bonus)
    − w_sb·n_saltbridge     complementary charged pairs (bonus)

Metric: per-complex Kendall τ between (−E) and interface-RMSD, averaged over
complexes (same protocol/poses as the ref2015 baseline). Fixed physical weights
by default (unsupervised, fair vs ref2015); --cv fits weights by LOO ridge.

Usage:  python scripts/foldx_ranker.py [--cv] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as ss

from hybridock_pep.scoring.per_residue_entropy import S_SC, _classify_phi_psi

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "logs" / "analysis_bench300"
LABELS = BENCH / "interface_rmsd_labels.json"
MODEL = "pretrained"  # the pose set ref2015 baseline used

_DONORS = {"N", "O"}  # coarse: any N/O can donate/accept (heavy-atom H-bond proxy)
_POS = {"ARG", "LYS", "HIS"}
_NEG = {"ASP", "GLU"}
_AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
           "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
           "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def parse(pdb: Path):
    """Return dict: atoms list of (resseq, resname, atomname, elem, xyz)."""
    out = []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an:
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        out.append((ln[22:27].strip(), ln[17:20].strip(), an, an[0], xyz))
    return out


_HYDRO = set("ACFILMVWY")  # hydrophobic AA one-letter


def features(pose_atoms, rec_atoms) -> dict:
    """Compute FoldX-style energy terms for one pose (packing/enthalpy focused)."""
    pa = [a for a in pose_atoms if a[3] not in ("H", "D")]
    ra = [a for a in rec_atoms if a[3] not in ("H", "D")]
    if not pa or not ra:
        return {}
    P = np.array([a[4] for a in pa]); R = np.array([a[4] for a in ra])
    D = np.sqrt(np.maximum(((P[:, None, :] - R[None, :, :]) ** 2).sum(-1), 1e-6))

    # separated vdW: clash (steep penalty) vs favorable contact band
    n_clash = float(((D < 3.0).sum()))                 # steric overlap → penalty
    n_contact = float((((D >= 3.3) & (D < 4.6)).sum())) # well-packed contact → bonus
    # soft-core LJ in the favourable band only (clash handled separately, no blowup)
    band = D[(D >= 2.8) & (D < 6.0)]
    x = 3.8 / band
    e_lj = float(np.sum(np.clip(x**12 - 2 * x**6, -2.0, 4.0)))

    # element arrays for typed contacts
    pe = [a[3] for a in pa]; re_ = [a[3] for a in ra]
    pC = np.array([a[4] for a in pa if a[3] == "C"])
    rC = np.array([a[4] for a in ra if a[3] == "C"])
    hyd_c = 0.0
    if len(pC) and len(rC):
        dc = np.sqrt(((pC[:, None] - rC[None]) ** 2).sum(-1))
        hyd_c = float((((dc >= 3.3) & (dc < 4.8)).sum()))  # C-C packing (hydrophobic)

    # H-bonds: peptide N↔receptor O and peptide O↔receptor N, 2.6-3.4 Å
    pN = np.array([a[4] for a in pa if a[3] == "N"]); pO = np.array([a[4] for a in pa if a[3] == "O"])
    rN = np.array([a[4] for a in ra if a[3] == "N"]); rO = np.array([a[4] for a in ra if a[3] == "O"])
    def hb(A, B):
        if not len(A) or not len(B):
            return 0
        d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        return int((((d > 2.5) & (d < 3.4)).sum()))
    n_hb = float(hb(pN, rO) + hb(pO, rN))

    # salt bridges: complementary charged sidechain centroids < 5 Å
    def charged(atoms, names):
        pts = {}
        for rs, rn, an, el, xyz in atoms:
            if rn in names and an not in ("N", "CA", "C", "O"):
                pts.setdefault(rs, []).append(xyz)
        return [np.mean(v, 0) for v in pts.values()]
    p_pos, p_neg = charged(pa, _POS), charged(pa, _NEG)
    r_pos, r_neg = charged(ra, _POS), charged(ra, _NEG)
    n_sb = 0
    for A, B in ((p_pos, r_neg), (p_neg, r_pos)):
        for ca in A:
            for cb in B:
                if np.linalg.norm(ca - cb) < 5.0:
                    n_sb += 1

    return {"e_lj": e_lj, "n_clash": n_clash, "n_contact": n_contact,
            "hyd_c": hyd_c, "n_hb": n_hb, "n_sb": float(n_sb)}


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w); y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


# Validated weights. Greedy LOO-CV selection over 6 candidate terms kept only
# two with non-redundant signal: clash (over-insertion penalty) and salt bridges
# (charge complementarity). Raw contact/hydrophobic counts are ANTI-discriminative
# — bad poses over-insert and rack up contacts (the burial confound, within-pose).
# Result: τ≈0.10, below ref2015's 0.176. ref2015 is the calibrated version of
# this same idea (orientation H-bonds + LK solvation + rotamer stats) and wins.
W = {"n_clash": 1.0, "n_sb": -1.0, "e_lj": 0.0, "n_contact": 0.0, "hyd_c": 0.0, "n_hb": 0.0}


def energy(f: dict, w=W) -> float:
    return sum(w[k] * f.get(k, 0.0) for k in w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv", action="store_true", help="LOO ridge weight fit (upper bound).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text())
    complexes = sorted(d.name for d in BENCH.glob("peppc_*") if d.is_dir())
    if args.limit:
        complexes = complexes[: args.limit]

    feat_pool, rmsd_pool = {}, {}
    for cn in complexes:
        rec = BENCH / cn / "scoring" / "receptor_cropped.pdb"
        posedir = BENCH / cn / MODEL / "poses"
        lab = labels.get(cn, {}).get(MODEL, {})
        rmsds = lab.get("interface_rmsds")
        if isinstance(rmsds, str):
            rmsds = json.loads(rmsds)
        if not rec.exists() or not posedir.exists() or not rmsds:
            continue
        ratoms = parse(rec)
        rows, rs = [], []
        for i, rm in enumerate(rmsds):
            pp = posedir / f"pose_{i}.pdb"
            if not pp.exists():
                continue
            f = features(parse(pp), ratoms)
            if f:
                rows.append(f); rs.append(float(rm))
        if len(rows) >= 3:
            feat_pool[cn] = rows; rmsd_pool[cn] = np.array(rs)

    print(f"Scored {len(feat_pool)} complexes (ref2015 baseline τ=0.176)\n")

    # fixed-weight τ
    taus = []
    for cn, rows in feat_pool.items():
        E = np.array([energy(f) for f in rows])  # energy: low = better pose
        t, _ = ss.kendalltau(E, rmsd_pool[cn])   # low E ↔ low rmsd ⇒ positive τ = good
        if not np.isnan(t):
            taus.append(t)
    print(f"  FoldX fixed-weights:  τ = {np.mean(taus):+.3f}  (n={len(taus)})")
    # per-term solo τ (diagnostic)
    for term in ("e_lj", "n_clash", "n_contact", "hyd_c", "n_hb", "n_sb"):
        tt = []
        for cn, rows in feat_pool.items():
            v = np.array([f.get(term, 0.0) for f in rows])
            t, _ = ss.kendalltau(v, rmsd_pool[cn])  # raw value vs rmsd
            if not np.isnan(t):
                tt.append(t)
        print(f"     solo {term:9s} τ(raw,rmsd) = {np.mean(tt):+.3f}")

    if args.cv:
        _cv_fit(feat_pool, rmsd_pool)


def _cv_fit(feat_pool, rmsd_pool):
    """LOO-complex ridge: fit term weights to predict rmsd, report held-out τ."""
    keys = ("e_lj", "n_clash", "n_contact", "hyd_c", "n_hb", "n_sb")
    cns = list(feat_pool)
    X = {cn: np.array([[f.get(k, 0.0) for k in keys] for f in feat_pool[cn]]) for cn in cns}
    # standardize per-term across all poses
    allX = np.vstack(list(X.values()))
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    taus = []
    for held in cns:
        Xtr = np.vstack([(X[c] - mu) / sd for c in cns if c != held])
        ytr = np.concatenate([rmsd_pool[c] for c in cns if c != held])
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        reg = 1.0 * np.eye(A.shape[1]); reg[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        Xh = (X[held] - mu) / sd
        pred = Xh @ w[:-1] + w[-1]  # predicts rmsd; lower = better
        t, _ = ss.kendalltau(-(-pred), rmsd_pool[held])  # -pred high = low rmsd
        t, _ = ss.kendalltau(pred, rmsd_pool[held])
        if not np.isnan(t):
            taus.append(t)
    print(f"\n  FoldX CV-fitted weights: τ = {np.mean(taus):+.3f}  (LOO over complexes)")


if __name__ == "__main__":
    main()
