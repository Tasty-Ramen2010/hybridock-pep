"""E19 step-2b — does the signal survive REALISTIC (RAPiDock-like) poses?

We have no affinity-labeled RAPiDock poses, so simulate pose scatter: displace each
crystal-65 peptide by a target backbone RMSD (rigid-body translation/rotation + per-atom
Gaussian) matching typical RAPiDock pose error, recompute pocket+interface features on the
PERTURBED pose, and re-run the crystal-65 LOO. Degradation curve over RMSD = the honest
deployment expectation. Averaged over seeds.

Crystal-pose r=0.576 is the documented UPPER BOUND. This estimates the real number.
No GPU. Reuses cached free-peptide PDBs in /tmp/e18v3_pep? No — recomputes from e0 pep_pdb.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from e19_decompose_recover import (AA3to1, APOLAR, AROM, CHARGED, EISENBERG,  # noqa: E402
                                   HPHOBIC_AA, NEG, POLAR, POS, _per_res_sasa,
                                   pocket_descriptors)
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]


def _rand_rigid(coords, target_rmsd, rng):
    """Apply rigid translation+rotation+jitter to hit ~target_rmsd backbone displacement."""
    c0 = coords - coords.mean(0)
    # small random rotation
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9
    th = rng.normal(0, 0.15)  # rad
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    out = c0 @ Rm.T + coords.mean(0)
    out = out + rng.normal(0, 1.0, size=3)  # rigid translation seed
    # per-atom jitter, then rescale whole displacement to target rmsd
    out = out + rng.normal(0, 0.5, size=out.shape)
    disp = out - coords
    cur = np.sqrt((disp ** 2).sum(1).mean())
    if cur > 1e-6:
        out = coords + disp * (target_rmsd / cur)
    return out


def _features_from_coords(pep_res_template, pep_coords, free_sasa_map, rec_atoms, seq):
    """Recompute interface features given perturbed peptide coords (list of residues)."""
    # build a temp structure-like: we need SASA of perturbed peptide WITHIN receptor.
    # Approx ΔSASA via change handled by ShrakeRupley on a merged temp; here we approximate
    # buried SASA by contact-based occlusion is complex — instead recompute on perturbed
    # complex by writing a temp PDB.
    raise NotImplementedError


def perturbed_record(pep_pdb, poc_pdb, seq, target_rmsd, rng):
    """Write perturbed complex, recompute pocket+interface features."""
    # read peptide atoms
    pep = P.get_structure("p", pep_pdb)[0]
    atoms = [a for r in pep.get_residues() if r.id[0] == " " for a in r]
    coords = np.array([a.coord for a in atoms])
    if target_rmsd > 0:
        new = _rand_rigid(coords, target_rmsd, rng)
    else:
        new = coords
    # write perturbed peptide + receptor as merged complex
    tmp = Path(f"/tmp/e19rob/{Path(pep_pdb).stem}_{target_rmsd}.pdb")
    tmp.parent.mkdir(exist_ok=True)
    lines = []
    pep_lines = [l for l in Path(pep_pdb).read_text().splitlines()
                 if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH"]
    for i, l in enumerate(pep_lines):
        if i < len(new):
            x, y, z = new[i]
            l = f"{l[:21]}P{l[22:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}"
        lines.append(l)
    for l in Path(poc_pdb).read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH":
            lines.append(l[:21] + "R" + l[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    # features via the existing extractor logic
    from e19_decompose_recover import interface_features
    fi = interface_features(pep_pdb, str(tmp), "P", len(seq))  # free SASA from unperturbed pep
    cxm = P.get_structure("m", str(tmp))[0]
    pk = pocket_descriptors(cxm, "P")
    if not fi or not pk:
        return None
    return {**fi, **pk}


def loo(recs, feats, y):
    X = np.array([[r.get(f, 0.0) for f in feats] for r in recs], float)
    pred = np.zeros(len(recs))
    for i in range(len(recs)):
        tr = [j for j in range(len(recs)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pearsonr(pred, y).statistic


def main():
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    v1 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e18_cr.json").read_text())}
    base = [r for r in e0 if r.get("pep_pdb") and r["pdb"].upper() in v1]
    print(f"crystal-65 complexes with poses: {len(base)}")
    rmsds = [0.0, 1.0, 2.0, 3.0]
    n_seed = 3
    print(f"\n{'RMSD(Å)':>8}{'pocket':>9}{'interface':>11}{'pock+iface':>12}  (LOO r, mean of seeds)")
    for tr in rmsds:
        seeds_pock, seeds_if, seeds_both = [], [], []
        for s in range(n_seed if tr > 0 else 1):
            rng = np.random.default_rng(s)
            recs, ys = [], []
            for r in base:
                pdb = r["pdb"].upper(); seq = v1[pdb]["seq"]
                try:
                    f = perturbed_record(r["pep_pdb"], r["poc_pdb"], seq, tr, rng)
                except Exception:
                    f = None
                if f:
                    recs.append(f); ys.append(v1[pdb]["y"])
            y = np.array(ys)
            seeds_pock.append(loo(recs, POCK, y))
            seeds_if.append(loo(recs, IFACE, y))
            seeds_both.append(loo(recs, POCK + IFACE, y))
        print(f"{tr:>8.1f}{np.mean(seeds_pock):>9.3f}{np.mean(seeds_if):>11.3f}"
              f"{np.mean(seeds_both):>12.3f}")
    print("\nInterpretation: if pock+iface holds near 2-3 Å (typical RAPiDock backbone RMSD),")
    print("the signal may survive realistic poses. If it collapses, r=0.576 is oracle-only.")


if __name__ == "__main__":
    main()
