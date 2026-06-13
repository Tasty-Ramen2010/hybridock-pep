"""E96 — Does a PepScorer-computable pose ranker beat our BSA+clash ranker on OUR real poses?

Ram's directive: validate before wiring. The PepScorer paper's computable-only model predicts
native RMSD at r=0.469 (vs 0.687 full, which needs non-OSI Rescore+ terms). But that number is on
THEIR HADDOCK pose set. The deployment-relevant question is narrower and must be tested on OUR poses:

    On real RAPiDock poses, does a ranker built ONLY on computable PepScorer features rank poses
    closer to native (higher within-complex Kendall tau vs true Cα-RMSD) than our shipped BSA+clash
    ranker (tau ~= 0.14)?

If yes -> wiring the structural pose-picker is justified (better best_pose.pdb for users).
If no  -> the 0.469 didn't transfer; don't add the dependency.

Computable PepScorer features (NO Rescore+ / no mordred install needed):
  * Ramachandran family — reuses PepScorer's OWN shipped phi_kde / psi_kde joblibs, exact regions.
  * 3D-shape family — RDKit analogs of the mordred CPSA/GeometricalIndex/MomentOfInertia/PBF set:
    PMI1/2/3, NPR1/2, asphericity, eccentricity, radius-of-gyration, inertial-shape-factor,
    spherocity, PBF (plane-best-fit). Faithful in spirit; not byte-identical to mordred.

Honesty: this is a faithful reproduction of the COMPUTABLE subset, not PepScorer's exact 92-feature
pipeline. It tests whether that family of signals transfers to our poses — which is the decision.
Leave-one-complex-out HGBR so the ranker is never graded on a complex it trained on.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"  # don't starve the two RAPiDock campaigns sharing this box

import joblib
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from Bio.PDB import PDBParser, Polypeptide  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Descriptors3D, rdMolDescriptors  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

CAMP = ROOT / "runs" / "e93_realpose_campaign"
P = PDBParser(QUIET=True)
PHI_KDE = joblib.load(ROOT / "docs/PepScorerRMSD/objects/phi_kde.joblib")
PSI_KDE = joblib.load(ROOT / "docs/PepScorerRMSD/objects/psi_kde.joblib")
CLASH_DIST = 3.0


# ---------- ground truth: Cα RMSD (Kabsch) ----------
def ca(pdb: Path) -> np.ndarray:
    m = P.get_structure("x", str(pdb))[0]
    return np.array([a.coord for ch in m for r in ch if r.id[0] == " " for a in r if a.name == "CA"])


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


# ---------- production ranker: BSA + clash ----------
def heavy(pdb: Path):
    lines, xyz = [], []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        lines.append(ln)
    return lines, (np.array(xyz) if xyz else np.empty((0, 3)))


def sasa(lines) -> float:
    import io
    if not lines:
        return 0.0
    s = P.get_structure("x", io.StringIO("\n".join(lines) + "\nEND\n"))
    ShrakeRupley().compute(s, level="A")
    return float(sum(a.sasa for a in s.get_atoms()))


def bsa_clash(pose: Path, rec_lines, rec_xyz):
    pl, pxyz = heavy(pose)
    if len(pxyz) == 0:
        return None
    # crop receptor near peptide for SASA speed
    d = np.linalg.norm(rec_xyz[:, None, :] - pxyz[None, :, :], axis=2)
    near = d.min(1) < 10.0
    crop_lines = [rec_lines[i] for i in np.where(near)[0]]
    bsa = sasa(crop_lines) + sasa(pl) - sasa(crop_lines + pl)
    n_clash = int((d < CLASH_DIST).sum())
    return bsa, n_clash


# ---------- PepScorer-computable features ----------
def rama_feats(pose: Path):
    s = P.get_structure("x", str(pose))[0]
    reg = [0, 0, 0, 0]
    phis, psis = [], []
    for ch in s:
        try:
            poly = Polypeptide.Polypeptide(ch)
        except Exception:  # noqa: BLE001
            continue
        for residue in poly.get_phi_psi_list():
            if None in residue:
                continue
            phi, psi = math.degrees(residue[0]), math.degrees(residue[1])
            phis.append(phi)
            psis.append(psi)
            if ((-130 < phi < -50) and (120 < psi < 180)) or ((-75 < phi < -60) and (-50 < psi < -25)):
                reg[0] += 1
            elif ((-150 < phi < -45) and (100 < psi < 180)) or ((-90 < phi < -45) and (-65 < psi < 0)):
                reg[1] += 1
            elif ((-180 < phi < -30)) or ((30 < phi < 105) and (-30 < psi < 90)):
                reg[2] += 1
            else:
                reg[3] += 1
    n = len(phis)
    if n == 0:
        return None
    phi_prob = float(np.mean(PHI_KDE.evaluate(phis)))
    psi_prob = float(np.mean(PSI_KDE.evaluate(psis)))
    return [reg[0] / n, reg[1] / n, reg[2] / n, reg[3] / n,
            float(np.mean(phis)), float(np.mean(psis)), phi_prob, psi_prob]


def shape_feats(pose: Path):
    mol = Chem.MolFromPDBFile(str(pose), sanitize=False, removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        return [
            rdMolDescriptors.CalcPMI1(mol), rdMolDescriptors.CalcPMI2(mol), rdMolDescriptors.CalcPMI3(mol),
            rdMolDescriptors.CalcNPR1(mol), rdMolDescriptors.CalcNPR2(mol),
            Descriptors3D.Asphericity(mol), Descriptors3D.Eccentricity(mol),
            Descriptors3D.RadiusOfGyration(mol), Descriptors3D.InertialShapeFactor(mol),
            Descriptors3D.SpherocityIndex(mol), rdMolDescriptors.CalcPBF(mol),
        ]
    except Exception:  # noqa: BLE001
        return None


def main():
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    print(f"=== E96 pose-ranker validation on {len(complexes)} real-pose complexes ===\n")

    rows = []  # (complex, true_rmsd, bsa, n_clash, [computable feats])
    per_cx_bsa_tau = []
    for cx in complexes:
        meta = bench.get(cx)
        if not meta:
            continue
        xtal = ca(Path(meta["peptide_pdb"]))
        rec_lines, rec_xyz = heavy(Path(meta["pocket_pdb"]))
        if len(rec_xyz) == 0:
            continue
        poses = sorted((CAMP / cx / "poses").glob("pose_*.pdb"),
                       key=lambda p: int(p.stem.split("_")[1]))[:20]
        local = []
        for p in poses:
            try:
                tr = rmsd(ca(p), xtal)
                if math.isnan(tr):
                    continue
                bc = bsa_clash(p, rec_lines, rec_xyz)
                rf = rama_feats(p)
                sf = shape_feats(p)
                if bc is None or rf is None or sf is None:
                    continue
                local.append((tr, bc[0], bc[1], rf + sf))
            except Exception:  # noqa: BLE001
                continue
        if len(local) < 8:
            continue
        # production BSA+clash ranker: -z(BSA)+z(clash), within-complex
        bsa = np.array([r[1] for r in local]); cl = np.array([r[2] for r in local])
        trs = np.array([r[0] for r in local])
        zb = (bsa - bsa.mean()) / (bsa.std() + 1e-9)
        zc = (cl - cl.mean()) / (cl.std() + 1e-9)
        rank_score = -zb + zc  # lower = better fit
        tau_bsa = kendalltau(rank_score, trs)[0]  # low score should track low RMSD
        per_cx_bsa_tau.append(tau_bsa)
        for r in local:
            rows.append((cx, r[0], r[3]))
        print(f"  {cx}: {len(local)} poses  bestRMSD={trs.min():.1f}Å  BSA+clash tau={tau_bsa:+.2f}")

    # PepScorer-computable ranker: leave-one-complex-out HGBR predicting RMSD
    cxs = sorted(set(r[0] for r in rows))
    X = np.array([r[2] for r in rows])
    y = np.array([r[1] for r in rows])
    gid = np.array([cxs.index(r[0]) for r in rows])
    per_cx_pep_tau = []
    for ci, cx in enumerate(cxs):
        tr = gid != ci
        te = gid == ci
        if te.sum() < 8 or tr.sum() < 50:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        t = kendalltau(pred, y[te])[0]
        if t == t:
            per_cx_pep_tau.append(t)

    print("\n=== WITHIN-COMPLEX POSE-RANKING τ vs native RMSD (mean over complexes) ===")
    print(f"  production BSA+clash ranker : τ = {np.nanmean(per_cx_bsa_tau):+.3f}  (n={len(per_cx_bsa_tau)})")
    print(f"  PepScorer-computable (LOCO) : τ = {np.nanmean(per_cx_pep_tau):+.3f}  (n={len(per_cx_pep_tau)})")
    delta = np.nanmean(per_cx_pep_tau) - np.nanmean(per_cx_bsa_tau)
    print(f"  Δτ (PepScorer − BSA+clash)  : {delta:+.3f}")
    print("\n  reading: τ>0 ⇒ ranker puts low-RMSD (near-native) poses first.")
    print("  Δτ > ~0.05 and positive ⇒ wiring the structural pose-picker is justified (better best_pose).")
    print("  Δτ ≤ 0 ⇒ the 0.469 didn't transfer to our poses; don't add the dependency.")
    json.dump({"bsa_tau": float(np.nanmean(per_cx_bsa_tau)),
               "pep_tau": float(np.nanmean(per_cx_pep_tau)),
               "delta": float(delta), "n_cx_bsa": len(per_cx_bsa_tau),
               "n_cx_pep": len(per_cx_pep_tau), "n_poses": len(rows)},
              open(ROOT / "runs" / "e96_poseranker_validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
