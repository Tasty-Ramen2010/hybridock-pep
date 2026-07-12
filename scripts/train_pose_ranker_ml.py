"""Train the ML pose ranker → data/pose_ranker_ml.joblib.

Learns to predict native Cα-RMSD from computable, OSI-clean pose features (Ramachandran +
RDKit 3D-shape; see src/hybridock_pep/scoring/pose_ranker_ml.py). Training data = real RAPiDock
poses from the cr65 real-pose campaign (runs/e93_realpose_campaign), with native RMSD computed
against the crystal peptide reference — exactly the deployment distribution that produced the
validated leave-one-complex-out τ = 0.406 (≈2× BSA+clash) in experiments/e96_poseranker_validation.py.

The φ/ψ Ramachandran KDEs are fitted here on the pooled training angles and bundled into the
artifact, so the runtime has NO external dependency. Re-run after adding new campaign poses to
refresh the model.

Bundle: {feature_names, phi_kde, psi_kde, model, n_train_poses, n_train_complexes}.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from hybridock_pep.scoring.pose_ranker_ml import (  # noqa: E402
    FEATURE_NAMES, shape_features, rama_features,
)

CAMP = ROOT / "runs" / "e93_realpose_campaign"
OUT = ROOT / "data" / "pose_ranker_ml.joblib"
P = PDBParser(QUIET=True)


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


def main() -> None:
    import math

    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    complexes = sorted([d.name for d in CAMP.iterdir() if (d / "poses").exists()])
    print(f"Training pose ranker on {len(complexes)} cr65 real-pose complexes")

    # Pass 1: collect all φ/ψ angles to fit the KDEs.
    all_phi: list[float] = []
    all_psi: list[float] = []
    from Bio.PDB import Polypeptide
    pose_paths: list[tuple[str, Path, float]] = []  # (complex, pose, native_rmsd)
    for cx in complexes:
        meta = bench.get(cx)
        if not meta:
            continue
        xtal = ca(Path(meta["peptide_pdb"]))
        for p in sorted((CAMP / cx / "poses").glob("pose_*.pdb"),
                        key=lambda q: int(q.stem.split("_")[1]))[:30]:
            try:
                tr = rmsd(ca(p), xtal)
                if math.isnan(tr):
                    continue
                pose_paths.append((cx, p, tr))
                struct = P.get_structure("x", str(p))[0]
                for chain in struct:
                    try:
                        poly = Polypeptide.Polypeptide(chain)
                    except Exception:  # noqa: BLE001
                        continue
                    for res in poly.get_phi_psi_list():
                        if None in res:
                            continue
                        all_phi.append(math.degrees(res[0]))
                        all_psi.append(math.degrees(res[1]))
            except Exception:  # noqa: BLE001
                continue
    phi_kde = gaussian_kde(all_phi)
    psi_kde = gaussian_kde(all_psi)
    print(f"  fitted φ/ψ KDEs on {len(all_phi)} angle pairs")

    # Pass 2: build feature matrix.
    X: list[list[float]] = []
    y: list[float] = []
    cx_set: set[str] = set()
    for cx, p, tr in pose_paths:
        rf = rama_features(p, phi_kde, psi_kde)
        if rf is None:
            continue
        sf = shape_features(p)
        if sf is None:
            continue
        X.append(rf + sf)
        y.append(tr)
        cx_set.add(cx)
    Xa, ya = np.array(X), np.array(y)
    print(f"  feature matrix {Xa.shape}  (target native RMSD: {ya.min():.1f}–{ya.max():.1f} Å)")

    model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=3, learning_rate=0.05, l2_regularization=1.0,
        min_samples_leaf=20, random_state=0,
    )
    model.fit(Xa, ya)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_names": FEATURE_NAMES, "phi_kde": phi_kde, "psi_kde": psi_kde,
                 "model": model, "n_train_poses": int(Xa.shape[0]),
                 "n_train_complexes": len(cx_set)}, OUT)
    size_kb = OUT.stat().st_size / 1024
    print(f"✓ wrote {OUT.relative_to(ROOT)} ({size_kb:.0f} KB, {Xa.shape[0]} poses / {len(cx_set)} complexes)")


if __name__ == "__main__":
    main()
