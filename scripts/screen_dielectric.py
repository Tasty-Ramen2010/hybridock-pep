"""Phase 1.1 experiment: screen the MM-GBSA internal dielectric (εin) on PepSet-6.

The protein-peptide MM/PBSA(GBSA) literature finds εin ~1.4-2.0 beats the
OpenMM default of 1.0 (JCIM 2018, BiB 2025). This runs our real MM-GBSA at
εin ∈ {1, 2, 4} on the 6 PepSet calibration complexes (best non-clashed pose
each) and reports Pearson r + RMSE of ΔG_MMGBSA vs experimental ΔG.

CPU-only by design (force_cpu=True) — never contends with the GPU production dock.

Usage:
    python scripts/screen_dielectric.py
    python scripts/screen_dielectric.py --eps 1 1.5 2 4
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single


def _read_atoms(pdb: Path) -> list[tuple[str, np.ndarray]]:
    """Return [(line, xyz)] for ATOM/HETATM records."""
    out = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            out.append((line, xyz))
    return out


def crop_receptor(receptor_pdb: Path, pose_pdb: Path, radius: float) -> Path:
    """Keep only receptor residues with any atom within `radius` Å of the peptide.

    Distant residues contribute near-identically to E_complex and E_receptor, so
    they cancel in ΔG_bind — cropping preserves the interaction signal while
    cutting atom count ~5-10× (GB minimization is ~O(N²)). Whole residues are
    kept so backbone connectivity stays intact for the force field.
    """
    pep = np.array([xyz for _, xyz in _read_atoms(pose_pdb)])
    rec_atoms = _read_atoms(receptor_pdb)
    if len(pep) == 0 or not rec_atoms:
        return receptor_pdb
    r2 = radius * radius
    keep_res: set[tuple[str, str]] = set()
    for line, xyz in rec_atoms:
        d2 = np.min(np.sum((pep - xyz) ** 2, axis=1))
        if d2 <= r2:
            keep_res.add((line[21], line[22:27]))  # chain, resseq+icode
    kept = [line for line, _ in rec_atoms if (line[21], line[22:27]) in keep_res]
    tmp = Path(tempfile.mkstemp(suffix="_pocket.pdb", prefix=f"{receptor_pdb.stem}_")[1])
    tmp.write_text("\n".join(kept) + "\nEND\n")
    return tmp

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "runs" / "calibration_production"
SCORES = json.loads((ROOT / "data" / "training_scores_production.json").read_text())

_PKD_TO_DG = -1.3633  # ΔG = -RT ln(10) pKd at 298 K


def experimental_dg() -> dict[str, float]:
    out: dict[str, float] = {}
    with (ROOT / "data" / "training_complexes.csv").open() as f:
        for r in csv.DictReader(f):
            out[r["pdb_id"].lower()] = _PKD_TO_DG * float(r["experimental_pkd"])
    return out


def best_pose_path(pdb_id: str) -> Path | None:
    """Best non-clashed minimized pose for a complex (falls back to best available)."""
    min_dir = PROD / pdb_id / "poses_minimized"
    if not min_dir.is_dir():
        return None
    top = SCORES.get(pdb_id, {}).get("top_k_poses", [])
    # Prefer non-clashed poses in rank order; else any minimized pose that exists.
    for entry in sorted(top, key=lambda e: (e.get("is_clashed", False), e.get("vina_score", 0))):
        cand = min_dir / entry["pose"]
        if cand.exists():
            return cand
    existing = sorted(min_dir.glob("pose_*.pdb"))
    return existing[0] if existing else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0, 2.0, 4.0],
                    help="Internal dielectric values to screen.")
    ap.add_argument("--crop-radius", type=float, default=12.0,
                    help="Crop receptor to residues within this many Å of the peptide "
                         "(0 = no crop). Speeds up GB minimization; distant residues "
                         "cancel in ΔG.")
    args = ap.parse_args()

    dg_exp = experimental_dg()
    complexes = [d.name for d in sorted(PROD.iterdir()) if d.is_dir()]

    # Resolve receptor + pose per complex once.
    jobs: list[tuple[str, Path, Path, float]] = []
    for cid in complexes:
        rec = PROD / cid / "receptor_for_rapidock.pdb"
        pose = best_pose_path(cid)
        if rec.exists() and pose is not None and cid in dg_exp:
            if args.crop_radius > 0:
                cropped = crop_receptor(rec, pose, args.crop_radius)
                n_full = len(_read_atoms(rec))
                n_crop = len(_read_atoms(cropped))
                print(f"  {cid}: cropped receptor {n_full}→{n_crop} atoms "
                      f"(≤{args.crop_radius}Å)", flush=True)
                rec = cropped
            jobs.append((cid, rec, pose, dg_exp[cid]))
        else:
            print(f"  skip {cid}: rec={rec.exists()} pose={pose is not None} dg={cid in dg_exp}")

    print(f"Screening εin={args.eps} on {len(jobs)} complexes (CPU)\n")
    results: dict[float, list[tuple[str, float, float]]] = {e: [] for e in args.eps}
    for cid, rec, pose, dge in jobs:
        line = f"  {cid:6s} ΔG_exp={dge:6.2f} | "
        for eps in args.eps:
            try:
                dg = compute_mmgbsa_single(pose, rec, force_cpu=True, solute_dielectric=eps)
            except Exception as exc:  # noqa: BLE001 — experiment harness, log + continue
                dg = float("nan")
                line += f"εin={eps}: ERR({type(exc).__name__}) "
                continue
            results[eps].append((cid, dg, dge))
            line += f"εin={eps}:{dg:8.1f} "
        print(line, flush=True)

    print("\n=== εin screen summary (ΔG_MMGBSA vs ΔG_exp) ===")
    print(f"  {'εin':>5s} {'n':>3s} {'pearson_r':>10s} {'RMSE':>8s}")
    for eps in args.eps:
        rows = results[eps]
        if len(rows) < 3:
            print(f"  {eps:5.1f} {len(rows):3d}   too few")
            continue
        pred = np.array([d for _, d, _ in rows])
        y = np.array([g for _, _, g in rows])
        r = pearsonr(pred, y).statistic
        # RMSE after slope+intercept refit (MM-GBSA absolute scale is arbitrary).
        A = np.vstack([pred, np.ones_like(pred)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        rmse = float(np.sqrt(np.mean((m * pred + b - y) ** 2)))
        print(f"  {eps:5.1f} {len(rows):3d} {r:+10.3f} {rmse:8.2f}")


if __name__ == "__main__":
    main()
