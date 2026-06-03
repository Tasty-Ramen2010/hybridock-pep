#!/usr/bin/env python3
"""
extract_vina_ad4_features.py — Per-pose Vina (--score_only) + AutoDock4 scores as
confidence-head features. These are the physics signals the current head IGNORES.

Motivation: Vina (charge-blind Gaussian) and AD4 (charge-aware, steeper LJ) are
different functional forms from ref2015, and both have genuine within-complex
variance across poses. The cross-session analysis predicted they complement the
encoder; this produces the data to test it.

Reuses production infrastructure exactly:
  - Existing pose ligand PDBQTs: <complex>/scoring/pdbqts/<model>/pose_{i}.pdbqt
  - Existing receptor PDBQT:      <complex>/scoring/receptor.pdbqt
  - Existing AD4 maps:            <complex>/scoring/maps/receptor.*.map
  - Vina grid (center + box) is parsed from the AD4 .map header so Vina scores on
    the SAME box AutoGrid used for AD4 — no guessing, train/serve consistent.

Output: feats_bench300_vina_ad4.pkl  →  {(cname, mkey, pose_idx): np.array([vina, ad4])}

Run (score-env has the vina python package; rapidock does not):
  ~/miniconda3/envs/score-env/bin/python scripts/extract_vina_ad4_features.py
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
log = logging.getLogger("vina_ad4")

REPO       = Path(__file__).resolve().parent.parent
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
OUT_PKL    = REPO / "logs" / "diagnosis" / "feats_bench300_vina_ad4.pkl"

# Poses outside the grid / unscoreable get this sentinel (semantically: bad pose).
_VINA_FAIL = 25.0
_AD4_FAIL  = 100.0


def parse_grid(map_path: Path) -> tuple[list[float], list[float]]:
    """Parse AutoGrid .map header → (center[3], box_size_ang[3])."""
    spacing = 0.375
    npts = [60, 60, 60]
    center = [0.0, 0.0, 0.0]
    with open(map_path) as fh:
        for _ in range(8):
            line = fh.readline()
            if not line:
                break
            parts = line.split()
            if parts and parts[0] == "SPACING":
                spacing = float(parts[1])
            elif parts and parts[0] == "NELEMENTS":
                npts = [int(x) for x in parts[1:4]]
            elif parts and parts[0] == "CENTER":
                center = [float(x) for x in parts[1:4]]
    box = [n * spacing for n in npts]
    return center, box


def score_complex(scoring_dir: Path, mkey: str, n_poses: int) -> dict[int, np.ndarray]:
    """Return {pose_idx: [vina, ad4]} for one (complex, model)."""
    from vina import Vina

    receptor_pdbqt = scoring_dir / "receptor.pdbqt"
    maps_dir = scoring_dir / "maps"
    pdbqt_dir = scoring_dir / "pdbqts" / mkey
    map_prefix = str(maps_dir / "receptor")
    if not (receptor_pdbqt.exists() and (maps_dir / "receptor.HD.map").exists()):
        return {}

    pose_pdbqts = {i: pdbqt_dir / f"pose_{i}.pdbqt" for i in range(n_poses)}
    pose_pdbqts = {i: p for i, p in pose_pdbqts.items() if p.exists()}
    if not pose_pdbqts:
        return {}

    center, box = parse_grid(maps_dir / "receptor.HD.map")

    # --- Vina score_only (one instance, maps once) ---
    vina_scores: dict[int, float] = {}
    try:
        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.compute_vina_maps(center=center, box_size=box)
        for i, p in pose_pdbqts.items():
            try:
                v.set_ligand_from_file(str(p))
                vina_scores[i] = float(v.score()[0])
            except Exception:
                vina_scores[i] = _VINA_FAIL
    except Exception as exc:
        log.debug("vina instance failed %s/%s: %s", scoring_dir.name, mkey, exc)

    # --- AD4 score (load_maps, NOT set_receptor) ---
    ad4_scores: dict[int, float] = {}
    try:
        va = Vina(sf_name="ad4", verbosity=0)
        va.load_maps(map_prefix)
        for i, p in pose_pdbqts.items():
            try:
                va.set_ligand_from_file(str(p))
                ad4_scores[i] = float(va.score()[0])
            except Exception:
                ad4_scores[i] = _AD4_FAIL
    except Exception as exc:
        log.debug("ad4 instance failed %s/%s: %s", scoring_dir.name, mkey, exc)

    out = {}
    for i in pose_pdbqts:
        out[i] = np.array([vina_scores.get(i, _VINA_FAIL),
                           ad4_scores.get(i, _AD4_FAIL)], dtype=np.float32)
    return out


def main():
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    feat_map: dict = {}
    if OUT_PKL.exists():
        feat_map = pickle.load(open(OUT_PKL, "rb"))
        log.info("Resuming: %d existing entries", len(feat_map))

    data = json.load(open(BENCH_JSON))
    t0 = time.time()
    n_cx = 0
    for ci, (cname, models) in enumerate(data.items()):
        for mkey, res in models.items():
            poses_dir = Path(res["poses_dir"])
            scoring_dir = poses_dir.parent.parent / "scoring"
            if not scoring_dir.is_dir():
                continue
            n_poses = len(res.get("ref_rmsds", []))
            if n_poses == 0:
                continue
            if (cname, mkey, 0) in feat_map:           # already done
                continue
            res_scores = score_complex(scoring_dir, mkey, n_poses)
            for i, vec in res_scores.items():
                feat_map[(cname, mkey, i)] = vec
            n_cx += 1
        if (ci + 1) % 10 == 0:
            pickle.dump(feat_map, open(OUT_PKL, "wb"))
            log.info("  [%d/%d cx] %d entries  %.0fs", ci + 1, len(data), len(feat_map), time.time() - t0)

    pickle.dump(feat_map, open(OUT_PKL, "wb"))
    arr = np.stack(list(feat_map.values())) if feat_map else np.zeros((0, 2))
    log.info("Done. %d pose-scores → %s", len(feat_map), OUT_PKL)
    if len(arr):
        log.info("vina  median %.2f  p99 %.2f", np.median(arr[:, 0]), np.percentile(arr[:, 0], 99))
        log.info("ad4   median %.2f  p99 %.2f", np.median(arr[:, 1]), np.percentile(arr[:, 1], 99))


if __name__ == "__main__":
    main()
