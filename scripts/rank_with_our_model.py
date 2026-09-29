#!/usr/bin/env python
"""Rank each complex's poses with OUR pose ranker, not with Rosetta.

WHY NOT ref2015. RAPiDock's published pipeline ranks with ref2015, and inference.py will happily
do the same for us. But then the benchmark measures "RAPiDock sampling plus Rosetta", which is
not the tool we built and not a result we could honestly claim. HybriDock-Pep ships
pose_ranker_ml (predicted native Ca-RMSD, lower is better) with BSA-fit as the documented
fallback, so that is what ranks the poses here.

BOTH DIFFUSION ARMS GET THE SAME RANKER. Giving ours a ranker and leaving the baseline unranked,
or ranking the baseline with something weaker, would manufacture a win out of the scoring stage.
With one ranker across both, the only thing that differs is the generative checkpoint, which is
the thing the retrain actually changed.

Writes logs/ranking_<arm>.jsonl: one row per complex with the pose order our ranker chose.

Usage: rank_with_our_model.py --arm <name> --runs runs/recentset [--bench data/....csv]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--runs", default="runs/recentset")
    ap.add_argument("--bench", default="data/bench_balanced_length.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    from hybridock_pep.scoring.pose_ranker_ml import (
        compute_features, _load_bundle, DEFAULT_MODEL_PATH)
    bundle = _load_bundle(DEFAULT_MODEL_PATH)
    if bundle is None:
        print("pose_ranker_ml artifact missing; cannot rank"); return
    phi, psi, model = bundle["phi_kde"], bundle["psi_kde"], bundle["model"]

    rows = list(csv.DictReader((ROOT / a.bench).open()))
    if a.limit: rows = rows[:a.limit]
    base = ROOT / a.runs / a.arm
    out = ROOT / f"logs/ranking_{a.arm}.jsonl"
    n_ok = n_fb = 0
    with out.open("w") as fh:
        for r in rows:
            d = base / r["name"]
            poses = sorted(d.glob("rank*.pdb"), key=lambda p: int(p.stem[4:])) if d.exists() else []
            if not poses:
                continue
            scores, missing = [], 0
            for p in poses:
                f = compute_features(p, phi, psi)
                if f is None:
                    scores.append(None); missing += 1; continue
                try:
                    scores.append(float(model.predict(np.array([f]))[0]))
                except Exception:      # noqa: BLE001
                    scores.append(None); missing += 1
            # poses our ranker could not featurise go to the BACK, never silently to the front
            order = sorted(range(len(poses)),
                           key=lambda i: (scores[i] is None, scores[i] if scores[i] is not None else 0))
            if missing == len(poses): n_fb += 1
            else: n_ok += 1
            fh.write(json.dumps({"name": r["name"], "arm": a.arm, "order": order,
                                 "scores": scores, "n_poses": len(poses),
                                 "unfeaturised": missing}) + "\n")
    print(f"{a.arm}: ranked {n_ok} complexes, {n_fb} had no featurisable pose")
    print(f"  -> {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
