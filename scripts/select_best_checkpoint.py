#!/usr/bin/env python3
"""select_best_checkpoint.py — RAPiDock-style docking-metric checkpoint selection.

RAPiDock selects checkpoints on `valinf_rmsds_backbone_lt2` = fraction of INFERENCE poses
with backbone RMSD < 2A (actual docking success), NOT on val_loss. Every past fine-tune here
wrongly selected on val_loss, which was proven to lie (retrain #1: val_loss dropped while
docking degraded). This implements the correct selection.

For each epoch checkpoint, runs RAPiDock inference (finetuned model) on the held-out valinf set
(data/benchmark_valsel.csv, held-out Propedia, long/vlong-weighted) and computes the DIRECT
docking metric per length bucket. Picks the epoch that best docks LONG/VLONG (the retrain target)
without regressing short/med, and copies it to best_docking.pt.

Usage:
    python scripts/select_best_checkpoint.py \
        --ckpt-dir third_party/RAPiDock_finetuned/finetune_long_phase2_v2 \
        --valsel data/benchmark_valsel.csv --n 50
"""
from __future__ import annotations

import argparse
import csv
import glob
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python"


def score_ckpt(ckpt_dir: Path, ckpt: str, valsel: Path, n: int, infer_dir: Path,
               out: Path) -> dict:
    """Run inference for one checkpoint on valsel; return per-length best_direct stats."""
    cmd = [RAPIDOCK_PY, str(ROOT / "scripts" / "nsweep_coverage.py"),
           "--bench", str(valsel), "--n-max", str(n), "--batch", "8",
           "--infer-dir", str(infer_dir), "--model-dir", str(ckpt_dir), "--ckpt", ckpt,
           "--out", str(out)]
    subprocess.run(cmd, capture_output=True, text=True)
    ppr = out / "per_pose_rmsd.csv"
    if not ppr.exists():
        return {}
    meta = {r["name"]: r.get("length_bucket", "?") for r in csv.DictReader(open(valsel))}
    best = defaultdict(lambda: 1e9)
    for r in csv.DictReader(open(ppr)):
        best[r["name"]] = min(best[r["name"]], float(r["rmsd_direct"]))
    by = defaultdict(list)
    for name, b in best.items():
        if b < 1e9:
            by[meta.get(name, "?")].append(b)
    return {bk: v for bk, v in by.items()}


def valinf_score(stats: dict) -> tuple[float, float]:
    """Return (long/vlong Hit@2A fraction [primary], short/med mean best [regression guard])."""
    lv = stats.get("long", []) + stats.get("very_long", [])
    sm = stats.get("short", []) + stats.get("medium", [])
    hit_lv = sum(1 for x in lv if x <= 2.0) / len(lv) if lv else 0.0
    mean_sm = sum(sm) / len(sm) if sm else float("nan")
    return hit_lv, mean_sm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--valsel", type=Path, default=ROOT / "data" / "benchmark_valsel.csv")
    ap.add_argument("--infer-dir", type=Path,
                    default=ROOT / "third_party" / "RAPiDock_finetuned")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--epochs", type=str, default="all",
                    help="'all' or comma list e.g. 5,8,11,14,17,20")
    args = ap.parse_args()

    cks = sorted(glob.glob(str(args.ckpt_dir / "rapidock_finetuned_epoch*.pt")))
    if args.epochs != "all":
        keep = set(f"{int(e):03d}" for e in args.epochs.split(","))
        cks = [c for c in cks if any(k in c for k in keep)]
    print(f"[select] scoring {len(cks)} checkpoints on valinf set ({args.valsel.name}), N={args.n}")
    print(f"[select] metric = long/vlong Hit@2A (docking success), short/med mean = regression guard\n")

    results = []
    print(f"{'epoch':>6} | {'long best':>9} {'vlong best':>10} | {'L/VL Hit@2A':>11} | {'sm mean':>7}")
    print("-" * 60)
    for c in cks:
        ep = Path(c).stem.split("epoch")[-1]
        stats = score_ckpt(args.ckpt_dir, Path(c).name, args.valsel, args.n,
                           args.infer_dir, Path("/tmp/claude-1000/sel_ep" + ep))
        if not stats:
            print(f"{ep:>6} | (no poses)")
            continue
        hit_lv, mean_sm = valinf_score(stats)
        lm = sum(stats.get("long", [1e9])) / max(len(stats.get("long", [])), 1)
        vm = sum(stats.get("very_long", [1e9])) / max(len(stats.get("very_long", [])), 1)
        results.append((ep, hit_lv, mean_sm, lm, vm))
        print(f"{ep:>6} | {lm:9.2f} {vm:10.2f} | {hit_lv:10.0%} | {mean_sm:7.2f}", flush=True)

    if results:
        # best = max long/vlong Hit@2A, tie-break by lower short/med mean (no regression)
        best = max(results, key=lambda r: (r[1], -(r[2] if r[2] == r[2] else 1e9)))
        print(f"\n[select] BEST epoch = {best[0]}  (long/vlong Hit@2A={best[1]:.0%}, sm_mean={best[2]:.2f})")
        src = args.ckpt_dir / f"rapidock_finetuned_epoch{best[0]}.pt"
        dst = args.ckpt_dir / "best_docking.pt"
        shutil.copy2(src, dst)
        print(f"[select] copied {src.name} -> {dst.name}")
        with open(args.ckpt_dir / "valinf_selection.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "longvlong_hit2A", "shortmed_mean", "long_mean", "vlong_mean"])
            w.writerows(results)


if __name__ == "__main__":
    main()
