#!/usr/bin/env python
"""Dock the full 18x18 Coventry grid with our finetuned RAPiDock.

For every (peptide, binder) pair in data/coventry_pairs.csv we generate N peptide poses
in that binder's frame.  Nothing about the pair's measured affinity, and nothing about
which pairs are cognate, enters here -- every one of the 324 cells is run identically.

RECEPTOR.  The whole designed binder is passed, uncropped.  RAPiDock's training
convention crops the receptor to the pocket, but that crop is defined by the crystal
peptide, which we do not have and must not assume.  These binders are 96-253 aa, i.e.
about the size of a cropped pocket anyway, so handing over the entire protein and
letting the model find the groove is both the honest option and a workable one.

MODEL.  Upstream RAPiDock code path + our epoch-10 finetuned checkpoint -- the
`hybridock_ft_shipped` arm, the best-characterised configuration we have (RecentSet 345:
99.1% acceptable / 91.6% medium / 38.6% high, DockQ capri_peptide, oracle best-of-24).

Resumable: a pair whose output directory already holds poses is skipped.

Usage: coventry_dock_grid.py [--arm NAME] [--n 24] [--steps 16] [--limit 0] [--cognate-only]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import os
# The repo root and the rapidock interpreter differ between the workstation, the rtx6000 and the
# DGX, so both are overridable. Hardcoding them made this script die on the DGX with
# PermissionError on /home/igem before it docked a single cell.
ROOT = Path(os.environ.get("HDP_ROOT", "/home/igem/unknown_software"))
RAPIDOCK_PY = os.environ.get(
    "HDP_RAPIDOCK_PY", "/home/igem/miniconda3/envs/rapidock/bin/python")
# Our finetuned checkpoints carry a cross_type_embedding tensor that the UPSTREAM model class
# does not define, so an unpatched upstream tree cannot load them at all (strict=True). The local
# workstation's upstream copy is patched; a fresh clone on another box is not, which is why this
# has to be selectable rather than assumed.
INFER_DIR = ROOT / os.environ.get("HDP_INFER_DIR", "third_party/RAPiDock")
MODEL_DIR = ROOT / os.environ.get("COVENTRY_MODEL_DIR",
                                  "third_party/RAPiDock_finetuned/longft_tanh")
CKPT = os.environ.get("COVENTRY_CKPT", "rapidock_finetuned_epoch010.pt")


def run_pair(name: str, receptor: Path, seq: str, out_root: Path,
             n: int, steps: int, batch: int) -> tuple[int, float, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(["complex_name", "protein_description", "peptide_description"])
        w.writerow([name, str(receptor), seq])
        csv_path = tf.name
    cmd = [
        RAPIDOCK_PY, str(INFER_DIR / "inference.py"),
        "--protein_peptide_csv", csv_path,
        "--out", str(out_root), "--output_dir", str(out_root),
        "--model_dir", str(MODEL_DIR), "--ckpt", CKPT,
        "--N", str(n), "--batch_size", str(batch),
        "--inference_steps", str(steps), "--actual_steps", str(steps),
        "--no_final_step_noise", "--conformation_partial", "1:1:1",
        "--scoring_function", "none", "--confidence_ckpt", "null", "--cpu", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(INFER_DIR), capture_output=True, text=True)
    os.unlink(csv_path)
    poses = glob.glob(str(out_root / name / "rank*.pdb"))
    return len(poses), time.time() - t0, proc.stderr[-300:] if not poses else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hybridock_ft")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cognate-only", action="store_true",
                    help="run only the 18 diagonal cells (pipeline smoke test)")
    ap.add_argument("--binders", default=str(ROOT / "datasets/coventry/binders"),
                    help="directory of binder structures; swap for the AF3 set")
    a = ap.parse_args()

    bdir = Path(a.binders)
    out_root = ROOT / "runs/coventry" / a.arm
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = list(csv.DictReader(open(ROOT / "data/coventry_pairs.csv")))
    if a.cognate_only:
        pairs = [p for p in pairs if p["cognate"] == "1"]
    todo = []
    for p in pairs:
        rec = bdir / f"{p['binder_name']}.pdb"
        name = f"{p['peptide']}__{p['binder_label']}"
        if not rec.exists():
            print(f"  MISSING receptor {rec.name} for {name}", flush=True)
            continue
        if glob.glob(str(out_root / name / "rank*.pdb")):
            continue
        todo.append((name, rec, p["peptide_seq"]))
    if a.limit:
        todo = todo[:a.limit]

    print(f"arm={a.arm} receptors={bdir}", flush=True)
    print(f"model={MODEL_DIR.name}/{CKPT} via {INFER_DIR.name}", flush=True)
    print(f"{len(todo)} pairs to dock (N={a.n}, steps={a.steps})", flush=True)

    t0, done, failed = time.time(), 0, []
    for k, (name, rec, seq) in enumerate(todo, 1):
        npose, dt, err = run_pair(name, rec, seq, out_root, a.n, a.steps, a.batch)
        if npose:
            done += 1
        else:
            failed.append(name)
            print(f"  [{k}/{len(todo)}] {name:22s} NO POSES ({dt:.0f}s) {err[:160]}", flush=True)
        if k % 5 == 0 or k == 1 or npose == 0:
            el = time.time() - t0
            print(f"  [{k}/{len(todo)}] {name:22s} {npose:3d} poses {dt:5.1f}s  "
                  f"| {el / k:5.1f}s/pair  ETA {(len(todo) - k) * el / k / 60:5.1f} min",
                  flush=True)

    print(f"\ndocked {done}/{len(todo)}   failed: {failed or 'none'}")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    print("COVENTRY_DOCK_DONE")


if __name__ == "__main__":
    main()
