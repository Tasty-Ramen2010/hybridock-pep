#!/usr/bin/env python
"""Dock every peptide against every receptor in a block — the training set that actually matches.

WHY THIS AND NOT THREADING.  The threaded selectivity model reaches AUC 0.979 on its own synthetic
benchmark and 0.598 on the Coventry grid, against plain centred ref2015's 0.667.  The gap is a
distribution shift and it was measured, not guessed: threading puts every sequence on a crystal
backbone in the right place, while a docked pool is 79-81% backwards, and on docked poses i_fa_rep
is 98.7% of the ref2015 total's variance while scoring at chance on its own (AUC 0.515).  Interface
chemistry computed on a pose that is in the wrong place describes an interface that is not there.

So the fix is not a cleverer feature set.  It is to train on the same kind of pose we score at
test time: DOCKED, with the same generator, the same N, and the same best-of-N aggregation.

MATCHING THE TEST DISTRIBUTION EXACTLY MATTERS MORE THAN SCALE.  This uses longft_tanh epoch010 --
the checkpoint that produced the Coventry poses -- deliberately, not the newer longgeo retrain.
If the generator differs between training and test, the pose-error distribution differs, and that
is the exact mistake being corrected here.

One block of n complexes gives n x n docking runs; the diagonal is the crystallographic pair and
the off-diagonal is a peptide on a receptor it was never selected for. N is lower than the grid's
24 because the cost is quadratic and the aggregation, not the depth, is what has to match.

Usage: sel_dock_blocks.py [--blocks 12] [--members 8] [--n 8]
Output: runs/sel_dock/<block>_<i>_<j>/rank*.pdb, logs/sel_dock.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
BLOCKS = ROOT / "data/sel_blocks.json"
OUT_ROOT = ROOT / "runs/sel_dock"
LOG = ROOT / os.environ.get("SEL_DOCK_LOG", "logs/sel_dock.jsonl")
RAPIDOCK_PY = os.environ.get("HDP_RAPIDOCK_PY",
                             "/home/igem/miniconda3/envs/rapidock/bin/python")
INFER_DIR = ROOT / "third_party/RAPiDock"
#: the checkpoint that produced the Coventry poses -- see the module docstring
MODEL_DIR = ROOT / "third_party/RAPiDock_finetuned/longft_tanh"
CKPT = "rapidock_finetuned_epoch010.pt"


def dock(name: str, receptor: Path, seq: str, n: int, steps: int = 16) -> tuple[int, float]:
    dest = OUT_ROOT / name
    if list(dest.glob("rank*.pdb")):
        return len(list(dest.glob("rank*.pdb"))), 0.0
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write("complex_name,protein_description,peptide_description\n")
        tf.write(f"{name},{receptor.resolve()},{seq}\n")
        csv_path = tf.name
    try:
        subprocess.run(
            [RAPIDOCK_PY, str(INFER_DIR / "inference.py"),
             "--protein_peptide_csv", csv_path,
             "--out", str(dest), "--output_dir", str(dest),
             "--model_dir", str(MODEL_DIR), "--ckpt", CKPT,
             "--N", str(n), "--batch_size", str(min(n, 8)),
             "--inference_steps", str(steps), "--actual_steps", str(steps),
             "--no_final_step_noise", "--conformation_partial", "1:1:1",
             "--scoring_function", "none", "--confidence_ckpt", "null", "--cpu", "4"],
            capture_output=True, text=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired:
        pass
    finally:
        os.unlink(csv_path)
    # RAPiDock nests its output one level deep when --out and --output_dir agree
    poses = list(dest.rglob("rank*.pdb"))
    for p in poses:
        if p.parent != dest:
            target = dest / p.name
            if not target.exists():
                p.replace(target)
    return len(list(dest.glob("rank*.pdb"))), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help=("run shard I of N. A RAPiDock pass holds ~3 GB, so three fit on the "
                          "12 GB card and the quadratic cost drops from ~4 h to ~1.5 h."))
    args = ap.parse_args()

    blocks = json.loads(BLOCKS.read_text())[: args.blocks]
    # a cell finished by ANY shard counts as done, so shards never repeat each other's work
    done = set()
    for f in sorted(ROOT.glob("logs/sel_dock*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    jobs = []
    for b in blocks:
        m = b["members"][: args.members]
        for i, pi in enumerate(m):
            for j, pj in enumerate(m):
                name = f"b{b['block_id']}_{i}_{j}"
                if name in done:
                    continue
                jobs.append((name, b["block_id"], i, j, Path(pj["receptor"]), pi["seq"],
                             pi["pdb"], pj["pdb"]))
    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        jobs = jobs[si::sn]
    print(f"{len(jobs)} docking runs in shard {si}/{sn} ({len(done)} done grid-wide) — "
          f"{args.blocks} blocks x {args.members}x{args.members}, N={args.n}", flush=True)

    t0 = time.time()
    with LOG.open("a") as fh:
        for k, (name, blk, i, j, rec, seq, pdb_i, pdb_j) in enumerate(jobs, 1):
            npose, dt = dock(name, rec, seq, args.n)
            fh.write(json.dumps({"name": name, "block": blk, "pep_idx": i, "rec_idx": j,
                                 "cognate": int(i == j), "seq": seq, "receptor": str(rec),
                                 "pep_pdb": pdb_i, "rec_pdb": pdb_j,
                                 "n_poses": npose, "seconds": round(dt, 1)}) + "\n")
            fh.flush()
            if k % 10 == 0 or k == 1:
                rate = k / max(time.time() - t0, 1)
                print(f"  [{k}/{len(jobs)}] {name:14s} {npose:2d} poses  "
                      f"{dt:5.1f}s  eta {(len(jobs) - k) / max(rate, 1e-9) / 60:.0f} min",
                      flush=True)
    print("SEL_DOCK_DONE")


if __name__ == "__main__":
    main()
