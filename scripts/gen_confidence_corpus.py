#!/usr/bin/env python3
"""Generate a large labelled pose corpus for CONFIDENCE-MODEL co-training.

Why this exists
---------------
project_ranker_ceiling_jun10 measured oracle Hit@2A = 49.1% against a tau~0.14
ranking ceiling, and named the fix: "DFMDock-style dual-head co-training -- train
the encoder WITH RMSD labels", because "post-hoc rescoring with frozen embeddings
hits a wall". RAPiDock ships a full ConfidenceModel (models/model.py:38) that we
have never trained -- every run passes --confidence_ckpt null.

scripts/train_confidence_model.py exists but hard-freezes the encoder, i.e. it
implements exactly the frozen-embedding approach the memory says cannot work. The
un-run experiment is co-training with the encoder unfrozen, and that needs far more
than bench300's 240 complexes x 5 poses (~1200 labelled poses).

This script produces that corpus: sample poses from the PRETRAINED generator on
TRAINING-pool complexes (disjoint from the benchmark/test sets) and label every pose
with its Ca RMSD to the crystal peptide.

Shardable so local/dgx/rtx6000 can run disjoint slices concurrently, and resumable
so a crash or a machine dropping out costs only the in-flight complex.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsweep_coverage import ca_rmsd  # identical metric to every eval this project runs

RAPIDOCK_PY = os.environ.get("RAPIDOCK_PY", sys.executable)


def already_done(out_jsonl: Path) -> set[str]:
    done = set()
    if out_jsonl.exists():
        with open(out_jsonl) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["name"])
                except Exception:
                    continue
    return done


def run_complex(name, receptor, seq, n, steps, batch, pose_root, infer_dir,
                model_dir, ckpt):
    """Sample n poses and label each with Ca RMSD to the crystal peptide."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(["complex_name", "protein_description", "peptide_description"])
        w.writerow([name, receptor, seq])
        csv_path = tf.name
    # CLAUDE.md 7: absolute paths only across the RAPiDock boundary.
    pose_root = str(Path(pose_root).resolve())
    cmd = [
        RAPIDOCK_PY, str(Path(infer_dir) / "inference.py"),
        "--protein_peptide_csv", csv_path,
        "--out", pose_root, "--output_dir", pose_root,
        "--model_dir", str(model_dir), "--ckpt", ckpt,
        "--N", str(n), "--batch_size", str(batch),
        "--inference_steps", str(steps), "--actual_steps", str(steps),
        "--no_final_step_noise",
        "--conformation_partial", "1:1:1",
        "--scoring_function", "none", "--confidence_ckpt", "null",
        "--cpu", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(infer_dir), capture_output=True, text=True)
    dt = time.time() - t0
    os.unlink(csv_path)

    poses = sorted(glob.glob(str(Path(pose_root) / name / "rank*.pdb")))
    if not poses:
        return None, dt, (proc.stderr or "")[-300:]

    crystal = receptor.replace("_protein_pocket.pdb", "_peptide.pdb")
    labelled = []
    for p in poses:
        r = ca_rmsd(crystal, p)
        if r is None:
            continue
        direct, superposed = r
        labelled.append({"pose": p, "rmsd": round(direct, 4),
                         "rmsd_superposed": round(superposed, 4)})
    if not labelled:
        return None, dt, "rmsd failed for every pose"
    return labelled, dt, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-csv", required=True, help="training pool CSV (NOT a test set)")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--pose-root", required=True)
    ap.add_argument("--infer-dir", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--ckpt", default="rapidock_local.pt")
    ap.add_argument("--n", type=int, default=16, help="poses per complex")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = no cap")
    ap.add_argument("--protein-col", default="protein_description")
    ap.add_argument("--peptide-col", default="peptide_description")
    ap.add_argument("--name-col", default="complex_name")
    a = ap.parse_args()

    infer_dir = str(Path(a.infer_dir).expanduser().resolve())
    model_dir = a.model_dir or str(Path(infer_dir) / "train_models" / "CGTensorProductEquivariantModel")
    model_dir = str(Path(model_dir).expanduser().resolve())
    out_jsonl = Path(a.out_jsonl).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    Path(a.pose_root).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(a.pool_csv)))
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard]
    if a.limit:
        rows = rows[: a.limit]

    done = already_done(out_jsonl)
    todo = [r for r in rows if r[a.name_col] not in done]
    print(f"[gen] shard {a.shard}/{a.num_shards}: {len(rows)} assigned, "
          f"{len(done)} already done, {len(todo)} to do, N={a.n}", flush=True)

    n_ok = n_fail = n_pose = 0
    t_start = time.time()
    with open(out_jsonl, "a") as fh:
        for i, r in enumerate(todo):
            name = r[a.name_col]
            labelled, dt, err = run_complex(
                name, r[a.protein_col], r[a.peptide_col], a.n, a.steps, a.batch,
                a.pose_root, infer_dir, model_dir, a.ckpt)
            if labelled is None:
                n_fail += 1
                print(f"[{i+1}/{len(todo)}] {name} FAILED ({dt:.0f}s): {err}", flush=True)
                continue
            best = min(x["rmsd"] for x in labelled)
            nn = sum(1 for x in labelled if x["rmsd"] <= 2.0)
            fh.write(json.dumps({"name": name, "poses": labelled}) + "\n")
            fh.flush()
            n_ok += 1
            n_pose += len(labelled)
            rate = (time.time() - t_start) / max(n_ok, 1)
            print(f"[{i+1}/{len(todo)}] {name} n={len(labelled)} dt={dt:.0f}s "
                  f"best={best:.2f}A nn={nn} | ok={n_ok} poses={n_pose} "
                  f"avg={rate:.0f}s/cx", flush=True)

    print(f"[gen] DONE shard {a.shard}: ok={n_ok} failed={n_fail} poses={n_pose}", flush=True)


if __name__ == "__main__":
    main()
