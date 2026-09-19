#!/usr/bin/env python3
"""Warm the on-disk per-complex graph cache for the full-param training pool,
BEFORE launching the actual 350-epoch training run.

Why this exists (see conversation / docs postmortem for full detail):
InferenceDataset.get() (utils/inference_utils.py) builds each complex's graph
once and caches it to disk forever next to the source protein PDB. During the
first full-param launch, that one-time build happened INTERLEAVED with epoch 1's
training loop -- with ~22,503 complexes (20,564 train + 1,939 val) and no
pre-existing disk cache for most of them, that meant 5-70s/complex x thousands
of misses, buried inside epoch 1 with zero visible progress, while a SEPARATE
bug (self.lm_embeddings held in memory AND re-baked into every cached graph,
never freed) roughly doubled peak RSS -- together these drove the box to a
swap-exhausted OOM edge after 3.5h with zero epochs completed (killed manually).

This script does the same one-time graph-cache build standalone:
  - visible per-complex progress + ETA, not silently buried in epoch 1
  - resumable: re-checks the disk cache before rebuilding anything
  - explicitly drops each built graph from the in-process _mem_cache right after
    get() returns -- warming only needs get()'s disk-write side effect, not to
    hold all 22,503 graphs in RAM simultaneously
  - relies on the inference_utils.py fix (self.lm_embeddings[idx] = None after
    baking) to keep the one-time ESM-cache load (~20.8GB train / ~1.8GB val)
    from being duplicated as graphs accumulate

Run with the rapidock env's python3. CPU-only (esm_device is irrelevant here
since ESM embeddings come from the precomputed cache, not live computation).
"""
from __future__ import annotations
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # WSL2: unpinned OMP threads have
                                                  # caused 1300x slowdowns on this box

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from argparse import Namespace  # noqa: E402

REPO = Path("/home/igem/unknown_software")
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock_finetuned"))
from train_lastlayer import build_dataset  # noqa: E402

TRAIN_CSV = REPO / "data" / "fullparam_train_pool.csv"
VAL_CSV = REPO / "data" / "fullparam_val_pool.csv"
MODEL_YML = (REPO / "third_party" / "RAPiDock_finetuned" / "train_models"
             / "CGTensorProductEquivariantModel" / "model_parameters.yml")
# Scratch working dir -- graph cache itself is written next to each source PDB
# (persistent, shared with the real training run), NOT here. This is only used
# for the transient per-complex peptide-init PDBs get() writes along the way.
OUT_DIR = REPO / "third_party" / "RAPiDock_finetuned" / "finetune_fullparam_350ep_aug24"
LOG = REPO / "logs" / "fullparam_cache_warm.log"

CHUNK = 200  # progress/gc cadence


def log(msg: str, log_path: Path = LOG) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as fh:
        fh.write(line + "\n")


def warm(csv_path: Path, label: str, log_path: Path = LOG, out_dir: Path = OUT_DIR) -> None:
    log(f"=== warming {label} ({csv_path}) ===", log_path)
    with open(MODEL_YML) as f:
        params = yaml.safe_load(f)
    model_args = Namespace(**params)

    ds = build_dataset(str(csv_path), model_args, str(out_dir), esm_device="cpu")
    n = ds.len()
    log(f"{label}: {n} complexes, starting build/verify pass", log_path)

    t0 = time.time()
    n_ok = n_fail = n_precached = 0
    first_fail_logged = 0
    for i in range(n):
        already_cached = os.path.exists(
            os.path.join(os.path.dirname(ds.protein_descriptions[i]),
                         f"{ds.complex_names[i]}_graph_v1.pt")
        )
        try:
            ds.get(i)
        except Exception as exc:
            n_fail += 1
            if first_fail_logged < 20:
                log(f"  FAILED [{i}] {ds.complex_names[i]}: {exc}", log_path)
                first_fail_logged += 1
        else:
            n_ok += 1
            if already_cached:
                n_precached += 1
        finally:
            # Drop from the in-process cache immediately -- we only need the
            # disk-write side effect inside get(), not to retain all graphs.
            if hasattr(ds, "_mem_cache") and i in ds._mem_cache:
                del ds._mem_cache[i]

        if (i + 1) % CHUNK == 0 or (i + 1) == n:
            gc.collect()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = n - (i + 1)
            eta_h = remaining / rate / 3600 if rate > 0 else float("nan")
            log(f"  [{i+1}/{n}] ok={n_ok} fail={n_fail} precached={n_precached} "
                f"rate={rate:.2f}/s eta={eta_h:.1f}h", log_path)

    log(f"=== {label} done: {n_ok}/{n} ok ({n_precached} were already cached), "
        f"{n_fail} failed, {time.time()-t0:.0f}s total ===", log_path)

    del ds
    gc.collect()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                     help="Worker mode: warm exactly this one CSV (paired with a "
                          "<stem>_esm_cache.pt chunk file next to it) instead of the "
                          "default val-then-train sequence.")
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--log", type=str, default=None,
                     help="Worker mode: write to this log instead of the shared "
                          "fullparam_cache_warm.log (avoids interleaved writes across "
                          "parallel workers).")
    args = ap.parse_args()

    if args.csv is not None:
        # Worker mode -- one chunk, own log, own scratch out_dir (so concurrent
        # workers' os.makedirs/cp calls for peptide-init PDBs don't collide).
        log_path = Path(args.log) if args.log else LOG
        label = args.label or Path(args.csv).stem
        worker_out_dir = OUT_DIR.parent / f"{OUT_DIR.name}_{label}"
        warm(Path(args.csv), label, log_path=log_path, out_dir=worker_out_dir)
        log(f"=== WORKER {label} DONE ===", log_path)
    else:
        # Default: val first (smaller ESM cache, ~1.8GB -- cheap sanity check the
        # pipeline works before committing to the much bigger train pool).
        warm(VAL_CSV, "val pool")
        warm(TRAIN_CSV, "train pool")
        log("=== ALL WARMING DONE ===")
