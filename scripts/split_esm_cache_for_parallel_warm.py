#!/usr/bin/env python3
"""One-time split of the full-param train pool + its 20.8GB ESM cache into N
per-worker chunks, so a parallel graph-cache-warming pass can use all CPU cores
without each worker loading the full cache (which would multiply peak RSS by N
and reintroduce the exact OOM this pipeline just recovered from).

build_dataset()/_esm_cache_path() look for `<csv_stem>_esm_cache.pt` next to the
CSV -- writing N chunk CSVs alongside N matching chunk cache files means each
worker's build_dataset() call transparently picks up only its own slice, no
code changes needed in train_lastlayer.py or inference_utils.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path("/home/igem/unknown_software")
TRAIN_CSV = REPO / "data" / "fullparam_train_pool.csv"
TRAIN_ESM_CACHE = REPO / "data" / "fullparam_train_pool_esm_cache.pt"
OUT_DIR = REPO / "data" / "fullparam_warm_chunks"

N_WORKERS = 24


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    n = len(df)
    print(f"Loaded {n} rows from {TRAIN_CSV}")

    print(f"Loading full ESM cache ({TRAIN_ESM_CACHE}) -- one-time, this is the "
          f"20.8GB load...")
    cache = torch.load(TRAIN_ESM_CACHE, map_location="cpu")
    print(f"Loaded {len(cache)} cached embeddings")

    chunk_size = (n + N_WORKERS - 1) // N_WORKERS
    manifest = []
    for w in range(N_WORKERS):
        start, end = w * chunk_size, min((w + 1) * chunk_size, n)
        if start >= end:
            continue
        chunk_df = df.iloc[start:end]
        chunk_csv = OUT_DIR / f"chunk_{w:02d}.csv"
        chunk_df.to_csv(chunk_csv, index=False)

        chunk_cache_path = OUT_DIR / f"chunk_{w:02d}_esm_cache.pt"
        chunk_cache = {name: cache.get(name) for name in chunk_df["complex_name"]}
        missing = sum(1 for v in chunk_cache.values() if v is None)
        torch.save(chunk_cache, chunk_cache_path)
        print(f"  chunk {w:02d}: rows {start}-{end} ({len(chunk_df)}), "
              f"missing_from_cache={missing} -> {chunk_csv.name}")
        manifest.append(str(chunk_csv))

    # Free the full cache dict before exiting (not strictly needed at process
    # end, but keeps the intent obvious).
    del cache
    print(f"Wrote {len(manifest)} chunks to {OUT_DIR}")
    with open(OUT_DIR / "manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
