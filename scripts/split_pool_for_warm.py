#!/usr/bin/env python3
"""Split a pool (+ its ESM cache) into per-worker chunks for parallel graph warming.

Generalises scripts/split_esm_cache_for_parallel_warm.py, which hardcoded the
fullparam paths and did a full 20.8GB torch.load. Two changes that matter:
  * mmap=True on the source cache, so peak RSS stays small (this box has OOM'd on
    that file before and just took a hypervisor crash)
  * chunks UNIQUE complexes. Graph building is per-complex and the balanced pool
    repeats complexes up to 16x, so chunking the raw rows would build the same graph
    many times over.

build_dataset()/_esm_cache_path() look for `<csv_stem>_esm_cache.pt` beside the CSV,
so each worker transparently loads only its own slice with no code changes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import torch

def main() -> None:
    out_dir = Path(sys.argv[1]); n_workers = int(sys.argv[2])
    pairs = [(Path(a), Path(b)) for a, b in
             (p.split("=") for p in sys.argv[3:])]      # pool.csv=esm_cache.pt
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, caches = [], []
    for csv_p, esm_p in pairs:
        df = pd.read_csv(csv_p)
        frames.append(df)
        caches.append(torch.load(esm_p, map_location="cpu", mmap=True))
        print(f"{csv_p.name}: {len(df)} rows, cache {len(caches[-1])} entries")

    seen, rows, lookup = set(), [], {}
    for df, cache in zip(frames, caches):
        for r in df.to_dict("records"):
            n = r["complex_name"]
            if n in seen:
                continue
            seen.add(n); rows.append(r)
            if n in cache:
                lookup[n] = cache[n]
    uniq = pd.DataFrame(rows)
    print(f"\nunique complexes to warm: {len(uniq)}  (embeddings found: {len(lookup)})")
    missing = len(uniq) - len(lookup)
    if missing:
        print(f"WARNING: {missing} complexes have no cached embedding; they will be "
              f"recomputed by the worker (slow) -- check the source caches")

    size = (len(uniq) + n_workers - 1) // n_workers
    manifest = []
    for w in range(n_workers):
        s, e = w * size, min((w + 1) * size, len(uniq))
        if s >= e:
            continue
        part = uniq.iloc[s:e]
        csv_out = out_dir / f"chunk_{w:02d}.csv"
        part.to_csv(csv_out, index=False)
        sub = {n: [t.clone() for t in lookup[n]] for n in part["complex_name"] if n in lookup}
        torch.save(sub, out_dir / f"chunk_{w:02d}_esm_cache.pt")
        gb = (out_dir / f"chunk_{w:02d}_esm_cache.pt").stat().st_size / 1073741824
        print(f"  chunk {w:02d}: {len(part):5d} complexes, esm {gb:.2f} GB")
        manifest.append(str(csv_out))
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nwrote {len(manifest)} chunks -> {out_dir}")

if __name__ == "__main__":
    main()
