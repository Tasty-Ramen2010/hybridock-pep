#!/usr/bin/env python3
"""Build a SS x LENGTH balanced training pool for from-scratch RAPiDock training.

Motivation (measured, Sep-10):
    ss_class   short   med  long  vlong   total
    HELIX       3323  4583  1176   1216   10298  (50%)
    SHEET       3140   785   146    112    4183  (20%)
    PPII        2851  2111   416    306    5684  (28%)
The two empirically worst-performing regimes -- beta-sheet, and long/very-long --
are exactly the two rarest cells (SHEET-vlong = 112 = 0.54% of the pool, while
HELIX-vlong has 11x more). The docking length cliff measured today
(1.81A at <=8mer -> 9.79A at 17+mer) tracks that imbalance directly.

This builder equalizes the 12 (SS x length) cells by weighted repetition, so the
failure regimes get first-class representation instead of being drowned out.

Repetition is honest augmentation *for a diffusion model specifically*: each
exposure draws a different noise level, rotation and translation, so a repeated
complex is not a repeated training example the way it would be in plain supervised
classification. It is still not free -- 112 unique structures cannot become 3000 --
so the per-cell repeat factor is capped and reported rather than hidden.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SS_KEEP = ["HELIX", "SHEET", "PPII"]

# Complexes that trigger a CUDA device-side assert and poison the context, killing the
# whole run (one assert -> every later sample in the epoch fails). Identified Sep 8-9;
# they have silently reappeared twice since, each time because a pool was rebuilt from an
# unfiltered source, so the exclusion lives HERE rather than in a downstream CSV.
POISON = {"peppc_5XOS_C", "peppcf_4WA6_A_11_21", "peppcf_6HLA_A_82_89"}
BUCKETS = ["short", "med", "long", "vlong"]


def bucket(L: int) -> str:
    return "short" if L <= 8 else "med" if L <= 12 else "long" if L <= 16 else "vlong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="pool_ss_labelled.csv")
    ap.add_argument("--extra", nargs="*", default=[], help="extra source CSVs to merge")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--repeat", type=float, default=2.0,
                    help="average times each complex appears per epoch")
    ap.add_argument("--max-repeat", type=int, default=16,
                    help="cap on per-complex repetition for the rarest cells")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)

    rows = [r for r in csv.DictReader(open(a.labelled))
            if r["ss_class"] in SS_KEEP and r["complex_name"] not in POISON]
    seen = {r["complex_name"] for r in rows}

    # merge extra sources (already-labelled sheet sets etc.), skipping duplicates
    n_extra = 0
    for f in a.extra:
        try:
            for r in csv.DictReader(open(f)):
                name = r.get("complex_name") or r.get("name")
                rec = r.get("protein_description") or r.get("receptor")
                pep = r.get("peptide_description") or r.get("peptide_pdb")
                ss = r.get("ss_class")
                if not (name and rec and pep) or name in seen or name in POISON:
                    continue
                if not (Path(rec).exists() and Path(pep).exists()):
                    continue
                try:
                    L = int(r.get("pep_len") or 0)
                except ValueError:
                    continue
                if not L:
                    continue
                if ss not in SS_KEEP:
                    ss = "SHEET"   # these files are sheet-mined sets by construction
                rows.append({"complex_name": name, "protein_description": rec,
                             "peptide_description": pep, "source": Path(f).stem,
                             "pep_len": str(L), "ss_class": ss,
                             "length_bucket": bucket(L)})
                seen.add(name)
                n_extra += 1
        except FileNotFoundError:
            continue
    print(f"base labelled: {len(rows)-n_extra}  + merged extra: {n_extra}  = {len(rows)}")

    # ---- cells ----
    cells: dict[tuple, list] = defaultdict(list)
    for r in rows:
        cells[(r["ss_class"], r["length_bucket"])].append(r)

    # ---- held-out val (stratified, taken BEFORE any repetition so no leakage) ----
    train_cells, val_rows = defaultdict(list), []
    for k, v in cells.items():
        v = v[:]
        random.shuffle(v)
        n_val = max(1, int(len(v) * a.val_frac)) if len(v) >= 20 else 0
        val_rows += v[:n_val]
        train_cells[k] = v[n_val:]

    # ---- balance target ----
    total_train = sum(len(v) for v in train_cells.values())
    n_cells = len([k for k in train_cells if train_cells[k]])
    target = int(round(total_train * a.repeat / max(n_cells, 1)))
    print(f"\ntrain complexes: {total_train} | cells: {n_cells} | "
          f"target per cell: {target} (repeat={a.repeat}x avg)")

    out, comp = [], []
    for ss in SS_KEEP:
        for b in BUCKETS:
            v = train_cells.get((ss, b), [])
            if not v:
                continue
            cap = len(v) * a.max_repeat
            take = min(target, cap)
            reps = take / len(v)
            picked = []
            full = int(take // len(v))
            for _ in range(full):
                picked += v
            rem = take - len(picked)
            if rem > 0:
                picked += random.sample(v, rem)
            out += picked
            comp.append((f"{ss}-{b}", len(v), len(picked), reps,
                         "CAPPED" if take < target else ""))

    random.shuffle(out)
    fields = ["complex_name", "protein_description", "peptide_description",
              "source", "pep_len", "ss_class", "length_bucket"]
    for p, rs in ((a.out_train, out), (a.out_val, val_rows)):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rs)

    print(f"\n{'cell':16}{'unique':>8}{'per-epoch':>11}{'repeat':>9}  note")
    print("-" * 52)
    for name, uniq, n, reps, note in comp:
        print(f"{name:16}{uniq:8d}{n:11d}{reps:9.1f}x  {note}")
    print("-" * 52)
    print(f"{'TOTAL':16}{total_train:8d}{len(out):11d}")
    vc = Counter((r["ss_class"], r["length_bucket"]) for r in val_rows)
    print(f"\nheld-out val: {len(val_rows)} complexes across {len(vc)} cells "
          f"(stratified, split before repetition -> no train/val leakage)")
    print(f"train -> {a.out_train}\nval   -> {a.out_val}")


if __name__ == "__main__":
    main()
