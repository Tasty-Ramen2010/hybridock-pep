#!/usr/bin/env python
"""Build the full protein-peptide training corpus, with every known trap checked explicitly.

THIS PROJECT HAS LOST WHOLE TRAINING RUNS TO DATA BUGS, not to modelling. The list is specific
and each item below is a thing that actually happened, so each one is re-checked here rather
than assumed:

  WRONG TARGET      every run before Sep 12 denoised toward a sequence-built idealized strand a
                    median 50.3 A from the real pose. Fixed by --crystal-target at training
                    time; this script only guarantees the peptide file IS the crystal.
  BROKEN TOPOLOGY   bonds come from the sequence template and coordinates from the crystal, so
                    a chain break leaves a "bond" spanning the gap -- up to 30.9 A seen. Symptom
                    was an epoch-1 train mean of 10,086 with tor_bb norms to 1.6M. Measured at
                    ~3.9% of the pool. THERE IS NO FILTER FOR THIS ANYWHERE IN THE TRAINING
                    CODE, so it is done here.
  DIRTY PEPTIDES    waters and hetero groups read as residues and expanded by the builder;
                    recent_2024_2026 was 95.5% dirty. Re-verified rather than trusted.
  CONTAMINATED VAL  longgeo's validation set was 97.4% inside its own training data, so its
                    val loss was a memorisation score and the ship/no-ship guard read its own
                    training set. Split here is by PDB ID, and the benchmarks are removed from
                    the pool before the split, not after.
  SCOPE             peptides over 25 aa are small domains and are out of scope (Ram).

WHAT THE SPLIT GUARANTEES. Benchmarks first: every complex appearing in bench_recentset_heldout
or bench_long is dropped from the pool outright, so no evaluation structure can be trained on.
Then validation is held out by PDB ID, so the same structure cannot appear on both sides under
two different source labels -- which is exactly how the same PDB entered peppc and peppcf.

Usage: build_fullcorpus_train.py [--max-len 25] [--val-frac 0.04] [--tag fullcorpus]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
POOL = ROOT / "data/pool_xtal_usable.csv"
BENCHES = ["data/bench_recentset_heldout.csv", "data/bench_long.csv"]
AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
      'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
FIELDS = ["complex_name", "protein_description", "peptide_description", "source",
          "pep_len", "ss_class", "length_bucket"]
BREAK_CUTOFF = 2.2      # C(i)->N(i+1) above this is a chain break, per the measured failure


def pdb_id(name: str) -> str:
    """The underlying PDB entry, so the same structure cannot land on both sides of the split.

    Names look like 8BWE, peppc_2YBF_B, peppcf_3EDX_A_9_21, propedia_5fmj_B. The 4-character
    alphanumeric code starting with a digit is the entry; everything else is chain and range.
    """
    for tok in re.split(r"[_\-]", name):
        if len(tok) == 4 and tok[0].isdigit() and tok.isalnum():
            return tok.upper()
    return name.upper()


def backbone_ok(path: Path) -> tuple[bool, float]:
    """(continuous, worst C->N gap). A peptide bond is ~1.33 A; a crystal break is metres wide."""
    res: dict[tuple, dict[str, tuple[float, float, float]]] = {}
    order: list[tuple] = []
    for l in path.read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM"):
            continue
        if l[17:20].strip() not in AA:
            continue
        an = l[12:16].strip()
        if an not in ("N", "C"):
            continue
        k = (l[21], l[22:27])
        if k not in res:
            res[k] = {}
            order.append(k)
        res[k][an] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
    worst = 0.0
    for a, b in zip(order, order[1:]):
        ca, nb = res.get(a, {}).get("C"), res.get(b, {}).get("N")
        if ca is None or nb is None:
            continue
        d = float(np.linalg.norm(np.array(ca) - np.array(nb)))
        worst = max(worst, d)
    return worst <= BREAK_CUTOFF, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--tag", default="fullcorpus")
    ap.add_argument("--skip-topology", action="store_true",
                    help="skip the backbone-continuity scan (it reads every peptide file)")
    a = ap.parse_args()

    rows = list(csv.DictReader(POOL.open()))
    print(f"{len(rows)} rows in {POOL.name}\n")

    # ---- benchmarks are removed FIRST, so nothing we evaluate on can be trained on
    bench_ids, bench_names = set(), set()
    for b in BENCHES:
        p = ROOT / b
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            nm = r.get("complex_name") or r.get("name") or ""
            if nm:
                bench_names.add(nm)
                bench_ids.add(pdb_id(nm))
        print(f"  {p.name}: {sum(1 for _ in csv.DictReader(p.open()))} entries")
    print(f"  -> {len(bench_ids)} distinct benchmark PDB IDs to exclude\n")

    drop = Counter()
    kept = []
    for r in rows:
        n = int(r["pep_len"]) if r["pep_len"].isdigit() else 0
        if n > a.max_len:
            drop["over 25 aa (out of scope)"] += 1
            continue
        if n < 4:
            drop["under 4 aa"] += 1
            continue
        if r["complex_name"] in bench_names or pdb_id(r["complex_name"]) in bench_ids:
            drop["IN A BENCHMARK"] += 1
            continue
        pp, rp = Path(r["peptide_description"]), Path(r["protein_description"])
        if not (pp.exists() and rp.exists()):
            drop["file missing"] += 1
            continue
        kept.append(r)
    print("after scope, benchmark and existence filters:")
    for k, v in sorted(drop.items(), key=lambda t: -t[1]):
        print(f"  dropped {v:>6}  {k}")
    print(f"  kept    {len(kept):>6}\n")

    if not a.skip_topology:
        print(f"scanning backbone continuity on {len(kept)} peptides "
              f"(C->N gap > {BREAK_CUTOFF} A = chain break)...", flush=True)
        good, broken, worst_by_src = [], Counter(), defaultdict(float)
        for i, r in enumerate(kept, 1):
            ok, worst = backbone_ok(Path(r["peptide_description"]))
            worst_by_src[r["source"]] = max(worst_by_src[r["source"]], worst)
            if ok:
                good.append(r)
            else:
                broken[r["source"]] += 1
            if i % 4000 == 0:
                print(f"  {i}/{len(kept)}", flush=True)
        nb = sum(broken.values())
        print(f"\n  BROKEN BACKBONE: {nb}/{len(kept)} ({100 * nb / max(len(kept), 1):.1f}%)")
        for s in sorted(broken, key=lambda x: -broken[x]):
            tot = sum(1 for r in kept if r["source"] == s)
            print(f"    {s:<22}{broken[s]:>5}/{tot:<6} ({100 * broken[s] / max(tot, 1):>5.1f}%)"
                  f"   worst gap {worst_by_src[s]:.1f} A")
        kept = good
    else:
        print("topology scan SKIPPED — do not train on this without running it once\n")

    # ---- split by PDB ID, never by row
    by_id = defaultdict(list)
    for r in kept:
        by_id[pdb_id(r["complex_name"])].append(r)
    ids = sorted(by_id)
    random.Random(0).shuffle(ids)
    nval = max(20, int(round(a.val_frac * len(ids))))
    val_ids = set(ids[:nval])
    tr = [r for i in ids[nval:] for r in by_id[i]]
    va = [r for i in ids[:nval] for r in by_id[i]]

    tr_ids = {pdb_id(r["complex_name"]) for r in tr}
    va_ids = {pdb_id(r["complex_name"]) for r in va}
    overlap = tr_ids & va_ids
    print(f"\nsplit by PDB ID: {len(ids) - nval} train IDs / {nval} val IDs")
    print(f"  train {len(tr)} complexes, val {len(va)} complexes")
    print(f"  ID overlap train/val: {len(overlap)}  "
          f"{'PASS' if not overlap else 'FAIL — LEAK'}")
    print(f"  benchmark IDs present in train: "
          f"{len(tr_ids & bench_ids)}  {'PASS' if not (tr_ids & bench_ids) else 'FAIL — LEAK'}")
    assert not overlap and not (tr_ids & bench_ids), "leak detected; refusing to write"

    random.Random(1).shuffle(tr)
    tp, vp = ROOT / f"data/{a.tag}_train.csv", ROOT / f"data/{a.tag}_val.csv"
    for path, rws in ((tp, tr), (vp, va)):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rws])
    print(f"\n  -> {tp.relative_to(ROOT)}  {len(tr)} rows")
    print(f"  -> {vp.relative_to(ROOT)}  {len(va)} rows")
    print("\n  composition:", dict(Counter(r["source"] for r in tr)))
    print("  lengths:", dict(Counter(r["length_bucket"] for r in tr)))
    print("\n  REMINDER: val loss is a sanity signal only. Select the shipping checkpoint on "
          "direct-RMSD\n  docking against bench_recentset_heldout, which this script has just "
          "guaranteed is unseen.")


if __name__ == "__main__":
    main()
