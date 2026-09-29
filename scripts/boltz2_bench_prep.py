#!/usr/bin/env python
"""Phase A of the Boltz-2 arm: fetch receptor MSAs and write one YAML per complex. No GPU.

WHY THIS IS SPLIT FROM THE PREDICTION. Boltz's --use_msa_server fetches MSAs inline, which
serialises ~330 network round trips against GPU time and refetches everything on any re-run.
Worse, boltz calls compute_msa per TARGET, and with two protein chains it submits a PAIRED run
as well as an unpaired one, so 387 targets would be 774 submissions against a free shared
service. Doing it once here, keyed by receptor sequence hash, makes it 333 unpaired submissions
in batches and makes every later re-run free and offline.

THE PEPTIDE GETS NO MSA, deliberately. A 5-25mer has no meaningful alignment, single-sequence is
what peptide co-folding conventionally does, and leaving the peptide on `msa: empty` is also what
keeps boltz from triggering the paired-MSA path.

LEAKAGE, STATED UP FRONT because the figure has to carry it: Boltz-2's training cutoff is
2023-06-01 and this bench is 2020 (33) / 2021 (223) / 2022 (98) / 2023 (4). Essentially every
complex here is inside Boltz-2's training window. This arm is run because it was asked for and
because the number is interesting, but it is an upper bound on a memorised set, not a fair
head-to-head, and it must be labelled that way wherever it is plotted.

Usage: boltz2_bench_prep.py [--bench data/bench_balanced_length.csv] [--batch 25]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/igem/miniconda3/envs/boltz-env/lib/python3.11/site-packages")

ROOT = Path("/home/igem/unknown_software")
# --out lets a second bench reuse the SAME msa/ directory: MSAs are keyed by receptor
# sequence hash, so any receptor shared between benches is fetched once, not twice.
import os as _os
OUT = ROOT / _os.environ.get("BOLTZ_BENCH_DIR", "datasets/boltz2_bench")
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path) -> str:
    s, seen = [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        s.append(AA3.get(l[17:20].strip(), 'X'))
    return ''.join(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default=str(ROOT / "data/bench_balanced_length.csv"))
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--yaml-sub", default="yaml", help="subdir for the specs")
    a = ap.parse_args()

    (OUT / a.yaml_sub).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.bench)))
    recs: dict[str, str] = {}
    specs = []
    for r in rows:
        rseq = ca_seq(Path(r["receptor"]))
        pseq = r["seq"].strip().upper()
        # X residues are unparseable to boltz's token map and a peptide read off the CSV can
        # disagree with the crystal file; skip loudly rather than emit a spec that dies later.
        if not rseq or not pseq or 'X' in rseq or 'X' in pseq:
            print(f"  skip {r['name']}: rec={len(rseq)} pep={len(pseq)} (X or empty)")
            continue
        h = hashlib.md5(rseq.encode()).hexdigest()[:12]
        recs[h] = rseq
        specs.append((r["name"], h, rseq, pseq, r["length_bucket"]))

    print(f"{len(specs)} complexes, {len(recs)} unique receptors")

    from boltz.data.msa.mmseqs2 import run_mmseqs2
    todo = [(h, s) for h, s in recs.items() if not (OUT / f"msa/{h}.a3m").exists()]
    print(f"MSAs already on disk: {len(recs) - len(todo)}   to fetch: {len(todo)}")

    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        work = Path(tempfile.mkdtemp(prefix="b2msa_", dir="/tmp/claude-1000"))
        try:
            t0 = time.time()
            a3ms = run_mmseqs2([s for _, s in chunk], str(work), use_env=True,
                               use_pairing=False, host_url="https://api.colabfold.com")
            for (h, _), a3m in zip(chunk, a3ms):
                (OUT / f"msa/{h}.a3m").write_text(a3m)
            n = sum(1 for x in a3ms for l in x.splitlines() if l.startswith(">"))
            print(f"  batch {i // a.batch + 1}: {len(chunk)} receptors, {n} total hits, "
                  f"{time.time() - t0:.0f}s", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        time.sleep(3)          # be polite to a free shared service

    nw = 0
    for name, h, rseq, pseq, bucket in specs:
        msa = OUT / f"msa/{h}.a3m"
        if not msa.exists():
            print(f"  skip {name}: no MSA for receptor {h}")
            continue
        (OUT / a.yaml_sub / f"{name}.yaml").write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {rseq}\n      msa: {msa}\n"
            f"  - protein:\n      id: B\n      sequence: {pseq}\n      msa: empty\n")
        nw += 1
    print(f"\nwrote {nw} YAML specs to {OUT / 'yaml'}")


if __name__ == "__main__":
    main()
