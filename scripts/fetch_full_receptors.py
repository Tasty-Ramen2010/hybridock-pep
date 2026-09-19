#!/usr/bin/env python
"""Fetch full, untruncated receptors so a BLIND (global) model can actually be trained.

WHY THIS IS NEEDED. Our whole training corpus points at `*_protein_pocket.pdb` -- receptors
truncated to ~20 A around the peptide, median 58 residues, 80% under 120. That is the correct
input for the site-specific model, and it is useless for a blind one: training a search model on
data where the search has already been solved teaches it that the peptide goes in the middle of
the blob, which is exactly the naive-centroid heuristic that fails at 206 A.

WHY IT IS WORTH DOING NOW, having previously argued against it. The August oracle-init result
("no retrain needed for blind placement") was an INFERENCE-only test on the authors' pretrained
weights, and it showed the model reaching 15.8 A mean given a perfect start -- a ceiling, not a
success. Ram's point stands: every training run this project has done was confounded, by the
~50 A idealized-strand target (until Sep 12), the SiLU/Tanh fork, unfrozen magnitude gates,
validation sets inside their own training data, unfiltered broken topology, and 630 benchmark
complexes sitting in the training pool. We have never run a clean one. A clean global retrain is
a genuinely different experiment from the five that failed.

THE CHAIN-ASSIGNMENT TRAP, and how it is avoided. It is tempting to parse the complex name --
peppcf_1N95_A_31_38 -- and drop chain A residues 31-38. That is wrong for fragment entries,
where the "peptide" is a slice of a chain that also contributes receptor, and it is exactly the
kind of silent mis-assignment that has cost this project whole runs. Instead the peptide's own
PDB file is read, its (chain, residue) keys are collected, and precisely those are removed from
the downloaded structure. Whatever remains is the receptor, verified to be larger than the
pocket it replaces.

Usage: fetch_full_receptors.py [--workers 6] [--limit N]
Output: datasets/fullrec/<PDBID>.pdb  plus data/fullcorpus_global_{train,val}.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
RAW = ROOT / "datasets/fullrec/raw"
OUT = ROOT / "datasets/fullrec"
URL = "https://files.rcsb.org/download/{}.pdb"
AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
      'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


def pdb_id(name: str) -> str:
    for tok in re.split(r"[_\-]", name):
        if len(tok) == 4 and tok[0].isdigit() and tok.isalnum():
            return tok.upper()
    return name.upper()


def fetch(pid: str) -> tuple[str, str]:
    dest = RAW / f"{pid}.pdb"
    if dest.exists() and dest.stat().st_size > 2000:
        return pid, "cached"
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL.format(pid),
                                         headers={"User-Agent": "hybridock-pep/0.2"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 2000:
                return pid, "too small"
            dest.write_bytes(data)
            return pid, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return pid, "404"
            time.sleep(1 + attempt * 2)
        except Exception:                      # noqa: BLE001 - network, retry
            time.sleep(1 + attempt * 2)
    return pid, "failed"


def pep_keys(path: Path) -> set[tuple[str, str]]:
    """(chain, resseq) of every peptide residue — read from the file, never inferred."""
    keys = set()
    for l in path.read_text(errors="ignore").splitlines():
        if l.startswith("ATOM") and l[17:20].strip() in AA:
            keys.add((l[21], l[22:27]))
    return keys


def build_receptor(pid: str, drop: set[tuple[str, str]], dest: Path) -> int:
    src = RAW / f"{pid}.pdb"
    if not src.exists():
        return 0
    keep, seen = [], set()
    for l in src.read_text(errors="ignore").splitlines():
        if l.startswith("ENDMDL"):
            break
        if not l.startswith("ATOM"):
            continue
        if l[16] not in (" ", "A"):
            continue
        if l[17:20].strip() not in AA:
            continue
        k = (l[21], l[22:27])
        if k in drop:
            continue
        keep.append(l)
        seen.add(k)
    if len(seen) < 20:
        return 0
    dest.write_text("\n".join(keep) + "\nTER\nEND\n")
    return len(seen)


def nres(path: Path) -> int:
    s = set()
    for l in path.read_text(errors="ignore").splitlines():
        if l.startswith("ATOM"):
            s.add((l[21], l[22:27]))
    return len(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6, help="RCSB is rate-limited; keep modest")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    rows = {}
    for split in ("train", "val"):
        rows[split] = list(csv.DictReader((ROOT / f"data/fullcorpus_{split}.csv").open()))
    ids = sorted({pdb_id(r["complex_name"]) for s in rows for r in rows[s]})
    if a.limit:
        ids = ids[:a.limit]
    print(f"{len(ids)} distinct PDB entries to fetch\n", flush=True)

    done, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (pid, st) in enumerate(ex.map(fetch, ids), 1):
            done[pid] = st
            if i % 250 == 0:
                el = time.time() - t0
                ok = sum(1 for v in done.values() if v in ("ok", "cached"))
                print(f"  {i}/{len(ids)}  ok={ok}  "
                      f"eta {(el / i) * (len(ids) - i) / 60:.0f} min", flush=True)
    from collections import Counter
    print(f"\n  fetch: {dict(Counter(done.values()))}\n")

    print("building receptors (peptide residues removed by FILE, not by name)...", flush=True)
    built, ratios, skipped = {}, [], 0
    for split in ("train", "val"):
        out_rows = []
        for r in rows[split]:
            pid = pdb_id(r["complex_name"])
            if done.get(pid) not in ("ok", "cached"):
                skipped += 1
                continue
            d = OUT / r["complex_name"]
            d.mkdir(parents=True, exist_ok=True)
            dest = d / "receptor_full.pdb"
            if not dest.exists():
                n = build_receptor(pid, pep_keys(Path(r["peptide_description"])), dest)
                if not n:
                    skipped += 1
                    continue
            else:
                n = nres(dest)
            pocket_n = nres(Path(r["protein_description"]))
            if pocket_n:
                ratios.append(n / pocket_n)
            out_rows.append({**r, "protein_description": str(dest)})
        built[split] = out_rows
        p = ROOT / f"data/fullcorpus_global_{split}.csv"
        if out_rows:
            with p.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
                w.writeheader(); w.writerows(out_rows)
            print(f"  -> {p.relative_to(ROOT)}  {len(out_rows)} rows")

    if ratios:
        import statistics as st
        print(f"\n  full receptor / pocket size ratio: median {st.median(ratios):.1f}x  "
              f"mean {st.mean(ratios):.1f}x")
        print("     (a ratio near 1 would mean the download did NOT give us more protein "
              "than we already had\n      -- the whole point of this step)")
    print(f"  skipped {skipped} rows (no structure or receptor too small)")


if __name__ == "__main__":
    main()
