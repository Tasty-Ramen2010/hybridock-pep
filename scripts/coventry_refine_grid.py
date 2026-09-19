#!/usr/bin/env python
"""Refine and score every docked cell of the Coventry grid, in parallel.

For each (peptide, binder) cell: repack the interface and minimise each of the 24 poses
against a fixed receptor backbone (see coventry_refine.py for why this step is needed at
all), then record per-pose Rosetta energies and the post-refinement closest contact.

Runs CPU-only, so it can share the machine with the GPU docking grid. Resumable: cells
already present in the output JSONL are skipped, so it can be re-run as docking fills in.

Aggregates written per cell, all of which are computed identically for every cell -- no
knowledge of which pairs are cognate enters here:
  best_iface     most negative ref2015 interface energy over the 24 poses
  med_iface      median interface energy (less sensitive to one lucky pose)
  top3_iface     mean of the three most negative
  n_clean        poses whose closest heavy-atom contact is >= 2.5 A after refinement

Usage: coventry_refine_grid.py [workers] [--binders DIR]
Output: logs/coventry_refine.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
ARM = os.environ.get("COVENTRY_ARM", "hybridock_ft")
POSES = ROOT / "runs/coventry" / ARM
OUT = ROOT / f"logs/coventry_refine{'' if ARM == 'hybridock_ft' else '_' + ARM}.jsonl"
BINDERS = ROOT / ("datasets/coventry/binders_af3" if "af3" in ARM else "datasets/coventry/binders")
CLEAN_CUT = 2.5

_PR = None


def _init():
    global _PR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()


def _min_contact(pdb: str) -> float:
    import numpy as np
    from scipy.spatial import cKDTree
    rec, pep = [], []
    for l in open(pdb):
        if l.startswith("ATOM") and l[76:78].strip() != "H":
            xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            (pep if l[21] == "B" else rec).append(xyz)
    if not rec or not pep:
        return float("nan")
    return float(cKDTree(np.asarray(rec)).query(np.asarray(pep))[0].min())


def do_cell(args: tuple[str, str]) -> dict:
    import statistics as st
    import tempfile
    from coventry_refine import refine_one
    name, receptor = args
    d = POSES / name
    poses = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    rows = []
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000") as td:
        for p in poses:
            out = str(Path(td) / p.name)
            try:
                r = refine_one(_PR, receptor, str(p), out)
                r["min_contact"] = _min_contact(out)
                r["pose"] = p.name
                rows.append(r)
            except Exception as exc:  # noqa: BLE001 -- a bad pose must not kill the cell
                rows.append({"pose": p.name, "error": f"{type(exc).__name__}: {exc}"})
    ok = [r for r in rows if "ref2015_interface" in r]
    ifaces = sorted(r["ref2015_interface"] for r in ok)
    agg = {
        "name": name, "n_poses": len(poses), "n_scored": len(ok),
        "best_iface": ifaces[0] if ifaces else None,
        "med_iface": st.median(ifaces) if ifaces else None,
        "top3_iface": (sum(ifaces[:3]) / min(3, len(ifaces))) if ifaces else None,
        "n_clean": sum(1 for r in ok if r.get("min_contact", 0) >= CLEAN_CUT),
        "poses": rows,
    }
    return agg


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    bdir = Path(sys.argv[sys.argv.index("--binders") + 1]) if "--binders" in sys.argv else BINDERS

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    jobs = []
    for d in sorted(POSES.glob("*/")):
        name = d.name
        if name in done or not list(d.glob("rank*.pdb")):
            continue
        binder = name.split("__")[1][:-1] + "_1b1"
        rec = bdir / f"{binder}.pdb"
        if rec.exists():
            jobs.append((name, str(rec)))
    print(f"{len(jobs)} cells to refine ({len(done)} already done), {workers} workers",
          flush=True)
    if not jobs:
        return

    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, agg in enumerate(ex.map(do_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(agg) + "\n")
            fh.flush()
            if k % 10 == 0 or k == 1:
                print(f"  [{k}/{len(jobs)}] {agg['name']:22s} "
                      f"best_iface {agg['best_iface']:8.1f}  clean {agg['n_clean']:2d}/24",
                      flush=True)
    print("COVENTRY_REFINE_DONE")


if __name__ == "__main__":
    main()
