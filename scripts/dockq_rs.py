#!/usr/bin/env python
"""DockQ (CAPRI-peptide) scoring of RecentSet poses. capri_peptide=True is the
standard the RAPiDock paper reports: Fnat cutoff 4 A, interface cutoff 8 A."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
# Defaults are the RecentSet benchmark; DOCKQ_SET=balanced scores the length-balanced
# post-2020 set instead (same metric, different poses/output file).
_SETS = {
    "recentset": ("data/bench_recentset_heldout.csv", "runs/recentset", "logs/dockq_recentset.jsonl"),
    "balanced": ("data/bench_balanced_post2020.csv", "runs/balanced", "logs/dockq_balanced.jsonl"),
}
_SET = os.environ.get("DOCKQ_SET", "recentset")
if _SET not in _SETS:
    raise SystemExit(f"DOCKQ_SET must be one of {sorted(_SETS)}, got {_SET!r}")
_BENCH_REL, _RUNS_REL, _OUT_REL = _SETS[_SET]
BENCH = ROOT / _BENCH_REL
RUNS = ROOT / _RUNS_REL
OUT = ROOT / _OUT_REL
CAPRI = [("high", 0.80), ("medium", 0.49), ("acceptable", 0.23)]


def capri(dq: float) -> str:
    for label, lo in CAPRI:
        if dq >= lo:
            return label
    return "incorrect"


def _combine(rec: str, pep: str) -> str:
    """receptor atoms + peptide atoms relabelled to chain Z -> temp pdb path."""
    fd, out = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as fh:
        for line in open(rec):
            if line.startswith("ATOM") and line[21] != "Z":
                fh.write(line)
        for line in open(pep):
            if line.startswith("ATOM"):
                fh.write(line[:21] + "Z" + line[22:])
        fh.write("END\n")
    return out


def score_pose(rec: str, pep_native: str, pose: str) -> float:
    """Best receptor-peptide DockQ for one pose (nan on failure)."""
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    m = n = None
    try:
        m, n = _combine(rec, pose), _combine(rec, pep_native)
        ms, ns = load_PDB(m), load_PDB(n)
        chains = {c.id for c in ns} & {c.id for c in ms}
        cmap = {c: c for c in chains}
        res, _ = run_on_all_native_interfaces(ms, ns, chain_map=cmap, capri_peptide=True)
        vals = [v["DockQ"] for k, v in res.items() if "Z" in k]
        return max(vals) if vals else float("nan")
    except Exception:
        return float("nan")
    finally:
        for p in (m, n):
            if p:
                os.unlink(p)


def score_complex(args: tuple[str, str, str, str]) -> dict:
    name, rec, pep, arm = args
    d = RUNS / arm / name
    poses = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    return {"name": name, "arm": arm,
            "dockq": [score_pose(rec, pep, str(p)) for p in poses]}


def main() -> None:
    arms = sys.argv[1:] or ["rapidock_og", "hybridock_ft"]
    meta = {r["name"]: r for r in csv.DictReader(open(BENCH))}
    out = OUT
    # Resumable: skip complex-arms already in the output (a memory-pressure kill mid-run
    # must not force re-scoring thousands of poses).
    have = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                have.add((r["name"], r["arm"]))
    jobs = [(n, m["receptor"], m["peptide_pdb"], a)
            for a in arms for n, m in meta.items()
            if (RUNS / a / n).is_dir() and (n, a) not in have]
    workers = int(os.environ.get("DOCKQ_WORKERS", "3"))
    print(f"scoring {len(jobs)} complex-arms ({len(have)} already done) with "
          f"DockQ(capri_peptide=True), {workers} workers", flush=True)
    done = 0
    with out.open("a") as fh, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_complex, j): j for j in jobs}
        for f in as_completed(futs):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
