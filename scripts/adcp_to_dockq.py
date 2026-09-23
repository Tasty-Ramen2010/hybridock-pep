#!/usr/bin/env python
"""Score ADCP's ranked modes with the SAME DockQ (capri_peptide) pass as our poses.

ADCP writes <job>_ranked_N.pdb: one peptide, chain A, no MODEL wrapper, rank order
(mode 1 = best affinity). We relabel to chain Z and reuse scripts/dockq_rs.score_pose,
so RAPiDock-original, our finetune, and ADCP all share one metric and one crystal.

ADCP's own rmsd columns are 999.0 (no -ref given, deliberately) -- never read accuracy
from its logs, only from this pass.

Usage: adcp_to_dockq.py [runs/adcp]   ->  logs/dockq_adcp.jsonl
"""
from __future__ import annotations

import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dockq_rs import score_pose  # noqa: E402

ROOT = Path("/home/igem/unknown_software")
# ADCP_BENCH selects the bench; the length-balanced 387 is what the paper-style figures use.
import os as _os
BENCH = Path(_os.environ.get("ADCP_BENCH", str(ROOT / "data/bench_recentset_heldout.csv")))
OUT = Path(_os.environ.get("ADCP_OUT", str(ROOT / "logs/dockq_adcp.jsonl")))


def score_one(args: tuple[str, str, str, str]) -> dict:
    name, rec, pep, adcp_dir = args
    d = Path(adcp_dir) / name
    # ADCP names output <jobName>_ranked_N.pdb; the docking runner uses jobName="dock",
    # but accept any *_ranked_N.pdb so a renamed job still scores.
    modes = sorted(d.glob("*_ranked_*.pdb"),
                   key=lambda p: int(re.search(r"_ranked_(\d+)", p.name).group(1)))
    return {"name": name,
            "dockq": [score_pose(rec, pep, str(m)) for m in modes],
            "n_modes": len(modes)}


def main() -> None:
    adcp_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs/adcp")
    meta = {r["name"]: r for r in csv.DictReader(open(BENCH))}
    have = {n: bool(list((Path(adcp_dir) / n).glob("dock_ranked_*.pdb"))) for n in meta}
    jobs = [(n, m["receptor"], m["peptide_pdb"], adcp_dir)
            for n, m in meta.items() if have[n]]
    # Complexes where ADCP produced NO ranked output (its clusterADCP step dies with
    # "min() arg is an empty sequence" on a few targets) are written with an empty pose
    # list, NOT dropped. Our arms produced poses for all 345, so silently shrinking ADCP's
    # denominator to the targets it survived would inflate its success rate.
    crashed = sorted(n for n in meta if not have[n])
    print(f"scoring {len(jobs)} ADCP results; {len(crashed)} produced no ranked output "
          f"and count as failures: {', '.join(crashed) if crashed else '-'}", flush=True)
    out = OUT
    with out.open("a") as fh:
        for n in crashed:
            fh.write(json.dumps({"name": n, "dockq": [], "n_modes": 0,
                                 "no_output": True}) + "\n")
    done = 0
    with out.open("a") as fh, ProcessPoolExecutor(max_workers=int(_os.environ.get("DOCKQ_WORKERS", "6"))) as ex:
        futs = [ex.submit(score_one, j) for j in jobs]
        for f in as_completed(futs):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
