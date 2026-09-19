#!/usr/bin/env python
"""Extract pair features from the DOCKED all-by-all blocks — matched to the Coventry protocol.

This is the training set the selectivity model should have had from the start. Every earlier
attempt trained on THREADED poses, where the peptide sits on a crystal backbone in the right
place, and then was asked to score DOCKED poses, where 79-81% of the pool threads the groove
backwards. That mismatch is why a model reaching AUC 0.979 on its own benchmark collapsed to
0.529 on the real grid.

Three things are held identical to the Coventry extraction so nothing else can explain a
difference: the same generator and checkpoint (longft_tanh epoch010), the HARD repack with no
soft-repulsive pre-pass (the soft pass relieves bad poses along with good ones and costs
AUC 0.646 -> 0.579 on docked data), and the same best-of-N and top-5 aggregation.

Usage: sel_dock_features.py [workers]
Output: logs/sel_dock_features.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
POSES = ROOT / "runs/sel_dock"
OUT = ROOT / "logs/sel_dock_features.jsonl"
TOPK = 5

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["SEL_SOFT_PASS"] = "0"          # docked poses: their clashes are real signal
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF
    import sel_coventry_features as SCF
    SCF._PR = _PR
    SCF._SF = _SF
    SCF.SOFT_PASS = False


def do_cell(args: tuple) -> dict:
    import sel_coventry_features as SCF
    name, receptor, block, i, j, pep_pdb, rec_pdb = args
    row = {"name": name, "block": block, "pep_idx": i, "rec_idx": j,
           "cognate": int(i == j), "pep_pdb": pep_pdb, "rec_pdb": rec_pdb}
    t0 = time.time()
    poses = sorted((POSES / name).glob("rank*.pdb"),
                   key=lambda p: int(re.search(r"\d+", p.name).group()))
    feats = []
    for p in poses:
        try:
            feats.append(SCF.refine_and_feature(receptor, str(p)))
        except Exception:  # noqa: BLE001 -- a bad pose must not kill the cell
            continue
    if not feats:
        row["error"] = "no pose scored"
        return row
    feats.sort(key=lambda f: f["i_total"])
    keys = sorted({k for f in feats for k, v in f.items() if isinstance(v, (int, float))})
    row["best"] = {k: float(feats[0].get(k, 0.0)) for k in keys}
    top = feats[:TOPK]
    row["top5"] = {k: sum(float(f.get(k, 0.0)) for f in top) / len(top) for k in keys}
    row["n_poses"] = len(feats)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    jobs = []
    for f in sorted(ROOT.glob("logs/sel_dock_s*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["name"] in done or r.get("n_poses", 0) < 2:
                continue
            if not list((POSES / r["name"]).glob("rank*.pdb")):
                continue
            done.add(r["name"])
            jobs.append((r["name"], r["receptor"], r["block"], r["pep_idx"], r["rec_idx"],
                         r["pep_pdb"], r["rec_pdb"]))
    print(f"{len(jobs)} docked cells to featurise, {workers} workers", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, row in enumerate(ex.map(do_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 25 == 0 or k == 1:
                rate = k / max(time.time() - t0, 1)
                v = row.get("best", {}).get("i_total", float("nan"))
                print(f"  [{k}/{len(jobs)}] {row['name']:12s} i_total {v:8.1f}  "
                      f"eta {(len(jobs) - k) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    print("SEL_DOCK_FEATURES_DONE")


if __name__ == "__main__":
    main()
