#!/usr/bin/env python
"""Rank our benchmark poses with RAPiDock's OWN shipped ranker (ref2015), for comparison.

RAPiDock ships two ranking options. `--scoring_function confidence` needs a trained
ConfidenceModel and the authors ship NO weights for it, so it cannot be run out of the box.
Their default (default_inference_args.yaml) is `ref2015`: PyRosetta FastRelax on the peptide
in the receptor's field, then the Rosetta ref2015 energy of the relaxed complex, ascending.
This reproduces that exactly, using their own utils/pyrosetta_utils.relax_score, so the
comparison against our consensus ranker is like-for-like.

SAFETY: their relax_score() calls os.remove() on the peptide file it is given. We therefore
copy every pose into a scratch directory and hand it the COPY. Never point it at runs/.

Because FastRelax rewrites the pose, ref2015's top-1 is scored on the RELAXED pose -- that is
what a RAPiDock user would actually receive, so it is the fair thing to score.

Resumable: complexes already in the output file are skipped.

Usage: ref2015_rank.py <arm> [max_iter] [workers] [limit]
Output: logs/ref2015_rank_<arm>.jsonl   {name, arm, order, scores, dockq_relaxed}
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
RD = ROOT / "third_party/RAPiDock"
sys.path.insert(0, str(RD))
sys.path.insert(0, str(ROOT / "scripts"))

ARM = sys.argv[1] if len(sys.argv) > 1 else "hybridock_ft_shipped"
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 20      # RAPiDock's own code default
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
OUT = ROOT / f"logs/ref2015_rank_{ARM}.jsonl"


def main() -> None:
    import multiprocessing as mp

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import utils.pyrosetta_utils as pru  # noqa: PLC0415 -- initialises PyRosetta on import

    if not pru.PYROSETTA_AVAILABLE:
        raise SystemExit("PyRosetta unavailable; cannot run RAPiDock's ref2015 ranker")
    if MAX_ITER != 20:
        pru.RR.fast_relax.max_iter(MAX_ITER)

    from dockq_rs import score_pose  # noqa: PLC0415

    meta = {r["name"]: r for r in csv.DictReader(open(ROOT / "data/bench_recentset_heldout.csv"))}
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    names = sorted(n for n in meta if (ROOT / "runs/recentset" / ARM / n).is_dir() and n not in done)
    if LIMIT:
        names = names[:LIMIT]
    print(f"arm={ARM} max_iter={MAX_ITER} workers={WORKERS}: {len(names)} complexes to rank "
          f"({len(done)} already done)", flush=True)

    scratch = Path(tempfile.mkdtemp(prefix="ref2015_", dir="/tmp/claude-1000"))
    t0 = time.time()
    try:
        with OUT.open("a") as fh, mp.Pool(WORKERS) as pool:
            for k, name in enumerate(names, 1):
                m = meta[name]
                d = ROOT / "runs/recentset" / ARM / name
                files = sorted(d.glob("rank*.pdb"),
                               key=lambda p: int(re.search(r"\d+", p.name).group()))
                if not files:
                    continue
                work = scratch / name
                work.mkdir(parents=True, exist_ok=True)
                # COPIES: relax_score deletes its input peptide file.
                pep_in = [str(work / f"in{i}.pdb") for i in range(len(files))]
                relaxed = [str(work / f"out{i}.pdb") for i in range(len(files))]
                for src, dst in zip(files, pep_in):
                    shutil.copy(src, dst)
                try:
                    scores = pool.map(pru.relax_score,
                                      list(zip([m["receptor"]] * len(files), pep_in,
                                               relaxed, [True] * len(files))))
                except Exception as exc:  # noqa: BLE001 -- one bad complex must not stop the sweep
                    print(f"  {name}: relax failed ({type(exc).__name__}: {exc})", flush=True)
                    shutil.rmtree(work, ignore_errors=True)
                    continue
                pairs = [(s, i) for i, s in enumerate(scores) if s is not None]
                if not pairs:
                    shutil.rmtree(work, ignore_errors=True)
                    continue
                order = [i for _, i in sorted(pairs)]          # ascending energy = their ranking
                dq = [score_pose(m["receptor"], m["peptide_pdb"], relaxed[i])
                      if Path(relaxed[i]).exists() else float("nan") for i in order[:5]]
                fh.write(json.dumps({"name": name, "arm": ARM, "max_iter": MAX_ITER,
                                     "order": order, "scores": [float(s) for s, _ in sorted(pairs)],
                                     "dockq_relaxed": dq}) + "\n")
                fh.flush()
                shutil.rmtree(work, ignore_errors=True)
                if k % 5 == 0 or k == 1:
                    el = time.time() - t0
                    print(f"  {k}/{len(names)}  {el / k:.0f}s per complex  "
                          f"ETA {(len(names) - k) * el / k / 3600:.1f}h", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
