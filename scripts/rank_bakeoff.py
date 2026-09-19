#!/usr/bin/env python
"""Head-to-head of the pose rankers on RAPiDock's leak-free RecentSet, scored by DockQ.

Decides which ranker the tool should use as PRIMARY. All candidates are blind (they never see
the crystal); DockQ labels are only used to score the resulting top-1 pick.

  consensus   mean heavy-atom RMSD to the other poses of the run (scoring/consensus.py)
  bsa_fit     -z(buried surface area) + z(clashes)          (scoring/bsa_fit.py)
  combined    z(consensus) + z(bsa_fit), equal weight, per complex
  oracle      best-of-N ceiling            random  mean over poses (no ranker)

Usage: python scripts/rank_bakeoff.py [arm] [n_complexes]
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.bsa_fit import _read_heavy  # noqa: E402
from hybridock_pep.scoring.consensus import _heavy_coords  # noqa: E402

RUNS = ROOT / "runs/recentset"
LEVELS = [("acceptable", 0.23), ("medium", 0.49), ("high", 0.80)]
CLASH_DIST = 3.0
CROP = 10.0


def _z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def score_complex(job: tuple[str, str, str]) -> dict | None:
    """Return per-pose ranker scores for one complex, in rank-file order."""
    name, arm, receptor = job
    d = RUNS / arm / name
    files = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    if len(files) < 3:
        return None
    coords = [_heavy_coords(p) for p in files]
    shapes = [c.shape for c in coords if c.size]
    if not shapes:
        return None
    ref = max(set(shapes), key=shapes.count)
    keep = [i for i, c in enumerate(coords) if c.shape == ref]
    if len(keep) < 3:
        return None

    x = np.stack([coords[i] for i in keep])
    diff = x[:, None, :, :] - x[None, :, :, :]
    consensus = (np.sqrt((diff ** 2).sum(-1).mean(-1)).sum(1) / (len(keep) - 1))

    # BSA proxy: contacts within 4.5 A (monotone in buried area, far cheaper than SASA)
    # plus the same clash count bsa_fit uses. Kept identical across poses of one complex,
    # so it is a valid within-complex ranking signal.
    try:
        _, rec = _read_heavy(Path(receptor))
    except Exception:  # noqa: BLE001
        return None
    if rec.size == 0:
        return None
    contacts, clashes = [], []
    for i in keep:
        p = coords[i]
        sub = rec[np.linalg.norm(rec - p.mean(0), axis=1) < (np.linalg.norm(p - p.mean(0), axis=1).max() + CROP)]
        if sub.size == 0:
            contacts.append(0.0); clashes.append(0.0); continue
        dist = np.linalg.norm(p[:, None, :] - sub[None, :, :], axis=-1)
        contacts.append(float((dist < 4.5).sum()))
        clashes.append(float((dist < CLASH_DIST).sum()))
    bsa_fit = -_z(np.asarray(contacts)) + _z(np.asarray(clashes))

    return {"name": name, "arm": arm,
            "pose_idx": [int(re.search(r"\d+", files[i].name).group()) for i in keep],
            "consensus": consensus.tolist(), "bsa_fit": bsa_fit.tolist()}


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "hybridock_ft_shipped"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import csv
    meta = {r["name"]: r for r in csv.DictReader(open(ROOT / "data/bench_recentset_heldout.csv"))}
    dq: dict[str, list[float]] = {}
    for line in (ROOT / "logs/dockq_recentset.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["arm"] == arm:
                dq[r["name"]] = r["dockq"]
    names = sorted(n for n in dq if (RUNS / arm / n).is_dir())
    if limit:
        names = names[:limit]
    jobs = [(n, arm, meta[n]["receptor"]) for n in names]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("RANK_WORKERS", "4"))) as ex:
        res = [r for r in ex.map(score_complex, jobs, chunksize=4) if r]

    picks: dict[str, list[float]] = {k: [] for k in
                                     ("oracle", "consensus", "bsa_fit", "combined", "random")}
    for r in res:
        scores = dq[r["name"]]
        vals = [scores[i - 1] if i - 1 < len(scores) else float("nan") for i in r["pose_idx"]]
        ok = [v for v in vals if v == v]
        if not ok:
            continue
        c = np.asarray(r["consensus"]); b = np.asarray(r["bsa_fit"])
        comb = _z(c) + _z(b)
        picks["oracle"].append(max(ok))
        picks["random"].append(sum(ok) / len(ok))
        for key, arr in (("consensus", c), ("bsa_fit", b), ("combined", comb)):
            v = vals[int(np.argmin(arr))]
            picks[key].append(v if v == v else min(ok))

    n = len(picks["oracle"])
    print(f"\nRANKER BAKE-OFF  arm={arm}  n={n} complexes  (top-1 pick, DockQ capri_peptide)")
    print(f"  {'ranker':11s} {'medianDockQ':>11s} " + " ".join(f"{l[:4]:>7s}" for l, _ in LEVELS))
    for key in ("oracle", "combined", "consensus", "bsa_fit", "random"):
        v = picks[key]
        cells = " ".join(f"{100 * sum(1 for x in v if x >= t) / len(v):6.1f}%" for _, t in LEVELS)
        print(f"  {key:11s} {st.median(v):11.3f} {cells}")


if __name__ == "__main__":
    main()
