#!/usr/bin/env python
"""Extract the selectivity feature set on the real Coventry grid, from DOCKED poses.

This is the transfer test.  The selectivity model is trained on threaded poses -- every
sequence placed on a crystal backbone, so pose quality is held constant and only chemistry
varies.  Here nothing is held constant: each cell gets 24 RAPiDock poses, of which 79-81% thread
the groove backwards.  If the model survives that, it has learned chemistry; if it collapses,
it learned something about ideal backbones and the honest report is that it does not transfer.

Two aggregations are written per cell so the evaluation can choose between them:
  best   the single pose with the lowest interface energy -- the convention the grid has used
         throughout, and the one our dG model and ref2015 were compared under.
  top5   the mean feature vector over the five best poses.  There is a specific reason to
         expect this is better: on docked poses an interaction fingerprint scored from the
         rank-1 pose reaches r=0.116, and the same fingerprint averaged over 5 poses reaches
         r=0.430.  A single docked pose is a noisy draw; the ensemble is the estimate.

Usage: sel_coventry_features.py [workers] [--arm hybridock_ft]
Output: logs/sel_coventry_features.jsonl
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
# Which docked pose set to featurise. Selectable so a re-docked arm (a different RAPiDock
# checkpoint) can be scored through the identical extractor -- same hard repack, same
# aggregation -- which is the only way the checkpoint stays the single variable.
ARM = os.environ.get("SEL_ARM", "hybridock_ft")
POSES = ROOT / "runs/coventry" / ARM
ESM = ROOT / "datasets/coventry/binders"
AF3 = ROOT / "datasets/coventry/binders_af3"
OUT = ROOT / os.environ.get("SEL_COV_OUT", "logs/sel_coventry_features.jsonl")
#: Threading NEEDS a soft-repulsive pre-pass -- forcing a bulky sequence onto a backbone shaped
#: for a small one makes artificial clashes that a hard pass cannot escape, and every sequence
#: then scores 10^4 REU. A DOCKED pose is the opposite case: its clashes are real, and they are
#: the single most informative thing about it, because 79-81% of our poses thread the groove
#: backwards and that is what makes them expensive. Measured: with the soft pass, centred ref2015
#: on this grid ranks the cognate at 8.22 of 18; without it, 6.67. The soft pass relieves the bad
#: poses along with the good and throws the discrimination away.
SOFT_PASS = os.environ.get("SEL_SOFT_PASS", "1") != "0"
#: binders where our ESMFold model is the proven outlier (Boltz and AF3 agree against it)
BROKEN = {"n3", "n7", "pc21", "pc26"}
TOPK = 5

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF


def refine_and_feature(receptor: str, pose_pdb: str) -> dict:
    """Repack + minimise one docked pose, then extract the same pair features as training."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
    import sel_features as SF

    rec = _PR.pose_from_pdb(receptor)
    pep = _PR.pose_from_pdb(pose_pdb)
    info = pep.pdb_info()
    for k in range(1, pep.total_residue() + 1):
        info.chain(k, "p")
    n_rec = rec.total_residue()
    pose = rec.clone()
    pose.append_pose_by_jump(pep, n_rec)

    pep_sel = rs.ChainSelector("p")
    near = rs.NeighborhoodResidueSelector(pep_sel, SF.INTERFACE_CUT, False)
    movable = rs.OrResidueSelector(pep_sel, near)
    frozen = rs.NotResidueSelector(movable)
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(), frozen))

    mm = _PR.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False); mm.set_chi(False); mm.set_jump(True)
    sub = movable.apply(pose)
    for k in range(1, pose.total_residue() + 1):
        if sub[k]:
            mm.set_chi(k, True)
            if k > n_rec:
                mm.set_bb(k, True)

    if SOFT_PASS:
        soft = _PR.create_score_function("ref2015_soft")
        PackRotamersMover(soft, tf.create_task_and_apply_taskoperations(pose)).apply(pose)
        m0 = MinMover(mm, soft, "lbfgs_armijo_nonmonotone", 0.01, True)
        m0.max_iter(100); m0.apply(pose)
    PackRotamersMover(_SF, tf.create_task_and_apply_taskoperations(pose)).apply(pose)
    mn = MinMover(mm, _SF, "lbfgs_armijo_nonmonotone", 0.01, True)
    mn.max_iter(300); mn.apply(pose)

    split = list(pose.split_by_chain())
    pep_out = split[-1]
    rec_only = split[0].clone()
    for c in split[1:-1]:
        rec_only.append_pose_by_jump(c, rec_only.total_residue())
    tc, tr, tp = SF._terms_of(pose), SF._terms_of(rec_only), SF._terms_of(pep_out)
    out = {f"i_{t}": tc[t] - tr[t] - tp[t] for t in SF.TERMS}
    out["i_total"] = float(_SF(pose) - _SF(rec_only) - _SF(pep_out))
    out.update(SF._geometry(pose, n_rec))
    return out


def do_cell(args: tuple[str, str]) -> dict:
    name, binder = args
    row = {"name": name, "peptide": name.split("__")[0], "binder": binder}
    t0 = time.time()
    rec = (AF3 if binder in BROKEN else ESM) / f"{binder}_1b1.pdb"
    row["receptor_model"] = "af3" if binder in BROKEN else "esmfold"
    poses = sorted((POSES / name).glob("rank*.pdb"),
                   key=lambda p: int(re.search(r"\d+", p.name).group()))
    feats = []
    for p in poses:
        try:
            feats.append(refine_and_feature(str(rec), str(p)))
        except Exception:  # noqa: BLE001 -- a bad pose must not kill the cell
            continue
    if not feats:
        row["error"] = "no pose scored"
        return row
    feats.sort(key=lambda f: f["i_total"])
    # A pose with zero receptor contacts yields no geometry features at all, so the key set is
    # not the same for every pose. Take the union and treat a missing feature as zero -- which
    # is what "no contacts of this type" means for every one of them.
    keys = sorted({k for f in feats for k, v in f.items() if isinstance(v, (int, float))})
    row["best"] = {k: float(feats[0].get(k, 0.0)) for k in keys}
    top = feats[:TOPK]
    row["top5"] = {k: sum(float(f.get(k, 0.0)) for f in top) / len(top) for k in keys}
    row["n_poses"] = len(feats)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    jobs = []
    for d in sorted(POSES.glob("*__*")):
        if d.name in done or not list(d.glob("rank*.pdb")):
            continue
        jobs.append((d.name, d.name.split("__")[1][:-1]))
    print(f"{len(jobs)} Coventry cells ({len(done)} done), {workers} workers", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, row in enumerate(ex.map(do_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 10 == 0 or k == 1:
                rate = k / max(time.time() - t0, 1)
                v = row.get("best", {}).get("i_total", float("nan"))
                print(f"  [{k}/{len(jobs)}] {row['name']:20s} i_total {v:8.1f}  "
                      f"eta {(len(jobs) - k) / max(rate, 1e-6) / 60:.0f} min", flush=True)
    print("SEL_COVENTRY_DONE")


if __name__ == "__main__":
    main()
