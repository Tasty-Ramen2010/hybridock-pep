#!/usr/bin/env python
"""Why do non-binders score as well as binders? Decompose the interface energy.

Two hypotheses for why our grid says everything binds:

  H1  INDUCED FIT IS TOO PERMISSIVE.  Our scoring repacks every receptor side chain within
      8 A to accommodate the peptide. A de novo binder's groove is PRE-ORGANISED -- that is
      the entire design principle -- so letting it re-rotamer for all 18 peptides may erase
      the specificity we are trying to measure. Test: score with repacking (as now) and
      with the receptor side chains held rigid, and compare cognate/non-cognate separation.

  H2  WE ARE REWARDING BURIAL, NOT COMPLEMENTARITY.  Any peptide dropped into any deep
      groove gains large fa_atr. The physics that should say "no" -- desolvation of polar
      groups that lose water without gaining an H-bond (fa_sol, lk_ball_wtd), and repulsion
      from imperfect fit (fa_rep) -- may be too weak or may cancel. Test: split the
      interface energy into ref2015 terms and ask which, if any, separates cognate from
      non-cognate. Also normalise by buried area, since energy tracks size.

Outputs per (cell, pose, variant): each interface energy term, buried heavy-atom contacts,
and energy per contact.

Usage: coventry_why_wrong.py [workers] [poses_per_cell] [noncognate_per_row]
Output: logs/coventry_why_wrong.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
POSES = ROOT / "runs/coventry/hybridock_ft"
BINDERS = ROOT / "datasets/coventry/binders"
OUT = ROOT / "logs/coventry_why_wrong.jsonl"
CUT = 8.0
TERMS = ["fa_atr", "fa_rep", "fa_sol", "fa_elec", "lk_ball_wtd",
         "hbond_sc", "hbond_bb_sc", "fa_dun", "p_aa_pp", "rama_prepro", "omega"]
_PR = None
_SF = None


def _init():
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pyrosetta
    pyrosetta.init(" ".join(["-mute", "all", "-use_input_sc", "-ignore_unrecognized_res",
                             "-ignore_zero_occupancy", "false", "-load_PDB_components",
                             "false", "-no_fconfig", "-use_terminal_residues", "true"]),
                   silent=True)
    _PR = pyrosetta
    _SF = pyrosetta.create_score_function("ref2015")


def _terms(pose):
    from pyrosetta.rosetta.core.scoring import score_type_from_name
    _SF(pose)
    e = pose.energies().total_energies()
    out = {}
    for t in TERMS:
        try:
            st = score_type_from_name(t)
            out[t] = float(e[st] * _SF.get_weight(st))
        except Exception:  # noqa: BLE001 -- term absent in this build
            out[t] = 0.0
    return out


def score_variant(receptor_pdb, pose_pdb, repack: bool):
    """Interface energy, per term, with or without receptor side-chain repacking."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover, PackRotamersMover)

    rec = _PR.pose_from_pdb(receptor_pdb)
    pep = _PR.pose_from_pdb(pose_pdb)
    n_rec = rec.total_residue()
    info = pep.pdb_info()
    for i in range(1, pep.total_residue() + 1):
        info.chain(i, "p")
    pose = rec.clone()
    pose.append_pose_by_jump(pep, n_rec)

    pep_sel = rs.ChainSelector("p")
    near = rs.NeighborhoodResidueSelector(pep_sel, CUT, False)
    movable = rs.OrResidueSelector(pep_sel, near) if repack else pep_sel
    frozen = rs.NotResidueSelector(movable)

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(), frozen))
    PackRotamersMover(_SF, tf.create_task_and_apply_taskoperations(pose)).apply(pose)

    mm = _PR.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False); mm.set_chi(False); mm.set_jump(True)
    sub = movable.apply(pose)
    for i in range(1, pose.total_residue() + 1):
        if sub[i]:
            mm.set_chi(i, True)
            if i > n_rec:
                mm.set_bb(i, True)
    mn = MinMover(mm, _SF, "lbfgs_armijo_nonmonotone", 0.01, True)
    mn.max_iter(200)
    mn.apply(pose)

    tc = _terms(pose)
    split = list(pose.split_by_chain())
    pep_out = split[-1]
    rec_only = split[0].clone()
    for c in split[1:-1]:
        rec_only.append_pose_by_jump(c, rec_only.total_residue())
    tr, tp = _terms(rec_only), _terms(pep_out)
    iface = {k: tc[k] - tr[k] - tp[k] for k in tc}
    iface["total"] = float(_SF(pose) - _SF(rec_only) - _SF(pep_out))

    # buried contacts, to normalise energy by interface size
    import numpy as np
    from scipy.spatial import cKDTree
    def xyz(p):
        return np.array([[p.residue(i).xyz(j).x, p.residue(i).xyz(j).y, p.residue(i).xyz(j).z]
                         for i in range(1, p.total_residue() + 1)
                         for j in range(1, p.residue(i).nheavyatoms() + 1)])
    a, b = xyz(rec_only), xyz(pep_out)
    d, _ = cKDTree(a).query(b)
    iface["contacts"] = int((d < 4.5).sum())
    iface["min_dist"] = float(d.min())
    return iface


def do_cell(args):
    name, receptor, n_poses = args
    d = POSES / name
    ps = sorted(d.glob("rank*.pdb"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    ps = ps[:n_poses]
    rows = []
    for p in ps:
        for repack in (True, False):
            try:
                r = score_variant(receptor, str(p), repack)
                r.update(pose=p.name, repack=repack)
                rows.append(r)
            except Exception as exc:  # noqa: BLE001
                rows.append({"pose": p.name, "repack": repack,
                             "error": f"{type(exc).__name__}: {str(exc)[:60]}"})
    return {"name": name, "rows": rows}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_poses = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_non = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    pairs = list(csv.DictReader(open(ROOT / "data/coventry_pairs.csv")))
    cog = [p for p in pairs if p["cognate"] == "1"]
    rng = random.Random(0)
    non = []
    by_pep = {}
    for p in pairs:
        if p["cognate"] == "0" and p["measured"] == "0":
            by_pep.setdefault(p["peptide"], []).append(p)
    for pep, lst in by_pep.items():
        non += rng.sample(lst, min(n_non, len(lst)))

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    jobs = []
    for p in cog + non:
        name = f"{p['peptide']}__{p['binder_label']}"
        if name in done:
            continue
        rec = BINDERS / f"{p['binder_name']}.pdb"
        if (POSES / name).is_dir() and rec.exists():
            jobs.append((name, str(rec), n_poses))
    print(f"{len(cog)} cognate + {len(non)} non-binding cells, {n_poses} poses each, "
          f"x2 variants -> {len(jobs)} cells to run", flush=True)

    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, res in enumerate(ex.map(do_cell, jobs, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if k % 10 == 0 or k == 1:
                print(f"  [{k}/{len(jobs)}] {res['name']}", flush=True)
    print("WHY_WRONG_DONE")


if __name__ == "__main__":
    main()
