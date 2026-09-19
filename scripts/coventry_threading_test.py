#!/usr/bin/env python
"""Is our score blind to peptide SEQUENCE? Thread every sequence onto one backbone.

THE DECOMPOSITION THIS GIVES US.  Our grid could be failing for two completely different
reasons and the grid itself cannot tell them apart:

  (a) we cannot produce the right POSE  -- docking failure (we know this is partly true:
      79-81% of poses thread the groove backwards), or
  (b) given a perfect pose, we still cannot tell the right SEQUENCE from a wrong one --
      scoring failure.

So remove docking from the experiment entirely. For each binder, take the best pose of its
OWN cognate peptide -- a backbone we are as happy with as we will ever be -- and mutate the
peptide in place to each of the other peptides' sequences, keeping the backbone fixed.
Repack and score. Same groove, same backbone path, same length: the ONLY thing that varies
is the amino acid sequence.

If the cognate sequence wins on its own backbone, our scoring can read sequence and the
problem is docking. If it does not, the scoring is sequence-blind and no amount of better
docking will fix the grid.

Only same-length groups are compared, so no truncation or alignment choice enters:
  group A  n1 n2 n3 n4 n7                  (14-mers)
  group B  pc2 pc11 pc12 pc18 pc17         (16-mers)

Usage: coventry_threading_test.py [workers]
Output: logs/coventry_threading.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
POSES = ROOT / "runs/coventry/hybridock_ft"
BINDERS = ROOT / "datasets/coventry/binders"
OUT = ROOT / "logs/coventry_threading.jsonl"
GROUPS = [["n1", "n2", "n3", "n4", "n7"],
          ["pc2", "pc11", "pc12", "pc18", "pc17"]]
CUT = 8.0
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


def thread_and_score(receptor_pdb: str, pose_pdb: str, seq: str) -> dict:
    """Mutate the peptide in place to `seq`, repack the interface, minimise, score."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover, PackRotamersMover)
    from pyrosetta.toolbox import mutate_residue

    rec = _PR.pose_from_pdb(receptor_pdb)
    pep = _PR.pose_from_pdb(pose_pdb)
    if pep.total_residue() != len(seq):
        return {"error": f"length {pep.total_residue()} != {len(seq)}"}
    info = pep.pdb_info()
    for i in range(1, pep.total_residue() + 1):
        info.chain(i, "p")
    n_rec = rec.total_residue()
    pose = rec.clone()
    pose.append_pose_by_jump(pep, n_rec)

    # mutate the peptide in place; backbone coordinates are untouched
    for i, aa in enumerate(seq, start=1):
        mutate_residue(pose, n_rec + i, aa, pack_radius=6.0)

    pep_sel = rs.ChainSelector("p")
    near = rs.NeighborhoodResidueSelector(pep_sel, CUT, False)
    movable = rs.OrResidueSelector(pep_sel, near)
    frozen = rs.NotResidueSelector(movable)
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(), frozen))
    task = tf.create_task_and_apply_taskoperations(pose)

    mm = _PR.rosetta.core.kinematics.MoveMap()
    # Rigid-body jump and peptide backbone are RELAXABLE, exactly as in the main grid's
    # refinement. Freezing them makes any backbone-level overlap inescapable and the energy
    # blows up to 10^4 REU for every sequence, cognate included -- which is what happened on
    # the first two attempts. Every threaded sequence gets identical freedom, so the
    # comparison stays controlled; only the amino acids differ.
    mm.set_bb(False); mm.set_chi(False); mm.set_jump(True)
    sub = movable.apply(pose)
    for i in range(1, pose.total_residue() + 1):
        if sub[i]:
            mm.set_chi(i, True)
            if i > n_rec:
                mm.set_bb(i, True)

    # Threading a bulky sequence onto a backbone shaped for a small one creates clashes a
    # single hard-repulsive pass cannot escape, and the resulting +10^4 REU swamps every
    # comparison. Standard Rosetta practice: pack against a SOFTENED repulsive term first so
    # side chains can slide past each other, then re-pack and minimise with the real one.
    # Genuine steric incompatibility survives this; recoverable rotamer conflicts do not.
    soft = _PR.create_score_function("ref2015_soft")
    PackRotamersMover(soft, task).apply(pose)
    m0 = MinMover(mm, soft, "lbfgs_armijo_nonmonotone", 0.01, True)
    m0.max_iter(100)
    m0.apply(pose)
    PackRotamersMover(_SF, tf.create_task_and_apply_taskoperations(pose)).apply(pose)
    mn = MinMover(mm, _SF, "lbfgs_armijo_nonmonotone", 0.01, True)
    mn.max_iter(300)
    mn.apply(pose)

    from pyrosetta.rosetta.core.scoring import score_type_from_name
    def terms(p):
        _SF(p)
        e = p.energies().total_energies()
        return {t: float(e[score_type_from_name(t)] * _SF.get_weight(score_type_from_name(t)))
                for t in ("fa_atr", "fa_rep", "fa_sol", "fa_elec")}
    tc = terms(pose)
    split = list(pose.split_by_chain())
    pep_out = split[-1]
    rec_only = split[0].clone()
    for c in split[1:-1]:
        rec_only.append_pose_by_jump(c, rec_only.total_residue())
    tr, tp = terms(rec_only), terms(pep_out)
    out = {k: tc[k] - tr[k] - tp[k] for k in tc}
    out["total"] = float(_SF(pose) - _SF(rec_only) - _SF(pep_out))
    return out


def job(args):
    binder_label, binder_file, pose_file, pep_name, seq = args
    try:
        r = thread_and_score(binder_file, pose_file, seq)
    except Exception as exc:  # noqa: BLE001
        r = {"error": f"{type(exc).__name__}: {str(exc)[:70]}"}
    r.update(binder=binder_label, peptide=pep_name,
             cognate=int(binder_label == pep_name), pose=Path(pose_file).name)
    return r


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tg = {r["name"]: r for r in csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    bd = {r["grid_label"][:-1]: r["name"] for r in
          csv.DictReader(open(ROOT / "data/coventry_binders.csv"))}
    refine = {}
    for line in (ROOT / "logs/coventry_refine.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            refine[d["name"]] = d

    jobs = []
    for grp in GROUPS:
        for b in grp:
            cell = f"{b}__{b}B"
            d = refine.get(cell)
            if not d:
                continue
            ok = [r for r in d["poses"] if "ref2015_interface" in r]
            if not ok:
                continue
            best = min(ok, key=lambda r: r["ref2015_interface"])["pose"]
            pf = POSES / cell / best
            bf = BINDERS / f"{bd[b]}.pdb"
            if not pf.exists() or not bf.exists():
                continue
            for p in grp:
                jobs.append((b, str(bf), str(pf), p, tg[p]["sequence"]))
    print(f"{len(jobs)} threadings: each binder's own cognate backbone, "
          f"every same-length sequence threaded onto it", flush=True)

    rows = []
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, r in enumerate(ex.map(job, jobs, chunksize=1), 1):
            rows.append(r)
            if k % 10 == 0:
                print(f"  [{k}/{len(jobs)}]", flush=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {OUT}")
    print("THREADING_DONE")


if __name__ == "__main__":
    main()
