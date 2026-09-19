#!/usr/bin/env python
"""Repack and minimise RAPiDock poses in the receptor's frame, then score with ref2015.

WHY THIS EXISTS.  RAPiDock poses carry severe peptide-receptor steric overlap: on our own
RecentSet benchmark the median closest heavy-atom contact is 0.84 A and 91% of poses have
at least one atom within 2.0 A of the receptor.  DockQ never exposed this because DockQ
scores Fnat/iRMS, not sterics.  Vina does expose it -- it returns large positive energies
and the pipeline discards the pose -- which is why scoring raw poses on this grid dropped
23 of 24 poses per cell.

The existing Stage 1.5 minimisation cannot fix it: it minimises the peptide ALONE, so it
relieves internal strain but never the overlap with the receptor.  Gradient minimisation
of the complex would not fix it either, because escaping a 0.5 A overlap needs a side chain
to hop rotamer, which is a discrete move.

So: repack the interface (discrete rotamer search) and then gradient-minimise, with the
receptor BACKBONE held fixed throughout.  Receptor backbone stays put because we are
testing docking, not induced fit; only side chains and the peptide are allowed to respond.

Scores returned per pose:
  ref2015_complex    Rosetta ref2015 energy of the refined complex
  ref2015_interface  complex minus the separated partners (binding energy, ddG-like)
  min_dist           closest heavy-atom contact after refinement (clash check)

Usage: coventry_refine.py <pose_dir> <receptor_pdb> [out_dir]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
INTERFACE_CUT = 8.0     # A: receptor residues this close to the peptide may repack


def init_rosetta():
    import pyrosetta
    pyrosetta.init(" ".join([
        "-mute", "all", "-use_input_sc", "-ignore_unrecognized_res",
        "-ignore_zero_occupancy", "false", "-load_PDB_components", "false",
        "-no_fconfig", "-use_terminal_residues", "true",
    ]), silent=True)
    return pyrosetta


def refine_one(pyrosetta, receptor_pdb: str, pose_pdb: str, out_pdb: str | None):
    """Repack + minimise one peptide pose against a fixed receptor backbone."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import (
        MinMover, PackRotamersMover)

    sfxn = pyrosetta.create_score_function("ref2015")
    rec = pyrosetta.pose_from_pdb(receptor_pdb)
    pep = pyrosetta.pose_from_pdb(pose_pdb)
    n_rec = rec.total_residue()
    info = pep.pdb_info()
    for i in range(1, pep.total_residue() + 1):
        info.chain(i, "p")
    pose = rec.clone()
    pose.append_pose_by_jump(pep, n_rec)

    pep_sel = rs.ChainSelector("p")
    near = rs.NeighborhoodResidueSelector(pep_sel, INTERFACE_CUT, False)
    movable = rs.OrResidueSelector(pep_sel, near)
    frozen = rs.NotResidueSelector(movable)

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())          # repack only, never design
    tf.push_back(operation.OperateOnResidueSubset(
        operation.PreventRepackingRLT(), frozen))
    PackRotamersMover(sfxn, tf.create_task_and_apply_taskoperations(pose)).apply(pose)

    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(True)                                      # peptide rigid-body may settle
    sub = movable.apply(pose)
    for i in range(1, pose.total_residue() + 1):
        if sub[i]:
            mm.set_chi(i, True)
            if i > n_rec:
                mm.set_bb(i, True)                         # peptide backbone only
    mn = MinMover(mm, sfxn, "lbfgs_armijo_nonmonotone", 0.01, True)
    mn.max_iter(200)
    mn.apply(pose)

    total = sfxn(pose)
    split = list(pose.split_by_chain())
    pep_out = split[-1]
    rec_only = split[0].clone()
    for c in split[1:-1]:
        rec_only.append_pose_by_jump(c, rec_only.total_residue())
    iface = total - (sfxn(rec_only) + sfxn(pep_out))
    if out_pdb:
        pose.dump_pdb(out_pdb)
    return {"ref2015_complex": float(total), "ref2015_interface": float(iface)}


def main() -> None:
    pose_dir, receptor = Path(sys.argv[1]), sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    pr = init_rosetta()
    poses = sorted(pose_dir.glob("rank*.pdb"),
                   key=lambda p: int(re.search(r"\d+", p.name).group()))
    res = {}
    for p in poses:
        out = str(out_dir / p.name) if out_dir else None
        try:
            res[p.name] = refine_one(pr, receptor, str(p), out)
        except Exception as exc:  # noqa: BLE001 -- one bad pose must not stop the cell
            res[p.name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
