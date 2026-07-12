"""E57 — REAL backrub backbone ensemble ΔΔG (faithful flex-ddG sampler), done right.

e56b FAILED by collapse (constant +8.8, Spearman −0.37) because it averaged WT and MUT energies
SEPARATELY then subtracted — the constant WT-cavity penalty dominated both. flex-ddG's actual trick:
compute ΔΔG PAIRED PER BACKBONE (WT and mutant threaded onto the SAME backrub snapshot), then average
the per-backbone ΔΔG. That cancels the constant backbone term within each snapshot; the surviving
variation is real side-chain discrimination.

Protocol:
  1. Build ONE backrub ensemble from WT: BackrubMover (pivots = 8Å shell of mut site) + MonteCarlo
     (ref2015, kT=0.6), K snapshots. Backrub samples realistic backbone (rigid Cα-Cα rotations) so the
     pocket CAN breathe — unlike FastRelax (over-relaxes) or constrained-relax (frozen).
  2. Per snapshot: WT pack+min shell -> dG_bind_wt[k] (cached once, mut-independent).
  3. Per mutation, per snapshot: mutate + pack+min shell -> dG_bind_mut -> ddg[k]=mut-wt[k].
  4. ΔΔG = mean_k ddg[k].

Decisive test on 3SGB pos-12 (the buried-hotspot failure case e56/e56b couldn't rank). Usage:
  e57 <cx> <n_max> <K>
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e51_skempi_ddg as E51  # noqa: E402
import e54_ref2015_ddg as E54  # noqa: E402

WORK = Path("/tmp/skempi_work")


def init():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false -ex1 -ex2", silent=True)
    return pyrosetta


def shell_resnums(pose, pno, radius=8.0):
    from pyrosetta.rosetta.core.select.residue_selector import (NeighborhoodResidueSelector,
                                                               ResidueIndexSelector)
    sel = NeighborhoodResidueSelector(ResidueIndexSelector(str(pno)), radius, True)
    mask = sel.apply(pose)
    return [i for i in range(1, pose.total_residue() + 1) if mask[i]]


def pack_min_shell(pr, pose, resnums, sf):
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select.residue_selector import (ResidueIndexSelector,
                                                               NotResidueSelector)
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    idx = ResidueIndexSelector(",".join(str(r) for r in resnums))
    tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(),
                                                  NotResidueSelector(idx)))
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
    pack = PackRotamersMover(sf)
    pack.task_factory(tf)
    pack.apply(pose)
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    for r in resnums:
        mm.set_chi(r, True)
    mn = MinMover(mm, sf, "lbfgs_armijo_nonmonotone", 1e-2, True)
    mn.apply(pose)


def build_backrub_ensemble(pr, wt_pose, pivots, K, steps=50, seed0=700):
    import pyrosetta
    from pyrosetta.rosetta.protocols.backrub import BackrubMover
    from pyrosetta.rosetta.protocols.moves import MonteCarlo
    sf = pyrosetta.get_fa_scorefxn()
    bm = BackrubMover()
    from pyrosetta.rosetta.utility import vector1_unsigned_long
    v = vector1_unsigned_long()
    for r in pivots:
        v.append(r)
    bm.set_pivot_residues(v)
    bm.clear_segments()
    p = wt_pose.clone()
    bm.add_mainchain_segments(p)
    pyrosetta.rosetta.numeric.random.rg().set_seed(seed0)
    mc = MonteCarlo(p, sf, 0.6)
    snaps = []
    for _ in range(K):
        for _ in range(steps):
            bm.apply(p)
            mc.boltzmann(p)
        snaps.append(p.clone())
    return snaps


def dump(pose, tag):
    f = WORK / f"_e57{tag}.pdb"
    pose.dump_pdb(str(f))
    return f


def main():
    cx = sys.argv[1] if len(sys.argv) > 1 else "3SGB_E_I"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    pdb, gA, gB = cx.split("_")
    muts = E51.parse_skempi(cx)[:n_max]
    exp = json.loads(Path(f"/tmp/e51_{cx}.json").read_text())
    cache = Path(f"/tmp/e57_{cx}.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    pr = init()
    import pyrosetta
    sf = pyrosetta.get_fa_scorefxn()
    wt = E51.clean_complex(pr, E51.fetch(pdb), gA, gB)

    # all mutations here share pos (SKEMPI sat-mut); pivot shell from first mut residue
    m0 = muts[0]
    pno0 = wt.pdb_info().pdb2pose(m0["chain"], m0["resnum"])
    pivots = shell_resnums(wt, pno0, 8.0)
    print(f"=== E57 {cx} backrub-ensemble K={K}, pivots={len(pivots)} shell res, {len(muts)} muts ===",
          flush=True)

    # 1+2: build ensemble + per-snapshot WT dG_bind (cached)
    if "wt_dg" not in out:
        snaps = build_backrub_ensemble(pr, wt, pivots, K)
        wt_dgs = []
        for k, s in enumerate(snaps):
            sp = s.clone()
            pack_min_shell(pr, sp, pivots, sf)
            d = E54.dg_bind(pr, dump(sp, f"_WT{k}"), gA, gB)
            sp.dump_pdb(str(WORK / f"_e57snap{k}.pdb"))  # keep backbone for mutants
            if d is not None:
                wt_dgs.append(d)
        out["wt_dg"] = wt_dgs
        cache.write_text(json.dumps(out))
        print(f"  WT per-snapshot dG_bind: mean={np.mean(wt_dgs):+.2f} sd={np.std(wt_dgs):.2f} "
              f"(n={len(wt_dgs)})", flush=True)
    wt_dgs = out["wt_dg"]
    Kok = len(wt_dgs)

    # 3: per mutation, thread onto each saved snapshot backbone
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    for m in muts:
        key = f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}"
        if key in out or key not in exp:
            continue
        try:
            ddgs = []
            for k in range(Kok):
                snapf = WORK / f"_e57snap{k}.pdb"
                if not snapf.exists():
                    continue
                sp = pyrosetta.pose_from_pdb(str(snapf))
                pno = sp.pdb_info().pdb2pose(m["chain"], m["resnum"])
                if pno == 0:
                    break
                MutateResidue(pno, E51.A1[m["mut"]]).apply(sp)
                pack_min_shell(pr, sp, shell_resnums(sp, pno, 8.0), sf)
                d = E54.dg_bind(pr, dump(sp, f"_M{k}"), gA, gB)
                if d is not None:
                    ddgs.append(d - wt_dgs[k])
            if not ddgs:
                continue
            out[key] = dict(ddg=float(np.mean(ddgs)), sd=float(np.std(ddgs)),
                            ddg_exp=m["ddg_exp"], k=len(ddgs))
            cache.write_text(json.dumps(out))
            print(f"  {key} ΔΔG={np.mean(ddgs):+.2f}±{np.std(ddgs):.2f} exp={m['ddg_exp']:+.2f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {str(e)[:50]}", flush=True)

    from scipy.stats import spearmanr
    pairs = [(v["ddg"], v["ddg_exp"]) for k, v in out.items()
             if k not in ("WT", "wt_dg") and isinstance(v, dict) and abs(v.get("ddg", 99)) < 25]
    if len(pairs) >= 5:
        p = np.array([x[0] for x in pairs])
        e = np.array([x[1] for x in pairs])
        print(f"\n=== {cx} backrub-ensemble (paired) n={len(pairs)}: "
              f"Spearman={spearmanr(p, e).statistic:+.3f} ===", flush=True)
        print("  vs e56b constrained-collapse −0.37, naive-relax +0.10, single-point −0.48", flush=True)


if __name__ == "__main__":
    main()
