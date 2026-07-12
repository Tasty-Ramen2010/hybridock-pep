"""E54 — Rosetta ref2015 ΔΔG of binding (the FlexPepDock baseline) on our saved SKEMPI mutants.

FlexPepDock = ref2015. Re-score the mutant complexes E51 already built (in /tmp/skempi_work) with
ref2015 dG_bind = E(complex) − E(groupA) − E(groupB) (rigid separation, flex-ddG style).
ΔΔG_pred = dG_bind(mut) − dG_bind(wt). This is the number to BEAT. Loads ddg_exp from e51 cache.
Output cache lets e55 build hybrids (ref2015 + MM-GBSA + ensemble) and test beating it.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/skempi_work")
CXS = ["1PPF_E_I", "1CHO_EFG_I", "1R0R_E_I", "3SGB_E_I", "1AO7_ABC_DE"]


def init():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false", silent=True)
    return pyrosetta


def dg_bind(pr, pdb_file, groupA, groupB):
    """ref2015 dG_bind = E(complex) − E(A) − E(B), rigid chain separation."""
    import pyrosetta
    sf = pyrosetta.get_fa_scorefxn()
    pose = pyrosetta.pose_from_pdb(str(pdb_file))
    e_cx = sf(pose)
    # split by chain into the two groups
    chains = pose.split_by_chain()
    eA = eB = 0.0
    from pyrosetta.rosetta.core.pose import Pose
    poseA = Pose(); poseB = Pose()
    for i in range(1, pose.num_chains() + 1):
        ch = pose.split_by_chain(i)
        # chain letter of this sub-pose
        cl = pose.pdb_info().chain(pose.chain_begin(i))
        if cl in set(groupA):
            poseA.append_pose_by_jump(ch, poseA.total_residue()) if poseA.total_residue() else poseA.assign(ch)
        elif cl in set(groupB):
            poseB.append_pose_by_jump(ch, poseB.total_residue()) if poseB.total_residue() else poseB.assign(ch)
    if poseA.total_residue() == 0 or poseB.total_residue() == 0:
        return None
    return float(e_cx - sf(poseA) - sf(poseB))


def main():
    pr = init()
    for cx in CXS:
        pdb, gA, gB = cx.split("_")
        e51 = Path(f"/tmp/e51_{cx}.json")
        if not e51.exists():
            continue
        exp = json.loads(e51.read_text())
        cache = Path(f"/tmp/e54_{cx}.json")
        out = json.loads(cache.read_text()) if cache.exists() else {}
        wtf = WORK / f"{cx}_WT.pdb"
        if not wtf.exists():
            print(f"  {cx} no WT", flush=True); continue
        if "WT" not in out:
            d = dg_bind(pr, wtf, gA, gB)
            if d is None:
                print(f"  {cx} WT split fail", flush=True); continue
            out["WT"] = d; cache.write_text(json.dumps(out))
        dwt = out["WT"]
        n = 0
        for k in exp:
            if k == "WT" or k in out:
                continue
            mf = WORK / f"{cx}_{k}.pdb"
            if not mf.exists():
                continue
            try:
                dm = dg_bind(pr, mf, gA, gB)
                if dm is None:
                    continue
                out[k] = dm; n += 1
                if n % 30 == 0:
                    cache.write_text(json.dumps(out)); print(f"  {cx}: {n} scored", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {cx} {k} FAIL {str(e)[:40]}", flush=True)
        cache.write_text(json.dumps(out))
        # eval this complex
        from scipy.stats import pearsonr, spearmanr
        pairs = [(out[k] - dwt, exp[k]["ddg_exp"]) for k in out if k != "WT" and k in exp]
        if len(pairs) >= 5:
            p = np.array([x[0] for x in pairs]); e = np.array([x[1] for x in pairs])
            print(f"  >> {cx} ref2015 ΔΔG n={len(pairs)}: Pearson={pearsonr(p,e).statistic:+.3f} "
                  f"Spearman={spearmanr(p,e).statistic:+.3f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
