"""E56b — CONSTRAINED-relax backbone ensemble (flex-ddG's actual recipe).

Naive FastRelax over-relaxed (compressed ΔΔG). flex-ddG restrains backbone to start coords. Here:
mutate + FastRelax with constrain_relax_to_start_coords (set at init), local movemap, K models,
average ref2015 dG_bind ΔΔG. Tests if constrained backbone rescues 3SGB physics (incl pos-15 that
broke single-point). Self-contained init (no /tmp deps). Usage: e56b <cx> <n> <K>.
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
sys.path.insert(0, str(ROOT / "src"))
import e51_skempi_ddg as E51  # noqa: E402
import e54_ref2015_ddg as E54  # noqa: E402


def init_constrained():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false -relax:constrain_relax_to_start_coords "
                   "-relax:coord_constrain_sidechains -relax:ramp_constraints false", silent=True)
    return pyrosetta


def relax(pr, wt_pose, chain, resnum, mut_aa, seed):
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    from pyrosetta.rosetta.core.select.residue_selector import (NeighborhoodResidueSelector,
                                                                ResidueIndexSelector)
    from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
    p = wt_pose.clone() if mut_aa is None else None
    if mut_aa is not None:
        pno = wt_pose.pdb_info().pdb2pose(chain, resnum)
        if pno == 0:
            return None
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
        p = wt_pose.clone()
        MutateResidue(pno, E51.A1[mut_aa]).apply(p)
        sel = NeighborhoodResidueSelector(ResidueIndexSelector(str(pno)), 8.0, True)
    else:
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
        sel = None
    sf = pyrosetta.get_fa_scorefxn()
    from pyrosetta.rosetta.core.scoring import ScoreType
    sf.set_weight(ScoreType.coordinate_constraint, 1.0)
    fr = FastRelax(sf, 1)
    if sel is not None:
        mmf = MoveMapFactory(); mmf.all_bb(False); mmf.all_chi(False)
        mmf.add_bb_action(move_map_action.mm_enable, sel)
        mmf.add_chi_action(move_map_action.mm_enable, sel)
        fr.set_movemap_factory(mmf)
    fr.apply(p)
    return p


def dump(pose, cx, tag):
    f = Path(f"/tmp/skempi_work/{cx}_e56b{tag}.pdb"); pose.dump_pdb(str(f)); return f


def main():
    cx = sys.argv[1] if len(sys.argv) > 1 else "3SGB_E_I"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    pdb, gA, gB = cx.split("_")
    muts = E51.parse_skempi(cx)[:n_max]
    exp = json.loads(Path(f"/tmp/e51_{cx}.json").read_text())
    cache = Path(f"/tmp/e56b_{cx}.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    pr = init_constrained()
    wt = E51.clean_complex(pr, E51.fetch(pdb), gA, gB)
    if "WT" not in out:
        ds = []
        for s in range(K):
            wp = relax(pr, wt, None, None, None, 500 + s)
            d = E54.dg_bind(pr, dump(wp, cx, f"WT{s}"), gA, gB)
            if d is not None:
                ds.append(d)
        out["WT"] = float(np.mean(ds)); cache.write_text(json.dumps(out))
        print(f"  WT constrained-relax dG_bind={out['WT']:+.2f}", flush=True)
    dwt = out["WT"]
    print(f"=== E56b {cx} constrained-relax K={K}, {len(muts)} muts ===", flush=True)
    for m in muts:
        key = f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}"
        if key in out or key not in exp:
            continue
        try:
            dgs = []
            for s in range(K):
                mp = relax(pr, wt, m["chain"], m["resnum"], m["mut"], 1000 + s)
                if mp is None:
                    break
                d = E54.dg_bind(pr, dump(mp, cx, f"{key}_{s}"), gA, gB)
                if d is not None:
                    dgs.append(d)
            if not dgs:
                continue
            out[key] = dict(ddg=float(np.mean(dgs) - dwt), ddg_exp=m["ddg_exp"])
            cache.write_text(json.dumps(out))
            print(f"  {key} ΔΔG={np.mean(dgs)-dwt:+.2f} exp={m['ddg_exp']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {str(e)[:40]}", flush=True)
    from scipy.stats import spearmanr
    pairs = [(v["ddg"], v["ddg_exp"]) for k, v in out.items() if k != "WT" and abs(v.get("ddg", 99)) < 25]
    if len(pairs) >= 5:
        p = np.array([x[0] for x in pairs]); e = np.array([x[1] for x in pairs])
        print(f"\n=== {cx} constrained-relax n={len(pairs)}: Spearman={spearmanr(p,e).statistic:+.3f} ===", flush=True)


if __name__ == "__main__":
    main()
