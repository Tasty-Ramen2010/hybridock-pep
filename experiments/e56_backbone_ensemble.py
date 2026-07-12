"""E56 — backbone ensemble (flex-ddG's actual lever): does it rescue the physics ΔΔG?

Single-point physics fails on 3SGB (ref2015 −0.48, clash-broken models at pos 15) because mutate+pack
keeps the backbone rigid. FlexPepDock/flex-ddG relax the BACKBONE (backrub) over an ensemble. Test: per
mutation, generate K models with FastRelax(backbone+sidechain, interface) and average ref2015 dG_bind
ΔΔG. Does it (a) rescue 3SGB, (b) lift the physics correlation toward beating Δphys — the receptor-
selectivity-relevant signal (where Δphys cancels)?

Pilot on the failure case (3SGB) + the working case (1PPF). K=3 models. Usage: e56 <cx> <n> <K>.
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


def relax_mutant(pr, wt_pose, chain, resnum, mut_aa, seed):
    """Mutate + FastRelax with backbone movement in the interface neighbourhood (flex-ddG style)."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.select.residue_selector import (NeighborhoodResidueSelector,
                                                                ResidueIndexSelector)
    from pyrosetta.rosetta.core.select.movemap import MoveMapFactory
    pno = wt_pose.pdb_info().pdb2pose(chain, resnum)
    if pno == 0:
        return None
    pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
    p = wt_pose.clone()
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    MutateResidue(pno, E51.A1[mut_aa]).apply(p)
    sf = pyrosetta.get_fa_scorefxn()
    sel = NeighborhoodResidueSelector(ResidueIndexSelector(str(pno)), 8.0, True)
    mmf = MoveMapFactory()
    from pyrosetta.rosetta.core.select.movemap import move_map_action
    mmf.all_bb(False); mmf.all_chi(False)
    mmf.add_bb_action(move_map_action.mm_enable, sel)
    mmf.add_chi_action(move_map_action.mm_enable, sel)
    fr = FastRelax(sf, 1)
    fr.set_movemap_factory(mmf)
    fr.apply(p)
    return p


def main():
    cx = sys.argv[1] if len(sys.argv) > 1 else "3SGB_E_I"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    pdb, gA, gB = cx.split("_")
    muts = E51.parse_skempi(cx)[:n_max]
    exp = json.loads(Path(f"/tmp/e51_{cx}.json").read_text())
    cache = Path(f"/tmp/e56_{cx}.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    pr = E51.init_pr()
    wt = E51.clean_complex(pr, E51.fetch(pdb), gA, gB)
    if "WT" not in out:
        # WT also relaxed-ensemble averaged for a fair baseline
        ds = [E54.dg_bind(pr, _dump(pr, relax_mutant_wt(pr, wt, s), cx, "WTr%d" % s), gA, gB)
              for s in range(K)]
        ds = [d for d in ds if d is not None]
        out["WT"] = float(np.mean(ds)); cache.write_text(json.dumps(out))
        print(f"  WT relaxed dG_bind = {out['WT']:+.2f}", flush=True)
    dwt = out["WT"]
    print(f"=== E56 {cx}: backbone-ensemble ΔΔG, K={K}, {len(muts)} muts ===", flush=True)
    for m in muts:
        key = f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}"
        if key in out or key not in exp:
            continue
        try:
            dgs = []
            for s in range(K):
                mp = relax_mutant(pr, wt, m["chain"], m["resnum"], m["mut"], 1000 + s)
                if mp is None:
                    break
                d = E54.dg_bind(pr, _dump(pr, mp, cx, f"{key}r{s}"), gA, gB)
                if d is not None:
                    dgs.append(d)
            if not dgs:
                continue
            out[key] = dict(ddg=float(np.mean(dgs) - dwt), ddg_exp=m["ddg_exp"], k=len(dgs))
            cache.write_text(json.dumps(out))
            print(f"  {key} ΔΔG_bbens={np.mean(dgs)-dwt:+.2f} exp={m['ddg_exp']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {str(e)[:40]}", flush=True)
    # eval
    from scipy.stats import pearsonr, spearmanr
    pairs = [(v["ddg"], v["ddg_exp"]) for k, v in out.items() if k != "WT" and abs(v.get("ddg", 99)) < 25]
    if len(pairs) >= 5:
        p = np.array([x[0] for x in pairs]); e = np.array([x[1] for x in pairs])
        sp_single = [(exp[k]["ddg_pred"], exp[k]["ddg_exp"]) for k in out if k != "WT" and k in exp]
        ps = np.array([x[0] for x in sp_single]); es = np.array([x[1] for x in sp_single])
        print(f"\n=== {cx} backbone-ensemble (n={len(pairs)}): "
              f"Spearman={spearmanr(p,e).statistic:+.3f} (vs single-point MM-GBSA {spearmanr(ps,es).statistic:+.3f}) ===")
        print("  >> if bb-ensemble >> single-point, FlexPepDock's backbone lever rescues physics")


def relax_mutant_wt(pr, wt_pose, seed):
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
    p = wt_pose.clone()
    sf = pyrosetta.get_fa_scorefxn()
    fr = FastRelax(sf, 1)
    fr.apply(p)
    return p


def _dump(pr, pose, cx, tag):
    f = Path(f"/tmp/skempi_work/{cx}_e56{tag}.pdb")
    pose.dump_pdb(str(f))
    return f


if __name__ == "__main__":
    main()
