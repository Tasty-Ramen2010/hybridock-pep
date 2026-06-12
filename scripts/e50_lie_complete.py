"""E50 — COMPLETE the LIE: add the free-peptide leg (the desolvation reference e49 was missing).

e49 used the peptide energy at the BOUND (buried) geometry, so it never paid the charged
desolvation penalty. Real LIE / 3-traj MM-GBSA references the FREE, fully-solvated peptide:

    ΔG = <G_complex>_boundMD  -  <G_receptor>_boundMD  -  <G_peptide>_FREE_MD

The new quantity is <G_pep>_bound - <G_pep>_free = reorganization + DESOLVATION (large & positive
for charged peptides that lose the most GB solvation on burial). This is the term that could rescue
the charged column. ff14SB + GBn2, GPU. Per complex stores all ensemble-averaged components so a
small linear model can learn the LIE weighting. Cached/resumable. Charge-split eval vs e49.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring import mmgbsa as _m  # noqa: E402

N_FRAMES = 50
STEPS = 300
FREE_FRAMES = 60          # free peptide is small/fast; sample a bit more to relax it
FREE_STEPS = 300
USE_GPU = True


def _build(ff, pdb_path):
    """pdbfixer + PDBFile -> (topology, positions, n_chains)."""
    import openmm.app as app
    capped = _m._pdbfixer_addH(Path(pdb_path))
    try:
        obj = app.PDBFile(str(capped))
    finally:
        if capped != Path(pdb_path):
            capped.unlink(missing_ok=True)
    return obj


def bound_components(pose, rec, force_cpu):
    """Bound-complex MD; return per-frame mean (E_complex, E_rec@bound, E_pep@bound, E_int)."""
    import openmm, openmm.app as app, openmm.unit as unit
    plat, props = _m._get_platform(force_cpu)
    ff = app.ForceField(*_m._FF_FILES)
    rec_obj = _build(ff, rec); pep_obj = _build(ff, pose)
    n_rec = sum(1 for _ in rec_obj.topology.chains())
    mod = app.Modeller(rec_obj.topology, rec_obj.positions)
    mod.add(pep_obj.topology, pep_obj.positions)
    try:
        mod.addHydrogens(ff, pH=7.4)
    except Exception:  # noqa: BLE001
        pass
    topo, pos = mod.topology, mod.positions
    system = ff.createSystem(topo, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             soluteDielectric=_m._SOLUTE_DIELECTRIC, solventDielectric=_m._SOLVENT_DIELECTRIC)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(topo, system, integ, plat, props)
    sim.context.setPositions(pos); sim.minimizeEnergy(maxIterations=_m._MINIMIZE_MAXITER)
    chains = list(topo.chains()); rec_idx = list(range(n_rec)); pep_idx = list(range(n_rec, len(chains)))

    def comp(positions_q, keep):
        mm = app.Modeller(topo, positions_q); cs = list(mm.topology.chains())
        drop = [c for i, c in enumerate(cs) if i not in keep]
        if drop:
            mm.delete(drop)
        e, _ = _m._context_energy_kcal(mm.topology, mm.positions, ff, plat, props, minimize=False)
        return e

    Ec, Er, Ep = [], [], []
    for _ in range(N_FRAMES):
        sim.step(STEPS)
        st = sim.context.getState(getEnergy=True, getPositions=True)
        ec = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * _m._KJ_TO_KCAL
        p = st.getPositions()
        Ec.append(ec); Er.append(comp(p, rec_idx)); Ep.append(comp(p, pep_idx))
    Ec, Er, Ep = np.array(Ec), np.array(Er), np.array(Ep)
    eint = Ec - Er - Ep
    return dict(E_complex=float(Ec.mean()), E_rec=float(Er.mean()),
                E_pep_bound=float(Ep.mean()), E_int=float(eint.mean()), E_int_std=float(eint.std()))


def free_peptide(pose, force_cpu):
    """Free-peptide MD in GB solvent (started from the pose, relaxes); return <E_pep>_free."""
    import openmm, openmm.app as app, openmm.unit as unit
    plat, props = _m._get_platform(force_cpu)
    ff = app.ForceField(*_m._FF_FILES)
    pep = _build(ff, pose)
    mod = app.Modeller(pep.topology, pep.positions)
    try:
        mod.addHydrogens(ff, pH=7.4)
    except Exception:  # noqa: BLE001
        pass
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                             soluteDielectric=_m._SOLUTE_DIELECTRIC, solventDielectric=_m._SOLVENT_DIELECTRIC)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(mod.topology, system, integ, plat, props)
    sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=_m._MINIMIZE_MAXITER)
    E = []
    for _ in range(FREE_FRAMES):
        sim.step(FREE_STEPS)
        E.append(sim.context.getState(getEnergy=True).getPotentialEnergy()
                 .value_in_unit(unit.kilojoule_per_mole) * _m._KJ_TO_KCAL)
    return float(np.mean(E))


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e49 = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text()) if Path("/tmp/e49_ens_mmgbsa.json").exists() else {}
    cache = Path("/tmp/e50_lie.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}

    def cf(seq):
        return sum(c in "DEKR" for c in seq) / max(1, len(seq))
    todo = []
    for k, m in bench.items():
        pose = ROOT / f"logs/crystal65_n100/cr_{k}/poses/pose_0.pdb"; rec = ROOT / m["pocket_pdb"]
        if pose.exists() and rec.exists() and m.get("peptide_seq"):
            todo.append((k, pose, rec, m["dg_exp"], cf(m["peptide_seq"])))
    todo.sort(key=lambda t: t[4], reverse=True)   # high-charge first
    print(f"=== E50 complete-LIE on {len(todo)} complexes (free-peptide leg, GPU) ===", flush=True)
    for k, pose, rec, y, c in todo:
        if k in out:
            continue
        try:
            bc = bound_components(pose.resolve(), rec.resolve(), not USE_GPU)
            epf = free_peptide(pose.resolve(), not USE_GPU)
            reorg = bc["E_pep_bound"] - epf   # desolvation + reorganization (>0 = penalty)
            dg3 = bc["E_complex"] - bc["E_rec"] - epf
            out[k] = dict(y=y, cf=c, **bc, E_pep_free=epf, reorg=reorg, dg_3traj=dg3,
                          dg_single=e49.get(k, {}).get("dg_single"))
            cache.write_text(json.dumps(out))
            print(f"  {k} cf={c:.2f} y={y:+.1f} | <Eint>={bc['E_int']:+.1f} reorg={reorg:+.1f} "
                  f"dG3traj={dg3:+.1f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {k} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
    evaluate(out)


def evaluate(out):
    from scipy.stats import pearsonr
    ks = [k for k in out if out[k].get("dg_3traj") is not None]
    if len(ks) < 6:
        print(f"\n(only {len(ks)} done)"); return
    y = np.array([out[k]["y"] for k in ks]); cf = np.array([out[k]["cf"] for k in ks])
    hi = cf >= 0.30

    def col(f):
        return np.array([out[k][f] for k in ks])

    def rr(v, mask):
        vv, yy = v[mask], y[mask]
        ok = np.abs(vv) < 1e4
        return pearsonr(vv[ok], yy[ok]).statistic if ok.sum() > 3 and vv[ok].std() > 0 else float("nan")
    print(f"\n=== COMPLETE-LIE vs e49: charged-floor test (n={len(ks)}, charged={hi.sum()}) ===")
    print(f"  {'predictor':<20}{'all':>8}{'charged(cf>=.3)':>16}")
    feats = {"E_int": "e49 <E_int> (bound)", "reorg": "desolv/reorg only",
             "dg_3traj": "COMPLETE LIE (3-traj)"}
    for f, lbl in feats.items():
        v = col(f)
        print(f"  {lbl:<20}{rr(v, np.ones(len(y), bool)):>8.3f}{rr(v, hi):>16.3f}")
    # 2-feature LIE model: learn the weighting of interaction vs desolvation (LOO)
    X = np.column_stack([col("E_int"), col("reorg")])
    okm = (np.abs(X) < 1e4).all(1)
    Xm, ym = X[okm], y[okm]; p = np.zeros(len(ym))
    for i in range(len(ym)):
        tr = [j for j in range(len(ym)) if j != i]
        mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (Xm[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, ym[tr], rcond=None)
        p[i] = np.r_[1, (Xm[i] - mu) / sd] @ w
    print(f"  2-feat LIE [<E_int>,reorg] LOO  all r={pearsonr(p,ym).statistic:+.3f}")
    print("  >> does COMPLETE LIE (3-traj) beat e49 <E_int> on the CHARGED column? that's the floor.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(json.loads(Path("/tmp/e50_lie.json").read_text()))
    else:
        main()
