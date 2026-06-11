"""E33 — single-pose GB DESOLVATION (Ram's cheap-answer-to-expensive-question idea).

MM-GBSA is expensive because of MD SAMPLING. But the GB solvation free energy is a
SINGLE-POINT property of one conformation: G_solv = E(solvent ε=78.5) − E(vacuum ε=1).
Compute it on ONE minimized static pose — the proper desolvation physics our COUNT features
can't capture, at a fraction of MM-GBSA cost (no MD ensemble).

Per complex (ff14SB + GBn2, minimize complex once, components at bound geometry):
  g_solv_bind : ΔG_solv(cpx) − ΔG_solv(rec) − ΔG_solv(pep)   = desolvation free energy
  dE_gas      : E_gas(cpx) − E_gas(rec) − E_gas(pep)         = gas interaction (vdw+coul)
  dG_1pose    : dE_gas + g_solv_bind                          = single-pose MM-GBSA (no entropy)

Tests universality (sign-consistent crystal-65 vs 98?) + strength vs our cheap proxies.
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
from scipy.stats import pearsonr  # noqa: E402


def _energies(pose_pdb, receptor_pdb, platform_name="CUDA"):
    """Return (g_solv_bind, dE_gas, dG_1pose) for one static pose. None on failure."""
    import openmm
    import openmm.app as app
    from hybridock_pep.scoring.mmgbsa import (_FF_FILES, _MINIMIZE_MAXITER, _MINIMIZE_TOL,
                                              _pdbfixer_addH)
    try:
        plat = openmm.Platform.getPlatformByName(platform_name); props = {}
    except Exception:
        plat = openmm.Platform.getPlatformByName("CPU"); props = {}
    ff = app.ForceField(*_FF_FILES)
    rc = _pdbfixer_addH(Path(receptor_pdb)); pc = _pdbfixer_addH(Path(pose_pdb))
    try:
        ro = app.PDBFile(str(rc)); po = app.PDBFile(str(pc))
    finally:
        for t in (rc, pc):
            if t not in (Path(receptor_pdb), Path(pose_pdb)):
                t.unlink(missing_ok=True)
    n_rec = sum(1 for _ in ro.topology.chains())
    mod = app.Modeller(ro.topology, ro.positions)
    mod.add(po.topology, po.positions)
    try:
        mod.addHydrogens(ff, pH=7.4)
    except Exception:
        pass
    topo, pos = mod.topology, mod.positions

    def energy(topology, positions, eps_solvent, minimize=False):
        system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                                 soluteDielectric=1.0, solventDielectric=eps_solvent)
        integ = openmm.LangevinMiddleIntegrator(300, 1, 0.002)
        try:
            ctx = openmm.Context(system, integ, plat, props)
        except Exception:
            ctx = openmm.Context(system, openmm.LangevinMiddleIntegrator(300, 1, 0.002),
                                 openmm.Platform.getPlatformByName("CPU"), {})
        ctx.setPositions(positions)
        if minimize:
            openmm.LocalEnergyMinimizer.minimize(ctx, _MINIMIZE_TOL, _MINIMIZE_MAXITER)
        st = ctx.getState(getEnergy=True, getPositions=True)
        e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole) * 0.239006
        return e, st.getPositions()

    # minimize complex once (in solvent), capture minimized positions
    _, minpos = energy(topo, pos, 78.5, minimize=True)

    def species_energies(topology, positions):
        e_solv, _ = energy(topology, positions, 78.5)
        e_gas, _ = energy(topology, positions, 1.0)
        return e_gas, (e_solv - e_gas)  # gas energy, solvation free energy

    cg, cs = species_energies(topo, minpos)
    # receptor-only at bound geometry
    mr = app.Modeller(topo, minpos)
    chains = list(mr.topology.chains()); mr.delete(chains[n_rec:])
    rg, rs = species_energies(mr.topology, mr.positions)
    # peptide-only at bound geometry
    mp = app.Modeller(topo, minpos)
    chains2 = list(mp.topology.chains()); mp.delete(chains2[:n_rec])
    pg, ps = species_energies(mp.topology, mp.positions)

    g_solv = cs - rs - ps
    dE_gas = cg - rg - pg
    return dict(g_solv_bind=g_solv, dE_gas=dE_gas, dG_1pose=dE_gas + g_solv)


def build(which, limit=None):
    out_path = Path(f"/tmp/e33_{which}.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if which == "cr":
        e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
        geo = json.loads(Path("/tmp/e19_cr.json").read_text())
        items = [(g["pdb"].upper(), e0[g["pdb"].upper()].get("pep_pdb"),
                  e0[g["pdb"].upper()].get("poc_pdb"), g["y"])
                 for g in geo if g["pdb"].upper() in e0]
    else:
        b98 = json.loads(Path("/tmp/e28_feats.json").read_text())
        work = Path("/tmp/ppep_work")
        items = [(k, str(work / f"{k}_pep.pdb"), str(work / f"{k}_rec.pdb"), r["y"])
                 for k, r in b98.items()]
    n = 0
    for key, pep, rec, y in items:
        if key in out or not pep or not rec or not Path(pep).exists() or not Path(rec).exists():
            continue
        try:
            e = _energies(pep, rec)
            if e and all(np.isfinite(v) for v in e.values()):
                out[key] = dict(e, y=y); n += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  {key} FAIL {type(ex).__name__}: {str(ex)[:50]}", flush=True)
        if n % 5 == 0 and n:
            out_path.write_text(json.dumps(out)); print(f"  {which} {len(out)} done", flush=True)
        if limit and len(out) >= limit:
            break
    out_path.write_text(json.dumps(out))
    return [v for v in out.values()]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("cr", "both"):
        print("=== crystal-65 single-pose desolvation ===", flush=True); build("cr")
    if which in ("b98", "both"):
        print("=== the-98 single-pose desolvation ===", flush=True); build("b98")
    # eval if both present
    cp, bp = Path("/tmp/e33_cr.json"), Path("/tmp/e33_b98.json")
    if cp.exists() and bp.exists():
        cr = list(json.loads(cp.read_text()).values()); b98 = list(json.loads(bp.read_text()).values())
        ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])
        print(f"\n=== single-pose physics: sign-consistency (cr n={len(cr)}, 98 n={len(b98)}) ===")
        for f in ["g_solv_bind", "dE_gas", "dG_1pose"]:
            rc = pearsonr([r[f] for r in cr], ycr).statistic
            r9 = pearsonr([r[f] for r in b98], y98).statistic
            print(f"  {f:<14}{rc:+.3f} / {r9:+.3f}  {'UNIVERSAL' if rc*r9>0 and min(abs(rc),abs(r9))>0.1 else 'flip/weak'}")
        print("  (cheap proxies capped ~0.3; does single-pose GB desolvation beat them?)")


if __name__ == "__main__":
    main()
