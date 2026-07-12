"""E362 — the DERIVATIVE-PATH electrostatic (Ram): fix E361's charge-count artifact the way E332/E333 fixed charged FEP.

E361's raw ⟨Velec⟩_bound was a charge-COUNT artifact (corr −0.84 with n_charged): the gas-phase Coulomb sum that
"vanishes in solution" after desolvation. LIE/LRA fix it by the CHARGING derivative, bound−free, ×½ — the huge
charge-count self-energy cancels pointwise (same trick as E332/E333 difference-of-derivatives).

Proper electrostatic binding contribution (LRA, β=½; lit: charging free energy from end-state ⟨V⟩, reference =
zero-net-charge peptide + charging component):
  V_elec(state) = ⟨ E_full − E(peptide charges zeroed in BOTH Coulomb AND GB) ⟩        # peptide's FULL electrostatic
                                                                                        # incl. GB desolvation
  ΔΔG_elec = ½ [ V_elec(bound) − V_elec(free) ]                                         # bound−free cancels self-energy
The free-state V_elec includes the peptide–water (GB) solvation that COMPENSATES the raw Coulomb — this is the
desolvation term E361 omitted. Multi-window MD ensemble at each state (not single pose).

GATE: does ΔΔG_elec (a) LOSE the charge-count artifact [corr with n_charged → ~0, vs raw −0.84], and (b) gain
affinity signal / help the scorer? Run on the wide-range stratified set.

Run: OMP_NUM_THREADS=2 python experiments/e362_derivative_elec.py --gate --n 20
"""
from __future__ import annotations
import sys, json, argparse, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
import e360_prism_s_optimized as e360
import e358_conformational_entropy as e358
import e361_prism_ensemble as e361
import openmm as mm
from openmm import unit
KJ2KCAL = 0.239006


def _forces(system):
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    gb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "CustomGBForce")
    return nb, gb


def v_elec(ctx, nb, gb, pep_idx):
    """Peptide's FULL electrostatic (Coulomb+GB) = E_full − E(peptide charges zeroed in NB and GB). kcal/mol."""
    e_full = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    snb, sgb = {}, {}
    for i in pep_idx:
        q, s, e = nb.getParticleParameters(i); snb[i] = (q, s, e)
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, s, e)
        p = list(gb.getParticleParameters(i)); sgb[i] = list(p); p[0] = 0.0; gb.setParticleParameters(i, p)
    nb.updateParametersInContext(ctx); gb.updateParametersInContext(ctx)
    e_off = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    for i in pep_idx:
        nb.setParticleParameters(i, *snb[i]); gb.setParticleParameters(i, sgb[i])
    nb.updateParametersInContext(ctx); gb.updateParametersInContext(ctx)
    return (e_full - e_off) * KJ2KCAL


def state_velec(pdb, chains, pep_chain, bound, windows, ps, stride_frames=8):
    ff, top, pos, system, res_atoms = e360._build_hmr(pdb, chains, pep_chain, bound)
    nb, gb = _forces(system)
    pep_idx = [a.index for a in top.atoms() if a.residue.chain.id == pep_chain]
    n_frames = int(ps / 0.4); equil = int(50 / 0.004)
    vals = []
    for w in range(windows):
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond)
        ctx = mm.Context(system, integ, e360.PLATFORM, e360.PROPS); ctx.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
        integ.setTemperature(350 * unit.kelvin); ctx.setVelocitiesToTemperature(350 * unit.kelvin, 7 * w + 1); integ.step(2500)
        integ.setTemperature(300 * unit.kelvin); ctx.setVelocitiesToTemperature(300 * unit.kelvin, 7 * w + 3); integ.step(equil)
        for f in range(n_frames):
            integ.step(100)
            if f % stride_frames == 0:
                vals.append(v_elec(ctx, nb, gb, pep_idx))
    return float(np.mean(vals))


def ddg_elec(pdb, seq, windows, ps):
    ch = e358.find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    vb = state_velec(pdb, pep + rec, pep, True, windows, ps)
    vf = state_velec(pdb, pep, pep, False, windows, ps)
    return 0.5 * (vb - vf), vb, vf      # LRA β=½; bound−free cancels the charge-count self-energy


def _gate(out):
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    from e361_prism_ensemble import scorer_pred
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    y = np.array([o["y"] for o in out]); dg = np.array([o["ddg_elec"] for o in out])
    seqs = [o["seq"] for o in out]
    nq = np.array([sum(1 for a in s if a in "DEKR") for s in seqs])
    sc = np.array([scorer_pred(pep[o["pdb"]]) for o in out]); n = len(y)
    print(f"\n=== E362 derivative-path electrostatic  (n={n}, wide y std={y.std():.2f}) ===")
    print(f"  corr(ΔΔG_elec, n_charged) = {pearsonr(dg,nq)[0]:+.3f}   <- WAS −0.838 for raw Velec; want ≈0 (artifact gone)")
    print(f"  corr(ΔΔG_elec, affinity)  = {pearsonr(dg,y)[0]:+.3f}   <- WAS +0.14 raw; want more negative (favorable elec→tighter)")
    def loo(F):
        F = np.atleast_2d(F).T if F.ndim == 1 else F; p = np.zeros(n)
        for tr, te in LeaveOneOut().split(F):
            p[te] = LinearRegression().fit(F[tr], y[tr]).predict(F[te])
        return p
    for tag, F in [("scorer", sc), ("scorer+ΔΔG_elec", np.column_stack([sc, dg]))]:
        p = loo(F); print(f"  {tag:18s} r={pearsonr(p,y)[0]:+.3f} MAE={np.mean(np.abs(p-y)):.2f} RMSE={np.sqrt(np.mean((p-y)**2)):.2f}")
    print(f"  mean-base          MAE={np.mean(np.abs(y-y.mean())):.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--windows", type=int, default=2); ap.add_argument("--ps", type=int, default=200)
    a = ap.parse_args()
    print(f"platform {e360.PLATNAME}, derivative-path elec, {a.windows}×{a.ps}ps", flush=True)
    if a.smoke:
        for r in [json.loads(l) for l in open("data/pdbbind_peptides.jsonl")][:1]:
            t = time.time(); d, vb, vf = ddg_elec(r["pdb"], r["seq"], a.windows, a.ps)
            print(f"{r['pdb']}: ΔΔG_elec={d:+.1f} (Vb={vb:+.0f} Vf={vf:+.0f}) ({(time.time()-t)/60:.1f}m)", flush=True)
        return
    sub = e361.stratified(a.n)
    out = []
    for i, r in enumerate(sub):
        t = time.time()
        try:
            d, vb, vf = ddg_elec(r["pdb"], r["seq"], a.windows, a.ps)
            out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "ddg_elec": d, "vb": vb, "vf": vf})
            print(f"[{i+1}/{a.n}] {r['pdb']} y={float(r['y']):.1f} ΔΔG_elec={d:+.1f} (Vb={vb:+.0f} Vf={vf:+.0f}) ({(time.time()-t)/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{a.n}] {r['pdb']} FAIL {str(e)[:50]}", flush=True)
        json.dump(out, open("data/e362_derivative_elec.json", "w"))
    if a.gate and len(out) >= 6:
        _gate(out)


if __name__ == "__main__":
    main()
