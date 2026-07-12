"""E344 — continuum route: charge-morph TI in GBn2 IMPLICIT solvent (optionally + ECC), the desolvation fix.

ECC (E343) removed the fixed-charge salt-bridge OVERstabilization but left a residual on 2PCB (+4.3 vs +0.8) —
that residual is the desolvation self-energy subtraction, the exact thing explicit-water FEP got wrong (two
~90 kcal water legs that never converged). Implicit solvent computes the desolvation reaction field ANALYTICALLY:
no explicit water to sample, no PME net-charge finite-size artifact (no periodic box → no Rocklin term). This is
Ram's "compute the charged term with the right tool" done as a minimal change — same charge morph, GBn2 instead of
TIP3P/PME. --ecc adds the 0.75× charge scaling on top (continuum desolvation + electronic screening together).
  If 2PCB/3HFM land near exp → continuum is the charged-term engine; bolt it onto the neutral scorer.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e344_implicit_charged.py [--ecc]
"""
from __future__ import annotations
import sys, time, tempfile, argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch, ChainSel, CHARGED_SIDE
import openmm as mm
from openmm import app, unit
PLAT = mm.Platform.getPlatformByName("CUDA")

MORPHS = [0.0, 0.25, 0.5, 0.75, 1.0]
CASES = [("1IAR_A_B", "EA9Q", 3.11, -4.37),
         ("3HFM_HL_Y", "DY101N", 1.34, +4.28),
         ("2PCB_A_B", "DA34N", 0.82, +9.50)]


def build_implicit(tag, mut, kind, ecc=False):
    """Charge-morph system in GBn2 implicit solvent (no explicit water, no periodic box)."""
    from pdbfixer import PDBFixer
    pdb = tag.split("_")[0]
    groups = tag.split("_")[1:]
    mut_chain, resid = mut[1], int(mut[2:-1])
    chains = "".join(groups) if kind == "bound" else next((g for g in groups if mut_chain in g), mut_chain)
    st = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))
    tmp = tempfile.mktemp(suffix=".pdb"); io = PDBIO(); io.set_structure(st); io.save(tmp, ChainSel(chains))
    fx = PDBFixer(filename=tmp)
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms(); fx.addMissingAtoms()
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    model = app.Modeller(fx.topology, fx.positions); model.addHydrogens(ff)
    system = ff.createSystem(model.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    backbone = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "H1", "H2", "H3", "OXT", "HXT"}
    alch, resname = [], None
    for a in model.topology.atoms():
        if a.residue.chain.id == mut_chain and int(a.residue.id) == resid and a.residue.name in CHARGED_SIDE:
            resname = a.residue.name
            if a.name not in backbone:
                alch.append(a.index)
    if not alch:
        raise RuntimeError(f"mutated residue {mut} not found in {kind} topology")
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    gb = next(system.getForce(i) for i in range(system.getNumForces())
              if "GB" in system.getForce(i).__class__.__name__ or "Gmp" in system.getForce(i).__class__.__name__)
    q0 = {i: nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in alch}
    shift = sum(q0.values()) / len(q0)
    nb.addGlobalParameter("morph", 0.0)
    for i in alch:
        nb.addParticleParameterOffset("morph", i, (q0[i] - shift) - q0[i], 0.0, 0.0)
    # GB charge must track the morph too — CustomGBForce has a per-particle "charge"; add an energy-derivative-free
    # update via a parallel global? Simplest: use GBn2 built-in (GBSAOBCForce/CustomGBForce) with a computed
    # global. OpenMM's CustomGBForce supports addGlobalParameter + per-particle; we instead re-set GB charge each
    # window in the driver (see set_gb_charge).
    if ecc:
        for i in range(nb.getNumParticles()):
            q, s, e = nb.getParticleParameters(i); nb.setParticleParameters(i, q * 0.75, s, e)
        for k in range(nb.getNumParticleParameterOffsets()):
            p, idx, dq, ds, de = nb.getParticleParameterOffset(k)
            nb.setParticleParameterOffset(k, p, idx, dq * 0.75, ds, de)
    print(f"  [{kind}] {system.getNumParticles()} atoms, morph {resname}{resid} ({len(alch)} atoms), "
          f"GB={gb.__class__.__name__}, ecc={ecc}", flush=True)
    return system, model, nb, gb, alch, q0, shift


def deriv_curve(system, model, nb, gb, alch, q0, shift, morphs, n_equil, n_samp, n_stride, ecc):
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, PLAT); ctx.setPositions(model.positions); ctx.setParameter("morph", 0.0)
    sc = 0.75 if ecc else 1.0
    is_custom = gb.__class__.__name__ == "CustomGBForce"

    def set_gb(m):
        for i in alch:
            q = (q0[i] + m * ((q0[i] - shift) - q0[i])) * sc
            if is_custom:
                p = list(gb.getParticleParameters(i)); p[0] = q; gb.setParticleParameters(i, p)
            else:
                _, r, sr = gb.getParticleParameters(i); gb.setParticleParameters(i, q, r, sr)
        gb.updateParametersInContext(ctx)

    set_gb(0.0); mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)

    def U(m):
        ctx.setParameter("morph", m); set_gb(m)
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    d, out = 0.02, []
    for m in morphs:
        ctx.setParameter("morph", m); set_gb(m)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(n_equil)
        der = []
        for _ in range(n_samp):
            integ.step(n_stride)
            lo, hi = max(0., m - d), min(1., m + d)
            der.append((U(hi) - U(lo)) / (hi - lo)); ctx.setParameter("morph", m); set_gb(m)
        out.append(float(np.mean(der)))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1]:+8.2f}", flush=True)
    return out


def one(tag, mut, ecc):
    sb = build_implicit(tag, mut, "bound", ecc)
    db = deriv_curve(*sb, MORPHS, 1500, 80, 100, ecc)
    sf = build_implicit(tag, mut, "free", ecc)
    df = deriv_curve(*sf, MORPHS, 1500, 120, 100, ecc)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(np.array(db) - np.array(df), MORPHS))   # no Rocklin: implicit, no periodic box


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ecc", action="store_true"); a = ap.parse_args()
    print(f"=== E344 GBn2 IMPLICIT charge-morph TI (ecc={a.ecc}) vs exp ===", flush=True)
    res = []
    for tag, mut, exp, frozen in CASES:
        vals = []
        for rep in range(2):
            t = time.time()
            try:
                vals.append(one(tag, mut, a.ecc))
                print(f"  {tag} {mut} rep{rep}: calc={vals[-1]:+.2f}  ({(time.time()-t)/60:.1f}min)", flush=True)
            except Exception as e:
                print(f"  {tag} {mut} rep{rep}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        if vals:
            calc = float(np.mean(vals)); sp = float(np.std(vals)) if len(vals) > 1 else 0.0
            res.append((tag, mut, exp, frozen, calc, sp))
            print(f"  => {tag} {mut}: GB={calc:+.2f}±{sp:.2f}  explicit-frozen={frozen:+.2f}  exp={exp:+.2f}  "
                  f"|Δ|={abs(calc-exp):.2f}", flush=True)
    print("\n=== SUMMARY: does continuum solvent fix the desolvation subtraction? ===")
    print(f"{'case':16s} {'GB-implicit':>12s} {'expl-froz':>9s} {'exp':>6s} {'|err|':>6s}")
    for tag, mut, exp, frozen, calc, sp in res:
        print(f"{tag+' '+mut:16s} {calc:+.2f}±{sp:.2f}  {frozen:+.2f}  {exp:+.2f}  {abs(calc-exp):.2f}")
    if len(res) >= 3:
        c = np.array([r[4] for r in res]); e = np.array([r[2] for r in res])
        print(f"\nMAE GB-implicit = {np.mean(np.abs(c-e)):.2f}  (explicit-frozen was 6.37, ECC-explicit was 4.24)")


if __name__ == "__main__":
    main()
