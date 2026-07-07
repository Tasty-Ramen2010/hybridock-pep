"""E334 — ACCURACY validation of the charged-FEP tier against SKEMPI experimental ΔΔG.

E332/E333 are PRECISE (±0.7-1.5, two routes agree to 0.86) but UNVALIDATED. This checks ACCURACY: reproduce a
MEASURED charge-mutation ΔΔG. SKEMPI has isosteric charge-neutralising mutations (Asp→Asn, Glu→Gln) = exactly
our charge morph. Target 2O3B (clean 2-chain A/B): DB75N ΔΔG_exp=+5.90, EB24Q ΔΔG_exp=+5.40 kcal.

Method (same relative charge-morph as E333, one residue): morph the mutated residue's side-chain partial charges
→ neutral on a fixed topology; compute ⟨∂U/∂morph⟩ in the COMPLEX (bound = chains A+B) and the FREE MONOMER
(the chain carrying the mutation); ΔΔG_bind = ∫[⟨∂U/∂m⟩_bound − ⟨∂U/∂m⟩_free]dm + Rocklin. Compare to experiment.
ΔΔG_exp > 0 ⇒ the WT charge helps binding (mutation weakens it).

HONEST: charge-only morph = the electrostatic part of the mutation (Asp→Asn is near-isosteric, so this is most
of it, but not the small steric/H-bond change). Fixed-charge amber14 (no polarisation) also caps accuracy.

Run:  /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e334_skempi_validation.py 2O3B_A_B DB75N 5.90
"""
from __future__ import annotations
import os, sys, tempfile, urllib.request
from collections import defaultdict
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
ROOT = "/home/igem/unknown_software"
sys.path.insert(0, ROOT + "/scripts")
from e332_g1_charged_corrected import rocklin_correction

CHARGED_SIDE = {"ASP": ["CB", "CG", "OD1", "OD2"], "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
                "LYS": ["CE", "NZ", "HZ1", "HZ2", "HZ3"], "ARG": ["CD", "NE", "CZ", "NH1", "NH2"]}


def fetch(pdb):
    f = f"{ROOT}/data/rcsb_full/{pdb.lower()}.pdb"
    if not os.path.exists(f):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", f)
    return f


class ChainSel(Select):
    def __init__(self, chains): self.chains = set(chains)
    def accept_chain(self, c): return c.id in self.chains
    def accept_residue(self, r): return r.id[0] == " "


def build(tag, mut, kind):
    """kind='bound' (all tag chains) or 'free' (mutated chain only). Returns (system, model, alch_idx, dQnet)."""
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    pdb = tag.split("_")[0]
    groups = "".join(tag.split("_")[1:])          # e.g. "AB"
    mut_chain, resid = mut[1], int(mut[2:-1])
    chains = groups if kind == "bound" else mut_chain
    st = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))
    tmp = tempfile.mktemp(suffix=".pdb")
    io = PDBIO(); io.set_structure(st); io.save(tmp, ChainSel(chains))

    fx = PDBFixer(filename=tmp)
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues()
    fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms(); fx.addMissingAtoms()
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    model = app.Modeller(fx.topology, fx.positions)
    model.addHydrogens(ff)
    model.addSolvent(ff, model="tip3p", padding=1.0 * unit.nanometer, neutralize=True)
    system = ff.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)

    # alchemical atoms = the mutated residue's SIDE CHAIN (everything except backbone), so the neutralisation
    # captures the full formal charge (~±1), not just a sub-selection of the carboxylate/amine.
    backbone = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "H1", "H2", "H3", "OXT", "HXT"}
    resname = None
    alch = []
    for a in model.topology.atoms():
        if a.residue.chain.id == mut_chain and int(a.residue.id) == resid and a.residue.name in CHARGED_SIDE:
            resname = a.residue.name
            if a.name not in backbone:
                alch.append(a.index)
    if not alch:
        raise RuntimeError(f"mutated residue {mut} not found in {kind} topology")
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    q0 = {i: nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in alch}
    shift = sum(q0.values()) / len(q0)            # neutralise the residue side chain (net → 0)
    nb.addGlobalParameter("morph", 0.0)
    for i in alch:
        nb.addParticleParameterOffset("morph", i, (q0[i] - shift) - q0[i], 0.0, 0.0)
    dQnet = -sum(q0.values())
    print(f"  [{kind}] {system.getNumParticles()} atoms, morph residue {resname}{resid} "
          f"({len(alch)} atoms, ΔQ={dQnet:+.2f}e)", flush=True)
    return system, model, alch, dQnet


def deriv_curve(system, model, morphs, n_equil, n_samp, n_stride):
    import openmm as mm
    from openmm import unit
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"))
    ctx.setPositions(model.positions); ctx.setParameter("morph", 0.0)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    L = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(unit.angstrom)
    U = lambda m: (ctx.setParameter("morph", m),
                   ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))[1]
    d, out = 0.02, []
    for m in morphs:
        ctx.setParameter("morph", m); ctx.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(n_equil)
        der = []
        for _ in range(n_samp):
            integ.step(n_stride)
            lo, hi = max(0., m - d), min(1., m + d)
            der.append((U(hi) - U(lo)) / (hi - lo)); ctx.setParameter("morph", m)
        out.append((float(np.mean(der)), float(np.std(der) / np.sqrt(len(der)))))
        print(f"    morph={m:.2f}  <dU/dm>={out[-1][0]:+8.2f} ± {out[-1][1]:.2f}", flush=True)
    return out, L


def main():
    tag, mut, exp = sys.argv[1], sys.argv[2], float(sys.argv[3])
    print(f"=== E334 SKEMPI validation: {tag} {mut}  ΔΔG_exp={exp:+.2f} kcal ===", flush=True)
    morphs = [0.0, 0.25, 0.5, 0.75, 1.0]
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    db, Lb = deriv_curve(sysb, modb, morphs, 1000, 80, 100)
    sysf, modf, af, _ = build(tag, mut, "free")
    df, Lf = deriv_curve(sysf, modf, morphs, 1000, 150, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    be = np.array([v[1] for v in db]); fe = np.array([v[1] for v in df])
    diff = bnd - fre
    _trap = getattr(np, "trapezoid", None) or np.trapz
    ddg = float(_trap(diff, morphs))
    w = np.gradient(np.array(morphs)); err = float(np.sqrt(np.sum((w * np.sqrt(be**2 + fe**2))**2)))
    corr = rocklin_correction(dQ, Lb) - rocklin_correction(dQ, Lf)
    ddg_c = ddg + corr
    print(f"\nΔΔG_bind CALC = {ddg_c:+.2f} ± {err:.2f} kcal   (raw {ddg:+.2f}, Rocklin {corr:+.2f})")
    print(f"ΔΔG_bind EXP  = {exp:+.2f} kcal (SKEMPI)")
    print(f"|calc − exp|  = {abs(ddg_c - exp):.2f} kcal   {'✓ within ~1.5' if abs(ddg_c-exp)<1.5 else '✗ off'}")


if __name__ == "__main__":
    main()
