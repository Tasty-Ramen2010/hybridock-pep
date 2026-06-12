"""E72 — ELECTROSTATIC decomposition (the term that should separate strong/weak charged binders).

Ram: don't sample entropy — compute ELECTROSTATICS properly and learn it. MM-GBSA's binding energy
hides three physically distinct parts; we split them with OpenMM force groups + the zero-charge trick:
  E_vdw      Lennard-Jones (shape/packing)
  E_coul     direct Coulomb between partial charges (raw salt-bridge attraction)
  E_gbpol    GB polar solvation = the DESOLVATION PENALTY a charge pays to leave water
  net_elec = E_coul + E_gbpol   (the physically meaningful electrostatic binding contribution)

Binding Δ = E(complex) − E(receptor) − E(peptide) for each component. The hypothesis: raw Coulomb
FLIPS (docs E47) because it ignores desolvation; net_elec (coulomb screened by desolvation) should be
sign-stable and should DIFFERENTIATE strong vs weak on the CHARGED subset where we hit the floor.

Single-point first (fast, establishes if the static electrostatic split has signal). If net_elec works
on charged complexes -> wire it. If not -> the floor needs MD-averaged electrostatics (LIE-elec), and we
record per-frame E_coul/E_gbpol (NOT entropy) for the surrogate.

Reuses mmgbsa system build. the-98 poses in /tmp/ppep_work. Cached to /tmp/e72_elec.json.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

CACHE = Path("/tmp/e72_elec.json")
_KJ = 0.2390057361


def decompose(pdb_path):
    """Return (E_vdw, E_coul, E_gbpol) in kcal/mol for one structure (ff14SB + GBn2)."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from hybridock_pep.scoring.mmgbsa import _pdbfixer_addH

    fixed = _pdbfixer_addH(Path(pdb_path))
    pdb = app.PDBFile(str(fixed))
    ff = app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff, pH=7.0)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    # assign force groups: NonbondedForce -> 1, GB force -> 2
    nb = None
    for i, f in enumerate(system.getForces()):
        cn = f.__class__.__name__
        if cn == "NonbondedForce":
            f.setForceGroup(1); nb = f
        elif "GB" in cn or "CustomGBForce" in cn:
            f.setForceGroup(2)
        else:
            f.setForceGroup(0)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    except Exception:
        ctx = openmm.Context(system, integ)
    ctx.setPositions(mod.positions)
    # minimize to relieve pose clashes (raw poses blow up LJ); same as MM-GBSA's single-traj
    openmm.LocalEnergyMinimizer.minimize(ctx, 50.0, 150)

    def grp_energy(g):
        return ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) * _KJ

    e_nb_full = grp_energy(1)
    e_gbpol = grp_energy(2)
    # zero charges -> NonbondedForce group 1 becomes vdW only
    for idx in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sig, eps)
    for idx in range(nb.getNumExceptions()):
        p1, p2, qq, sig, eps = nb.getExceptionParameters(idx)
        nb.setExceptionParameters(idx, p1, p2, 0.0, sig, eps)
    nb.updateParametersInContext(ctx)
    e_vdw = grp_energy(1)
    e_coul = e_nb_full - e_vdw
    return float(e_vdw), float(e_coul), float(e_gbpol)


def binding_decomp(pep, rec):
    """Δcomponent = complex − receptor − peptide. Needs a merged complex pdb."""
    from hybridock_pep.scoring.geometry_features import _merge_complex
    cx = _merge_complex(Path(pep), Path(rec))
    vc, cc, gc = decompose(cx)
    vr, cr_, gr = decompose(rec)
    vp, cp, gp = decompose(pep)
    return dict(vdw=vc - vr - vp, coul=cc - cr_ - cp, gbpol=gc - gr - gp,
                net_elec=(cc - cr_ - cp) + (gc - gr - gp))


def main():
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [k for k in e49 if k not in out and (work / f"{k}_pep.pdb").exists()]
    print(f"=== E72 electrostatic decomposition. cached={len(out)} todo={len(todo)} ===", flush=True)
    for i, k in enumerate(todo):
        try:
            d = binding_decomp(work / f"{k}_pep.pdb", work / f"{k}_rec.pdb")
            v = e49[k]
            d.update(y=v["y"], cf=v["cf"], seq=v["seq"], L=v["L"],
                     net_charge=v["seq"].count("K") + v["seq"].count("R") - v["seq"].count("D") - v["seq"].count("E"))
            out[k] = d
            if i % 10 == 0:
                CACHE.write_text(json.dumps(out))
                print(f"  {i}/{len(todo)} {k}: vdw={d['vdw']:+.0f} coul={d['coul']:+.0f} "
                      f"gbpol={d['gbpol']:+.0f} net_elec={d['net_elec']:+.0f}", flush=True)
        except Exception as ex:
            print(f"  {k} FAIL {str(ex)[:50]}", flush=True)
    CACHE.write_text(json.dumps(out))

    rows = list(out.values())
    if len(rows) < 10:
        print("not enough scored yet"); return
    charged = [r for r in rows if abs(r["net_charge"]) >= 2]
    low = [r for r in rows if abs(r["net_charge"]) < 2]
    print(f"\n=== component vs ΔG (Spearman). all={len(rows)} charged(|Q|>=2)={len(charged)} low={len(low)} ===")
    print(f"{'component':<12}{'ALL':>9}{'charged':>9}{'low-Q':>9}")
    for f in ["vdw", "coul", "gbpol", "net_elec"]:
        a = spearmanr([r[f] for r in rows], [r["y"] for r in rows]).statistic
        c = spearmanr([r[f] for r in charged], [r["y"] for r in charged]).statistic if len(charged) > 5 else float("nan")
        l = spearmanr([r[f] for r in low], [r["y"] for r in low]).statistic if len(low) > 5 else float("nan")
        flag = "  <== separates charged!" if (not np.isnan(c) and abs(c) > 0.3) else ""
        print(f"  {f:<10}{a:>+9.3f}{c:>+9.3f}{l:>+9.3f}{flag}")
    # per-residue intensive versions
    print("\n  intensive (component / L):")
    for f in ["coul", "gbpol", "net_elec"]:
        for r in rows:
            r[f + "_L"] = r[f] / max(1, r["L"])
        c = spearmanr([r[f + "_L"] for r in charged], [r["y"] for r in charged]).statistic if len(charged) > 5 else float("nan")
        print(f"  {f+'_L':<12} charged={c:+.3f}")
    print("\n  >> net_elec separating strong/weak on CHARGED = the floor cracks (single-point).")
    print("  >> if flat on charged -> need MD-averaged electrostatics (record per-frame coul/gbpol).")


if __name__ == "__main__":
    main()
