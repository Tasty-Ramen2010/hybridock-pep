"""E92 — compute clean OpenMM vdW intermolecular FF energy on crystal-65 (CPU), to pair with the98.

Replaces the size-confounded full-Vina blend with the physically correct, sign-consistent intermolecular
LJ energy (same decomposition e72 used for the98). CPU only — does not touch the n=100 GPU run.
Appends to data/e92_cr65_vdw.jsonl (crash-safe).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/e92_cr65_vdw.jsonl"
_KCAL = 1.0  # decompose returns kcal/mol already


def decompose(pdb_path):
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from hybridock_pep.scoring.mmgbsa import _pdbfixer_addH
    fixed = _pdbfixer_addH(Path(pdb_path))
    pdb = app.PDBFile(str(fixed))
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    for i, f in enumerate(system.getForces()):
        cn = f.__class__.__name__
        if cn == "NonbondedForce":
            f.setForceGroup(1); nb = f
        elif "GB" in cn or "Girifalco" in cn or cn.startswith("CustomGB"):
            f.setForceGroup(2)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(pdb.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 50.0, 150)

    def grp_energy(g):
        return ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(
            unit.kilocalorie_per_mole)
    e_nb_full = grp_energy(1)
    # zero charges -> NonbondedForce group 1 becomes vdW only
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sig, eps)
    nb.updateParametersInContext(ctx)
    e_vdw = grp_energy(1)
    return float(e_vdw)


def binding_vdw(pep, rec):
    from hybridock_pep.scoring.geometry_features import _merge_complex
    cx = _merge_complex(Path(pep), Path(rec))
    return decompose(cx) - decompose(rec) - decompose(pep)


def main():
    b = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    done = {json.loads(l)["pdb"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    print(f"=== E92 clean vdW on crystal-65 ({len(b)} complexes, CPU) ===", flush=True)
    for i, meta in enumerate(b):
        pdb = meta["pdb"]
        if pdb in done:
            continue
        t0 = time.time()
        try:
            vdw = binding_vdw(meta["peptide_pdb"], meta["pocket_pdb"])
            with OUT.open("a") as fh:
                fh.write(json.dumps(dict(pdb=pdb, y=meta["dg_exp"], vdw=vdw)) + "\n")
            print(f"  {i+1}/{len(b)} {pdb}: vdw={vdw:+.1f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} FAIL {str(e)[:50]}", flush=True)
    print("=== E92 done ===", flush=True)


if __name__ == "__main__":
    main()
