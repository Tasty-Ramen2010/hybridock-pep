"""E18 v2 — short-MD dynamics for structure-based entropy.

Provides:
  run_bound_dynamics(pep_pdb, poc_pdb, prod_ps)  -> per-peptide-residue RMSF + (φ,ψ,ω) samples
  run_free_dynamics(pep_pdb, prod_ps)            -> same, peptide alone (unbound reference)
  dihedral_entropy(samples)                       -> per-residue Boltzmann histogram entropy

Reuses the OpenMM/GBn2/CUDA machinery patterns from e9 (_build_ff, PDBFixer prep).
This is the real Stage 2: entropy from actual 3D fluctuation, bound vs free, replacing
the sequence-only W_unbound and the stubbed W_bound≡1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e9_md_ensemble_ie import _build_ff, KJ2KCAL  # noqa: E402

KB = 0.0019872041  # kcal/mol/K
T = 300.0
# Ramachandran bins for dihedral-histogram entropy (coarse 60° bins -> 6x6 = 36 cells).
_NBIN = 6


def _prep_generic(pdb_chains, keep_water=False):
    """PDBFixer-prep a merged set of (path, chainid) -> (topology, positions, chain_atom_idx).
    chain_atom_idx: dict chainid -> list of atom indices."""
    import openmm.app as app  # noqa: F401
    from pdbfixer import PDBFixer

    merged = Path("/tmp/e18v2_merged.pdb")
    lines = []
    for src, ch in pdb_chains:
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and (keep_water or ln[17:20] != "HOH"):
                lines.append(ln[:21] + ch + ln[22:])
    merged.write_text("\n".join(lines) + "\nEND\n")
    fixer = PDBFixer(filename=str(merged))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    chain_idx = {}
    for atom in fixer.topology.atoms():
        chain_idx.setdefault(atom.residue.chain.id, []).append(atom.index)
    return fixer.topology, fixer.positions, chain_idx


def _residue_backbone_atoms(topology, chain_id):
    """Return list per peptide residue of dict{name->atom_index} for N,CA,C and CA index list."""
    res_bb = []
    ca_idx = []
    for res in topology.residues():
        if res.chain.id != chain_id:
            continue
        d = {}
        for a in res.atoms():
            if a.name in ("N", "CA", "C"):
                d[a.name] = a.index
        if "CA" in d:
            ca_idx.append(d["CA"])
        res_bb.append(d)
    return res_bb, ca_idx


def _dihedral(p):
    """p: (4,3) array -> dihedral in degrees."""
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1n = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def _run_md_collect(topology, positions, chain_id, prod_ps, frame_every_ps=2,
                    platform_name="CUDA"):
    """Run MD, collect per-frame peptide backbone coords. Returns (rmsf[res], phipsi samples)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    ff = _build_ff()
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(T * unit.kelvin, 1 / unit.picosecond,
                                        0.002 * unit.picoseconds)
    try:
        plat = mm.Platform.getPlatformByName(platform_name)
        sim = app.Simulation(topology, system, integ, plat)
    except Exception:
        sim = app.Simulation(topology, system, integ, mm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.step(2500)  # 5 ps equil
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(e0):
        raise ValueError("equilibration diverged")

    res_bb, ca_idx = _residue_backbone_atoms(topology, chain_id)
    # residue list ordered; need consecutive N,CA,C for φ=C(i-1)-N-CA-C, ψ=N-CA-C-N(i+1)
    n_frames = max(4, prod_ps // frame_every_ps)
    steps = int(frame_every_ps / 0.002)
    ca_traj = []  # frames x nres x 3
    dih_traj = []  # frames x nres x 3 (phi,psi,omega)
    for _ in range(n_frames):
        sim.step(steps)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        if not np.isfinite(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
            break
        pos = np.array(st.getPositions().value_in_unit(unit.angstrom))
        ca_traj.append(pos[ca_idx])
        dih = []
        for i, bb in enumerate(res_bb):
            phi = psi = omega = np.nan
            try:
                if i > 0 and "C" in res_bb[i-1] and {"N", "CA", "C"} <= bb.keys():
                    phi = _dihedral(pos[[res_bb[i-1]["C"], bb["N"], bb["CA"], bb["C"]]])
                if i < len(res_bb)-1 and {"N", "CA", "C"} <= bb.keys() and "N" in res_bb[i+1]:
                    psi = _dihedral(pos[[bb["N"], bb["CA"], bb["C"], res_bb[i+1]["N"]]])
                if i < len(res_bb)-1 and "CA" in bb and "C" in bb and {"N", "CA"} <= res_bb[i+1].keys():
                    omega = _dihedral(pos[[bb["CA"], bb["C"], res_bb[i+1]["N"], res_bb[i+1]["CA"]]])
            except Exception:
                pass
            dih.append([phi, psi, omega])
        dih_traj.append(dih)
    ca_traj = np.array(ca_traj)          # F x R x 3
    dih_traj = np.array(dih_traj)        # F x R x 3
    if len(ca_traj) < 3:
        raise ValueError("too few frames")
    # per-residue RMSF after superposing each frame to the mean (simple: subtract per-frame centroid)
    cen = ca_traj - ca_traj.mean(axis=1, keepdims=True)
    mean_struct = cen.mean(axis=0)
    rmsf = np.sqrt(((cen - mean_struct) ** 2).sum(axis=2).mean(axis=0))  # per residue
    return rmsf, dih_traj


def dihedral_entropy(dih_traj):
    """Per-residue Boltzmann histogram entropy over (φ,ψ) plane. dih_traj: F x R x 3."""
    F, R, _ = dih_traj.shape
    ent = np.zeros(R)
    for r in range(R):
        phi = dih_traj[:, r, 0]
        psi = dih_traj[:, r, 1]
        ok = np.isfinite(phi) & np.isfinite(psi)
        if ok.sum() < 3:
            ent[r] = np.nan
            continue
        bins = np.linspace(-180, 180, _NBIN + 1)
        H, _, _ = np.histogram2d(phi[ok], psi[ok], bins=[bins, bins])
        p = H.flatten()
        p = p[p > 0] / p.sum()
        ent[r] = -np.sum(p * np.log(p))  # nats; Boltzmann S = kB * this
    return ent  # per-residue dihedral entropy (nats)


def run_bound_dynamics(pep_pdb, poc_pdb, prod_ps=100):
    topo, pos, _ = _prep_generic([(pep_pdb, "P"), (poc_pdb, "R")])
    rmsf, dih = _run_md_collect(topo, pos, "P", prod_ps)
    return rmsf, dihedral_entropy(dih)


def run_free_dynamics(pep_pdb, prod_ps=100):
    topo, pos, _ = _prep_generic([(pep_pdb, "P")])
    rmsf, dih = _run_md_collect(topo, pos, "P", prod_ps)
    return rmsf, dihedral_entropy(dih)


if __name__ == "__main__":
    import json
    rows = json.loads(Path("/tmp/e0_rows.json").read_text())
    rows = [r for r in rows if r.get("pep_pdb")][:3]
    for r in rows:
        try:
            rb, sb = run_bound_dynamics(r["pep_pdb"], r["poc_pdb"], 60)
            rf, sf = run_free_dynamics(r["pep_pdb"], 60)
            dS = np.nansum(sb - sf)
            print(f"{r['pdb']}: bound RMSF mean {np.nanmean(rb):.2f}Å free {np.nanmean(rf):.2f}Å "
                  f"| ΣΔS_dih(bound-free) {dS:+.2f} nats  (exp ΔG {r['y']:.1f})", flush=True)
        except Exception as e:
            print(f"{r['pdb']}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
