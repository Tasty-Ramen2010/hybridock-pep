"""E349 — PRISM Layer 3: MM-RISM charged correction. Does 3D-RISM cavity hydration flip the buried-charge sign?

E348 proved 100 ps of MD can't hydrate a buried site (1BRS −8.45 → −6.62, still wrong sign). 3D-RISM fills cavities
by the integral equation in one static-structure shot — no diffusion timescale — giving the buried-site water that
provides the high apparent dielectric (10–20) that stabilises the buried charge. This computes the charged
residue's binding contribution as an MM-RISM double-difference:

  state energy  E(state) = E_gas_MM(prmtop)  +  ΔG_solv_RISM(exchem)          [RISM includes buried cavity water]
  ΔΔG = [E(bound,charged) − E(bound,neutral)] − [E(free,charged) − E(free,neutral)]

charged = GLU/ASP (formal −1); neutral = GLH/ASH (tleap protonates the carboxylate → net 0). SKEMPI sign: ΔΔG > 0
⇒ WT charge helps binding. If RISM flips the two buried wrong-sign cases (1BRS, 1E96) toward exp where explicit-FEP
(−8, −3) could not, PRISM Layer 3 = RISM cavity hydration is the Axis-B fix. Runs on CPU (ambertools) — no GPU
contention with the campaign.

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e349_prism_rism.py
"""
from __future__ import annotations
import sys, os, subprocess, tempfile, shutil
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch, ChainSel


def _isnum(t):
    try:
        float(t); return True
    except ValueError:
        return False


def parse_exchem(stdout):
    """Total excess chemical potential (solvation free energy, kcal/mol) from rism3d.snglpnt stdout."""
    val = np.nan
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("rism_excessChemicalPotential"):
            toks = [t for t in s.split()[1:] if _isnum(t)]
            if toks:
                val = float(toks[0])
    return val

AMBER = Path("/home/igem/miniconda3/envs/ambertools")
RISM = AMBER / "bin" / "rism3d.snglpnt"; TLEAP = AMBER / "bin" / "tleap"; PDB4AMBER = AMBER / "bin" / "pdb4amber"
XVV = AMBER / "dat" / "rism1d" / "cSPCE" / "cSPCE_kh.xvv"
ENV = {**os.environ, "AMBERHOME": str(AMBER), "PATH": f"{AMBER/'bin'}:{os.environ.get('PATH','')}"}
NEUTRAL = {"ASP": "ASH", "GLU": "GLH"}
CASES = [("1BRS_A_D", "EA73Q", 1.45, -8.45), ("1E96_A_B", "DA38N", 2.16, -3.01),
         ("2PCB_A_B", "DA34N", 0.82, 4.34)]   # 2PCB = salt-bridge control (should stay ~right)
WORK = Path("/home/igem/unknown_software/runs/e349_prism"); WORK.mkdir(parents=True, exist_ok=True)


def gas_mm(prm, rst):
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile, NoCutoff
    import openmm as mm
    from openmm import unit
    p = AmberPrmtopFile(str(prm)); c = AmberInpcrdFile(str(rst))
    sysm = p.createSystem(nonbondedMethod=NoCutoff, constraints=None)
    ctx = mm.Context(sysm, mm.VerletIntegrator(1.0 * unit.femtosecond), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(c.positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


def state_energy(tag, mut, kind, charged):
    """Build one state (bound/free × charged/neutral) → E_gas_MM + exchem_RISM (kcal/mol)."""
    pdb = tag.split("_")[0]; groups = tag.split("_")[1:]
    wt, ch, resid = mut[0], mut[1], int(mut[2:-1])
    chains = "".join(groups) if kind == "bound" else next((g for g in groups if ch in g), ch)
    wd = WORK / f"{tag}_{mut}_{kind}_{'chg' if charged else 'neu'}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    st = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))
    raw = wd / "raw.pdb"; io = PDBIO(); io.set_structure(st); io.save(str(raw), ChainSel(chains))
    clean = wd / "clean.pdb"
    subprocess.run([str(PDB4AMBER), "-i", str(raw), "-o", str(clean), "--nohyd", "--dry"],
                   env=ENV, capture_output=True, timeout=300, cwd=wd)
    if not clean.exists():
        raise RuntimeError("pdb4amber failed")
    # for the NEUTRAL state, rename the target residue GLU/ASP → GLH/ASH so tleap protonates the carboxylate
    if not charged:
        lines = clean.read_text().splitlines()
        out = []
        for ln in lines:
            if ln.startswith(("ATOM", "HETATM")) and ln[21] == ch and ln[22:26].strip() == str(resid):
                rn = ln[17:20].strip()
                if rn in NEUTRAL:
                    ln = ln[:17] + f"{NEUTRAL[rn]:>3s}" + ln[20:]
            out.append(ln)
        clean.write_text("\n".join(out) + "\n")
    (wd / "leap.in").write_text(
        "source leaprc.protein.ff14SB\nmol = loadpdb clean.pdb\nsaveamberparm mol mol.prmtop mol.rst7\nquit\n")
    subprocess.run([str(TLEAP), "-f", "leap.in"], env=ENV, capture_output=True, timeout=600, cwd=wd)
    prm, rst = wd / "mol.prmtop", wd / "mol.rst7"
    if not (prm.exists() and rst.exists()):
        raise RuntimeError("tleap failed")
    p = subprocess.run([str(RISM), "--pdb", str(clean), "--prmtop", str(prm), "--rst", str(rst),
                        "--xvv", str(XVV), "--closure", "kh", "--buffer", "10", "--grdspc", "0.5,0.5,0.5",
                        "--tolerance", "1e-4"], env=ENV, capture_output=True, timeout=4000, cwd=wd, text=True)
    exchem = parse_exchem(p.stdout)
    if exchem is None or np.isnan(exchem):
        raise RuntimeError(f"RISM failed rc={p.returncode}: {p.stderr[-200:]}")
    return gas_mm(prm, rst) + exchem


def ddg(tag, mut):
    e = {}
    for kind in ("bound", "free"):
        for chg in (True, False):
            e[(kind, chg)] = state_energy(tag, mut, kind, chg)
    return (e[("bound", True)] - e[("bound", False)]) - (e[("free", True)] - e[("free", False)])


def main():
    print("=== E349 PRISM Layer 3: MM-RISM charged correction (cavity hydration) ===", flush=True)
    import time
    for tag, mut, exp, expl in CASES:
        t = time.time()
        try:
            d = ddg(tag, mut)
            flip = (expl < 0) and (d > 0)
            better = abs(d - exp) < abs(expl - exp)
            print(f"  {tag} {mut}: MM-RISM={d:+.2f}  explicit-FEP={expl:+.2f}  exp={exp:+.2f}  "
                  f"({(time.time()-t)/60:.0f}min)  "
                  f"{'SIGN FLIP toward exp ✓✓' if flip else ('closer ✓' if better else 'no help')}", flush=True)
        except Exception as ex:
            print(f"  {tag} {mut}: FAIL {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    print("\nVERDICT: if RISM flipped 1BRS/1E96 toward exp, PRISM Layer 3 = cavity hydration is the Axis-B fix; "
          "wire it as the buried-residue route. 2PCB (salt bridge) should stay near its explicit value.")


if __name__ == "__main__":
    main()
