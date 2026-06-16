"""E229 — THE pocket-water MD pilot (Ram's 500ns idea, scoped to a checkable test). For each of the 23
multi-binder receptors (data/e228_pilot_manifest.json): carve the apo pocket, solvate in explicit TIP3P,
restrain protein heavy atoms (WaterMap-style: fixed pocket, free water), run short MD, and measure
GIST-LITE hydration descriptors of the pocket water:

  occ           mean # waters in the pocket sphere
  mean_hb       mean polar-neighbor count per pocket water  (H-bond satisfaction; LOW = unhappy/displaceable)
  frac_unhappy  fraction of pocket waters with < 3 polar neighbors  (the WaterMap "displaceable" waters)
  enclosure     mean # protein heavy atoms within 4.5 A of a pocket water  (burial of the hydration shell)

Then test: do these predict the receptor BASELINE (mean affinity) past the static-sequence wall (0.15)?
A pocket full of unhappy/enclosed water should bind peptides better (displacing it pays).

Usage:
  python3 scripts/e229_pocket_md_pilot.py --smoke              # 1 receptor, tiny MD, CPU — validate pipeline
  python3 scripts/e229_pocket_md_pilot.py --platform CUDA      # full 23-receptor run (background)
  python3 scripts/e229_pocket_md_pilot.py --eval-only          # just re-run the correlation from cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e180_protdcal_925 as e180  # noqa: E402

MANIFEST = ROOT / "data" / "e228_pilot_manifest.json"
CACHE = ROOT / "data" / "e229_hydration.jsonl"
WORK = ROOT / "runs" / "e229_md"; WORK.mkdir(parents=True, exist_ok=True)
POCKET_R = 16.0     # A around binding site kept for MD
WATER_R = 8.0       # A pocket sphere for hydration analysis
T2O = e180.T2O


def site_and_apo(pdb, pepseq, pep_ch):
    """From the cached RCSB structure: binding-site centroid (peptide-atom mean) + apo receptor PDB path."""
    from Bio.PDB import PDBIO, PDBParser, Select
    f = e180.fetch(pdb)
    if f is None:
        return None, None
    st = PDBParser(QUIET=True).get_structure(pdb, str(f))
    model = st[0]
    if pep_ch not in [c.id for c in model]:
        return None, None
    pep_atoms = [a.coord for r in model[pep_ch] if r.id[0] == " " for a in r]
    if not pep_atoms:
        return None, None
    site = np.mean(pep_atoms, axis=0)

    # Keep WHOLE chains that come near the site (no per-residue cut → no severed/mis-protonated residues).
    near_chains = set()
    for c in model:
        if c.id == pep_ch:
            continue
        if any(np.linalg.norm(a.coord - site) <= POCKET_R for r in c if r.id[0] == " " for a in r):
            near_chains.add(c.id)

    class ApoNear(Select):
        def accept_chain(self, c):
            return c.id in near_chains

        def accept_residue(self, r):
            return r.id[0] == " "   # standard residues only, but keep the chain whole
    apo = WORK / f"{pdb}_apo.pdb"
    io = PDBIO(); io.set_structure(st); io.save(str(apo), ApoNear())
    return site, apo


def build_and_run(apo_pdb, site, n_prod_steps, report_every, platform_name):
    """Fix → solvate → restrain heavy atoms → minimize → equilibrate → production; collect frame positions
    of (water-O, protein polar, protein heavy). Returns dict of accumulated descriptors.
    `site` = binding-site centroid (A) in the original-PDB frame; used to id the groove-lining protein
    atoms whose live centroid defines the pocket each frame (translation-robust)."""
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(apo_pdb))
    fixer.findMissingResidues(); fixer.missingResidues = {}      # do NOT rebuild big loops
    fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    mod = app.Modeller(fixer.topology, fixer.positions)
    # BEFORE solvation (coords still in original frame): groove-lining protein atoms within 8 A of the site.
    pre = np.array([[v.x, v.y, v.z] for v in mod.positions]) * 10.0   # nm → A
    groove = [a.index for a in mod.topology.atoms()
              if a.element is not None and a.element.symbol != "H"
              and np.linalg.norm(pre[a.index] - site) <= 8.0]
    if len(groove) < 3:
        return None
    mod.addSolvent(ff, model="tip3p", padding=1.0 * unit.nanometer, neutralize=True)

    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    # restrain protein heavy atoms (WaterMap-style fixed pocket)
    prot_res = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "CYX"}
    restr = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restr.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    for p in ("x0", "y0", "z0"):
        restr.addPerParticleParameter(p)
    pos = mod.positions
    prot_heavy, prot_polar, water_o = [], [], []
    for atom in mod.topology.atoms():
        rn = atom.residue.name
        if rn in prot_res:
            if atom.element is not None and atom.element.symbol != "H":
                prot_heavy.append(atom.index)
                p = pos[atom.index].value_in_unit(unit.nanometer)
                restr.addParticle(atom.index, [p[0], p[1], p[2]])
                if atom.element.symbol in ("N", "O"):
                    prot_polar.append(atom.index)
        elif rn in ("HOH", "WAT") and atom.name in ("O", "OW"):
            water_o.append(atom.index)
    system.addForce(restr)

    # NVT (no barostat: box already at standard water density; barostat + position-restraint = instability).
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    try:
        plat = mm.Platform.getPlatformByName(platform_name)
    except Exception:  # noqa: BLE001
        plat = mm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(mod.topology, system, integ, plat)
    sim.context.setPositions(mod.positions)
    # PHASE 1 — hydrate the dry cleft: LOOSE restraint (keeps the truncated fragment intact while bulk
    # water diffuses into the pocket). Heavy minimization first to relax truncation/added-atom clashes.
    sim.context.setParameter("k", 50.0)
    sim.minimizeEnergy(tolerance=10 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=2000)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    for _kt in (1.0, 5.0, 50.0):               # gentle ramp to avoid blowups
        sim.context.setParameter("k", _kt); sim.step(2000)
    sim.step(max(2000, n_prod_steps // 5))     # hydration/equilibration (loose restraint)
    # PHASE 2 — clamp pocket to bound-like shape and SAMPLE water.
    sim.context.setParameter("k", 1000.0)
    sim.minimizeEnergy(maxIterations=500)
    sim.step(2000)                              # re-equilibrate restrained

    prot_heavy = np.array(prot_heavy); prot_polar = np.array(prot_polar); water_o = np.array(water_o)
    acc = {"occ": [], "mean_hb": [], "frac_unhappy": [], "enclosure": []}
    nframes = 0
    for _ in range(max(1, n_prod_steps // report_every)):
        sim.step(report_every)
        st = sim.context.getState(getPositions=True)
        xyz = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        center = xyz[groove].mean(axis=0)              # pocket center = live groove-lining-atom centroid
        wo = xyz[water_o]
        dpk = np.linalg.norm(wo - center, axis=1)
        pk = water_o[dpk <= WATER_R]
        if len(pk) == 0:
            continue
        wpk = xyz[pk]
        # polar neighbors of each pocket water: protein N/O + other water O within 3.5 A
        polar_xyz = np.vstack([xyz[prot_polar], wo])
        tree = cKDTree(polar_xyz)
        nb = tree.query_ball_point(wpk, 3.5)
        nbc = np.array([len(x) - 1 for x in nb])        # minus self
        # enclosure: protein heavy atoms within 4.5 A
        ptree = cKDTree(xyz[prot_heavy])
        enc = np.array([len(x) for x in ptree.query_ball_point(wpk, 4.5)])
        acc["occ"].append(len(pk))
        acc["mean_hb"].append(float(nbc.mean()))
        acc["frac_unhappy"].append(float((nbc < 3).mean()))
        acc["enclosure"].append(float(enc.mean()))
        nframes += 1
    if nframes == 0:
        return None
    return {k: float(np.mean(v)) for k, v in acc.items()} | {"nframes": nframes,
            "n_water_o": int(len(water_o)), "n_prot_heavy": int(len(prot_heavy))}


def run_md(args):
    man = json.load(open(MANIFEST))
    recs = man["receptors"][: args.limit] if args.limit else man["receptors"]
    done = {json.loads(l)["rep_pdb"] for l in CACHE.read_text().splitlines()} if CACHE.exists() else set()
    n_steps = 5000 if args.smoke else args.prod_steps
    rep_every = 1000 if args.smoke else args.report_every
    print(f"=== E229 pocket-MD: {len(recs)} receptors, prod_steps={n_steps}, platform={args.platform} ===",
          flush=True)
    for i, rc in enumerate(recs, 1):
        rep = rc["peptides"][0]
        pdb = rep["pdb"]
        if pdb in done:
            continue
        t0 = time.time()
        try:
            site, apo = site_and_apo(pdb, rep["seq"], rep["pep_ch"])
            if apo is None:
                raise RuntimeError("no apo/site")
            desc = build_and_run(apo, site, n_steps, rep_every, args.platform)
            if desc is None:
                raise RuntimeError("no pocket waters sampled")
            row = {"rep_pdb": pdb, "n_pep": rc["n_pep"], "y_mean": rc["y_mean"], "y_std": rc["y_std"], **desc}
            with open(CACHE, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  [{i}/{len(recs)}] {pdb} n_pep={rc['n_pep']} occ={desc['occ']:.1f} "
                  f"unhappy={desc['frac_unhappy']:.2f} enc={desc['enclosure']:.1f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(recs)}] {pdb} FAILED: {e}", flush=True)
            if args.smoke:
                traceback.print_exc()
        finally:
            for p in WORK.glob(f"{pdb}_apo.pdb"):
                p.unlink(missing_ok=True)
    eval_only()


def eval_only():
    if not CACHE.exists():
        print("no cache yet"); return
    rows = [json.loads(l) for l in CACHE.read_text().splitlines()]
    if len(rows) < 5:
        print(f"only {len(rows)} receptors done — need >=5 for a correlation"); return
    y = np.array([r["y_mean"] for r in rows])
    feats = ["occ", "mean_hb", "frac_unhappy", "enclosure"]
    print(f"\n=== HYDRATION → RECEPTOR BASELINE (n={len(rows)} receptors) ===")
    print(f"  receptor-baseline std = {y.std():.2f}  (the variance the static wall caps at r~0.15)")
    for f in feats:
        x = np.array([r[f] for r in rows])
        if x.std() < 1e-9:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"  {f:<14} r={r:+.3f}")
    # multivariate leave-one-out Ridge
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r[f] for f in feats] for r in rows])
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=2.0).fit(sc.transform(X[tr]), y[tr])
        pred[i] = m.predict(sc.transform(X[i:i + 1]))[0]
    rmv = float(np.corrcoef(pred, y)[0, 1])
    print(f"\n  GIST-lite multivariate (LOO Ridge):  r={rmv:+.3f}")
    sig = 0.41 if len(rows) <= 25 else 0.30
    print(f"  significance bar at n={len(rows)}: r≈{sig:.2f}")
    print(f"  VERDICT: {'*** BREAKS THE WALL *** dynamic water beats static 0.15' if rmv > sig else 'does NOT clear the bar — wall is FEP-absolute-bound'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 receptor, tiny MD, for pipeline validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--prod-steps", type=int, default=250000)   # 0.5 ns @ 2 fs
    ap.add_argument("--report-every", type=int, default=2500)    # ~100 frames
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.limit = 1; a.platform = "CPU"
    if a.eval_only:
        eval_only()
    else:
        run_md(a)


if __name__ == "__main__":
    main()
