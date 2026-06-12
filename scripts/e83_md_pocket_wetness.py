"""E83 — direct MD pocket-wetness: does REAL water occupancy de-flip the salt-bridge REWARD?

E82: decomposing charge into penalty/reward, the desolvation PENALTY (buried unpaired charge in a dry
patch) is sign-stable, but the salt-bridge REWARD flips (+0.15 cr65 / -0.08 the98). The reward flips
because the favorable Coulomb-minus-desolvation NET depends on the local dielectric, and our STATIC
hydrophobic-contact proxy for "dry" is too crude to know the real dielectric. HYPOTHESIS: a DIRECT MD
measurement of water occupancy around each charged group is the true dielectric, so a reward conditioned
on MD wetness should stop flipping.

Per charged peptide group, from short explicit TIP3P+PME MD (E77 machinery):
  water_occ   = mean # water O within 3.5 Å of the charged atom, averaged over frames  (real local wetness)
  paired/buried/local_dry_static recorded too (to compare static vs MD wetness)
Reward_MD     = Σ [paired]·[buried]·(1 - water_occ_norm)   (low water = low dielectric = strong bridge)
Test: is Reward_MD sign-stable across charged-cr65 + charged-the98 where the static reward flipped?
Also: corr(static local_dry, MD water_occ) -> is the cheap static label even a good wetness proxy?

GPU, charged strong/weak extremes of BOTH datasets. Persists per-charge rows to data/e83_md_wetness.jsonl.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/e83_md_wetness.jsonl"
WORK = Path.home() / ".e83_work"
WORK.mkdir(exist_ok=True)
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
CHG_ATOMS = {"LYS": {"NZ"}, "ARG": {"NH1", "NH2", "NE"}, "HIS": {"ND1", "NE2"},
             "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
SOLV = {"HOH", "WAT", "NA", "CL", "K", "SOD", "CLA", "NA+", "CL-"}


def run_md(pep_pdb, rec_pdb, n_frames=15, steps=2500, equil=4000):
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from hybridock_pep.scoring.geometry_features import _merge_complex
    from hybridock_pep.scoring.mmgbsa import _pdbfixer_addH

    cx = _merge_complex(Path(pep_pdb), Path(rec_pdb))
    fixed = _pdbfixer_addH(Path(cx))
    pdb = app.PDBFile(str(fixed))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff, pH=7.0)
    mod.addSolvent(ff, model="tip3p", padding=1.0 * unit.nanometer, neutralize=True,
                   ionicStrength=0.15 * unit.molar)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    chain_res: dict = {}
    for res in mod.topology.residues():
        if res.name in SOLV:
            continue
        chain_res.setdefault(res.chain.id, set()).add(res.index)
    if len(chain_res) < 2:
        raise RuntimeError("need 2 chains")
    pep_chain = min(chain_res, key=lambda c: len(chain_res[c]))
    # charged peptide groups -> list of (sign, [atom idx]); receptor charged atoms; water O idx
    groups, rec_chg_idx, water_o, rec_heavy = [], [], [], []
    cur = None
    for atom in mod.topology.atoms():
        rn = atom.residue.name
        if rn in ("HOH", "WAT"):
            if atom.element is not None and atom.element.symbol == "O":
                water_o.append(atom.index)
            continue
        if rn in SOLV:
            continue
        is_pep = atom.residue.chain.id == pep_chain
        sgn = 1 if rn in POS3 else (-1 if rn in NEG3 else 0)
        if is_pep:
            if rn in CHG_ATOMS and atom.name in CHG_ATOMS[rn]:
                if cur is None or cur[0] != atom.residue.index:
                    cur = [atom.residue.index, sgn, []]
                    groups.append(cur)
                cur[2].append(atom.index)
        else:
            if atom.element is not None and atom.element.symbol != "H":
                rec_heavy.append(atom.index)
            if sgn and rn in CHG_ATOMS and atom.name in CHG_ATOMS[rn]:
                rec_chg_idx.append((atom.index, sgn))
    if not groups:
        return None
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        sim = app.Simulation(mod.topology, system, integ, openmm.Platform.getPlatformByName("CUDA"))
    except Exception:
        sim = app.Simulation(mod.topology, system, integ, openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=400)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(equil)
    wat = np.array(water_o, int)
    occ = np.zeros(len(groups))           # mean water O within 3.5 Å of group charged atoms
    paired = np.zeros(len(groups))        # mean frames with opposite rec charge within 4.5 Å
    for _ in range(n_frames):
        sim.step(steps)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        wtree = cKDTree(pos[wat])
        for gi, (_, sgn, aidx) in enumerate(groups):
            cx_ = pos[aidx]
            nw = sum(len(wtree.query_ball_point(c, 0.35)) for c in cx_)
            occ[gi] += nw
            pr = False
            for ai, asg in rec_chg_idx:
                if asg == -sgn:
                    if min(np.linalg.norm(c - pos[ai]) for c in cx_) < 0.45:
                        pr = True; break
            paired[gi] += 1.0 if pr else 0.0
    occ /= n_frames; paired /= n_frames
    return [dict(sign=int(g[1]), water_occ=float(occ[i]), paired_frac=float(paired[i]))
            for i, g in enumerate(groups)]


def stage(picks):
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    staged = []
    for cid, v in picks:
        if v["ds"] == "the98":
            k = cid[3:]
            sp, sr = Path("/tmp/ppep_work") / f"{k}_pep.pdb", Path("/tmp/ppep_work") / f"{k}_rec.pdb"
            dp, dr = WORK / f"{k}_pep.pdb", WORK / f"{k}_rec.pdb"
            if not dp.exists() and sp.exists():
                shutil.copy(sp, dp); shutil.copy(sr, dr)
            if dp.exists():
                staged.append((k, dp, dr, v["y"], "the98"))
        else:
            k = cid[3:]; br = bench.get(k)
            if br:
                staged.append((k, Path(br["peptide_pdb"]), Path(br["pocket_pdb"]), v["y"], "cr65"))
    return staged


def main():
    e82 = json.load(open("/tmp/e82_local_dry.json"))
    ch = [(k, v) for k, v in e82.items()]
    c = sorted([x for x in ch if x[1]["ds"] == "cr65"], key=lambda kv: kv[1]["y"])
    n = sorted([x for x in ch if x[1]["ds"] == "the98"], key=lambda kv: kv[1]["y"])
    picks = c + n   # FULL charged set (was extremes-only); already-done ids are skipped
    staged = stage(picks)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
    print(f"=== E83 MD pocket-wetness. {len(staged)} charged complexes (both datasets extremes) ===", flush=True)
    for k, pep, rec, y, ds in staged:
        if k in done:
            continue
        t0 = time.time()
        try:
            groups = run_md(pep, rec)
            if not groups:
                print(f"  {ds:5} {k:14} no charged group", flush=True); continue
            row = dict(id=k, ds=ds, y=y, groups=groups)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            wmean = np.mean([g["water_occ"] for g in groups])
            print(f"  {ds:5} {k:14} y={y:+6.1f} ngrp={len(groups)} <water_occ>={wmean:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {ds:5} {k:14} FAIL {str(e)[:60]}", flush=True)
    analyze()


def analyze():
    if not OUT.exists():
        return
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    if len(rows) < 8:
        print(f"\n(only {len(rows)} rows — need >=8)")
        return
    from scipy.stats import pearsonr
    # per complex: reward_MD = mean over paired buried-ish groups of (1 - occ_norm); penalty = unpaired dry
    # normalize water_occ across all groups
    allocc = np.array([g["water_occ"] for r in rows for g in r["groups"]])
    omax = allocc.max() + 1e-9
    for r in rows:
        rew = pen = 0.0
        for g in r["groups"]:
            dry = 1.0 - g["water_occ"] / omax        # MD dryness (low water = dry)
            if g["paired_frac"] >= 0.5:
                rew += dry                            # paired in dry env = reward
            else:
                pen += dry                            # unpaired in dry env = penalty
        L = len(r["groups"])
        r["reward_MD"] = rew / L; r["penalty_MD"] = pen / L
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]

    def pr(rs, f):
        x = np.array([r[f] for r in rs]); y = np.array([r["y"] for r in rs])
        return pearsonr(x, y)[0] if len(rs) > 4 and np.std(x) > 0 else np.nan
    print(f"\n=== E83 MD-wetness conditioning (cr65={len(c)} the98={len(n)}) ===")
    print("(static reward FLIPPED +0.15/-0.08. does MD wetness de-flip it?)")
    print(f"{'feature':<16}{'cr65':>9}{'the98':>9}  verdict")
    for f in ["reward_MD", "penalty_MD"]:
        rc, rn = pr(c, f), pr(n, f)
        st = "STABLE <==" if (rc == rc and rn == rn and rc * rn > 0) else "flip"
        print(f"  {f:<14}{rc:>+9.3f}{rn:>+9.3f}  {st}")
    print("  >> reward_MD sign-stable = explicit water recovers the favorable half (real charged lever).")
    print("     still flips = the reward needs full FEP free energy, not just water occupancy.")


if __name__ == "__main__":
    main()
