"""E77 — explicit-water BRIDGE satisfaction: does a buried charge satisfied by a WATER de-flip E75?

Ram's hydration-gradient idea, decoded, is GIST / 3D-RISM: make the solvent explicit and spatial so the
desolvation environment (the thing GBn2 is blind to) becomes visible. The cheap layer of that idea (water
COUNT per area) is just GB-desolvation re-skinned and already washed (E72). The decisive, genuinely-new
layer his framing exposes:

  E75 penalized a buried peptide charge as "unsatisfied" if it had no DIRECT receptor partner within 4.5 Å
  -- and that penalty FLIPPED sign across datasets. But a buried charge can be satisfied by a BRIDGING
  WATER, which a single static pose with no explicit solvent cannot see. HYPOTHESIS: E75 flipped because it
  mislabeled water-bridged charges as penalties.

Test: explicit TIP3P+PME MD (the E76 machinery, validated), then per interface peptide charged group count:
  direct_sat   = receptor charged/polar heavy atom within 4.5 Å (no water needed)
  water_bridge = a PERSISTENT water O (occupancy >=0.5 across frames) within 3.4 Å of BOTH the peptide
                 charged atom AND a receptor heavy atom  -> a real bridging water
  unsatisfied  = neither
Features: water_sat_frac, total_sat_frac (direct OR water), n_unsat_water (water-corrected unsatisfied).
Decisive readout: sign-stability of the water-corrected satisfaction across cr65 + the98 charged subsets.
If the static (direct-only) unsat flips but the water-corrected one holds sign -> water bridges were the
missing physics, and Ram's explicit-hydration intuition is right where the cheap average-density version
is not.

GPU, charged extremes of BOTH datasets only. Persists per-complex rows to data/e77_water_bridge.jsonl,
inputs cached under ~/.e77_work (survives /tmp wipes).
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data/e77_water_bridge.jsonl"
WORK = Path.home() / ".e77_work"        # persistent (survives /tmp wipe)
WORK.mkdir(exist_ok=True)

SOLVENT = {"HOH", "WAT", "NA", "CL", "K", "SOD", "CLA", "NA+", "CL-"}
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
CHG_ATOMS = {"LYS": {"NZ"}, "ARG": {"NH1", "NH2", "NE"}, "HIS": {"ND1", "NE2"},
             "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
REC_POLAR = {"N", "O", "OD1", "OD2", "OE1", "OE2", "NZ", "NH1", "NH2", "NE",
             "ND1", "NE2", "ND2", "OG", "OG1", "OH", "SG"}


def run_complex(pep_pdb, rec_pdb, n_frames=20, steps=2500, equil=5000):
    """Solvate complex (TIP3P+PME), run MD, return water-bridge satisfaction of peptide charged groups."""
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

    # peptide = smaller protein chain by residue count
    chain_res: dict = {}
    for res in mod.topology.residues():
        if res.name in SOLVENT:
            continue
        chain_res.setdefault(res.chain.id, set()).add(res.index)
    if len(chain_res) < 2:
        raise RuntimeError(f"need 2 protein chains, got {list(chain_res)}")
    pep_chain = min(chain_res, key=lambda c: len(chain_res[c]))

    # index sets
    water_o, rec_heavy, pep_chg_groups = [], [], []  # pep_chg_groups: list of (resid, sign, [atom idx])
    cur = None
    for atom in mod.topology.atoms():
        rn = atom.residue.name
        if rn in ("HOH", "WAT") and atom.element is not None and atom.element.symbol == "O":
            water_o.append(atom.index)
            continue
        if rn in SOLVENT:
            continue
        is_pep = atom.residue.chain.id == pep_chain
        if not is_pep:
            if atom.element is not None and atom.element.symbol != "H":
                rec_heavy.append(atom.index)
            continue
        # peptide atom
        if rn in CHG_ATOMS and atom.name in CHG_ATOMS[rn]:
            key = (atom.residue.index, 1 if rn in POS3 else -1)
            if cur is None or cur[0] != atom.residue.index:
                cur = [atom.residue.index, key[1], []]
                pep_chg_groups.append(cur)
            cur[2].append(atom.index)
    if not pep_chg_groups:
        return dict(skip="no charged peptide group")
    rec_heavy = np.array(rec_heavy, int)

    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                            0.002 * unit.picoseconds)
    try:
        sim = app.Simulation(mod.topology, system, integ,
                             openmm.Platform.getPlatformByName("CUDA"))
    except Exception:
        sim = app.Simulation(mod.topology, system, integ, openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)
    sim.step(equil)

    # per charged group: count frames with a bridging water and whether interface
    n_grp = len(pep_chg_groups)
    bridge_frames = np.zeros(n_grp)
    direct_frames = np.zeros(n_grp)
    iface_frames = np.zeros(n_grp)
    wat_o = np.array(water_o, int)
    for _ in range(n_frames):
        sim.step(steps)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        rec_xyz = pos[rec_heavy]
        rtree = cKDTree(rec_xyz)
        wxyz = pos[wat_o]
        wtree = cKDTree(wxyz)
        for gi, (_, _sign, aidx) in enumerate(pep_chg_groups):
            cxyz = pos[aidx]
            # interface? any rec heavy within 6 Å (0.6 nm)
            dmin = rtree.query(cxyz, k=1)[0].min()
            if dmin > 0.6:
                continue
            iface_frames[gi] += 1
            # direct partner within 4.5 Å (0.45 nm)
            if dmin <= 0.45:
                direct_frames[gi] += 1
                continue
            # water bridge: a water O within 3.4 Å of a charged atom AND within 3.4 Å of a rec heavy atom
            near_w = set()
            for c in cxyz:
                near_w.update(wtree.query_ball_point(c, 0.34))
            bridged = False
            for wi in near_w:
                if rtree.query(wxyz[wi], k=1)[0] <= 0.34:
                    bridged = True
                    break
            if bridged:
                bridge_frames[gi] += 1
    occ = lambda a: a / max(1, n_frames)
    # a group counts as interface if present >50% of frames
    iface = occ(iface_frames) >= 0.5
    n_iface = int(iface.sum())
    if n_iface == 0:
        return dict(skip="no interface charged group")
    direct_sat = (occ(direct_frames) >= 0.5) & iface
    water_sat = (occ(bridge_frames) >= 0.5) & iface & ~direct_sat
    n_direct = int(direct_sat.sum())
    n_water = int(water_sat.sum())
    n_unsat_static = n_iface - n_direct                       # E75 view (direct only)
    n_unsat_water = n_iface - n_direct - n_water              # water-corrected
    return dict(
        n_iface_chg=n_iface, n_direct=n_direct, n_water_bridge=n_water,
        n_unsat_static=n_unsat_static, n_unsat_water=n_unsat_water,
        water_sat_frac=n_water / n_iface,
        total_sat_frac=(n_direct + n_water) / n_iface,
        unsat_static_frac=n_unsat_static / n_iface,
        unsat_water_frac=n_unsat_water / n_iface,
    )


def stage_inputs(picks):
    """Copy/locate pep+rec PDBs into persistent WORK; return list of (id, pep, rec, y, ds)."""
    staged = []
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    for cid, v in picks:
        ds = v["ds"]
        if ds == "the98":
            k = cid[3:]
            src_p, src_r = Path("/tmp/ppep_work") / f"{k}_pep.pdb", Path("/tmp/ppep_work") / f"{k}_rec.pdb"
            dp, dr = WORK / f"{k}_pep.pdb", WORK / f"{k}_rec.pdb"
            if not dp.exists() and src_p.exists():
                shutil.copy(src_p, dp); shutil.copy(src_r, dr)
            if dp.exists():
                staged.append((k, dp, dr, v["y"], ds))
        else:  # cr65
            k = cid[3:]
            br = bench.get(k)
            if br:
                staged.append((k, Path(br["peptide_pdb"]), Path(br["pocket_pdb"]), v["y"], ds))
    return staged


def main():
    e74 = json.load(open("/tmp/e74_charged.json"))
    ch = [(k, v) for k, v in e74.items() if abs(v["net_charge"]) >= 2]
    c65 = sorted([x for x in ch if x[1]["ds"] == "cr65"], key=lambda kv: kv[1]["y"])
    c98 = sorted([x for x in ch if x[1]["ds"] == "the98"], key=lambda kv: kv[1]["y"])
    # strong/weak extremes of each dataset (max contrast, sign-stability test)
    picks = c65[:6] + c65[-6:] + c98[:6] + c98[-6:]
    staged = stage_inputs(picks)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
    print(f"=== E77 explicit-water bridge satisfaction. {len(staged)} charged complexes "
          f"(cr65+the98 extremes) ===", flush=True)
    for k, pep, rec, y, ds in staged:
        if k in done:
            continue
        t0 = time.time()
        try:
            r = run_complex(pep, rec)
            if "skip" in r:
                print(f"  {ds:5} {k:14} SKIP {r['skip']}", flush=True)
                continue
            row = dict(id=k, ds=ds, y=y, **r)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {ds:5} {k:14} y={y:+6.1f} iface={r['n_iface_chg']} direct={r['n_direct']} "
                  f"water={r['n_water_bridge']} unsat:{r['n_unsat_static']}->{r['n_unsat_water']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {ds:5} {k:14} FAIL {str(e)[:70]}", flush=True)
    analyze()


def analyze():
    if not OUT.exists():
        return
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    if len(rows) < 8:
        print(f"\n(only {len(rows)} rows — need >=8 for the sign-stability test)")
        return
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]

    def sp(rs, f):
        x = np.array([r[f] for r in rs], float); y = np.array([r["y"] for r in rs], float)
        m = ~(np.isnan(x) | np.isnan(y))
        return spearmanr(x[m], y[m]).statistic if m.sum() > 4 else np.nan

    print(f"\n=== E77 sign-stability on charged (cr65={len(c)}, the98={len(n)}) ===")
    print("(ΔG: lower=stronger. unsat should correlate POSITIVE with ΔG = more unsat -> weaker.)")
    print(f"{'feature':<22}{'all':>9}{'cr65':>9}{'the98':>9}  stable?")
    for f in ["unsat_static_frac", "unsat_water_frac", "water_sat_frac", "total_sat_frac"]:
        a, cc, nn = sp(rows, f), sp(c, f), sp(n, f)
        st = "YES" if (not np.isnan(cc) and not np.isnan(nn) and cc * nn > 0) else "FLIP/na"
        mark = "  <== sign-stable" if st == "YES" and min(abs(cc), abs(nn)) > 0.2 else ""
        print(f"  {f:<22}{a:>+9.3f}{cc:>+9.3f}{nn:>+9.3f}  {st}{mark}")
    tw = sum(r["n_water_bridge"] for r in rows); td = sum(r["n_direct"] for r in rows)
    tu = sum(r["n_unsat_water"] for r in rows)
    print(f"\n  totals: direct-satisfied={td}  WATER-bridged={tw}  still-unsatisfied={tu}")
    print("  >> if unsat_static_frac FLIPS but unsat_water_frac holds sign (both +, |r|>0.2),")
    print("     water bridges were the missing physics E75 couldn't see -> Ram's hydration idea wins")
    print("     on the QUALITY layer (which water), not the cheap COUNT layer.")


if __name__ == "__main__":
    main()
