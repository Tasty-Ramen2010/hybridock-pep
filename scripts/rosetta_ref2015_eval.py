"""Rosetta ref2015 interface-energy comparison on the 65 crystal complexes (CPU).

FlexPepDock correlates a REWEIGHTED Rosetta interface energy with binding affinity
(lit: r=0.59 within-target). We compute the same physics via PyRosetta: ref2015
InterfaceAnalyzer interface dG (dG_separated) on the crystal pose, optionally after a
short FastRelax (closer to FlexPepDock's refinement). Then correlate with experimental ΔG
and run the same LOO harness as our geometry model — a fair head-to-head on OUR dataset.

CPU only. ~seconds-minutes/complex. Writes /tmp/rosetta_ref2015.json (crash-safe).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
RELAX = "--relax" in sys.argv


def init_rosetta():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false", silent=True)
    return pyrosetta


def score_complex(pyrosetta, pep_pdb, poc_pdb, relax=False):
    """Merge peptide(chain P)+pocket(chain R), score ref2015, return interface dG."""
    merged = Path("/tmp/_rosetta_tmp.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (poc_pdb, "R")):
        for l in Path(src).read_text().splitlines():
            if l.startswith(("ATOM", "HETATM")) and l[17:20] != "HOH":
                lines.append(l[:21] + ch + l[22:])
    merged.write_text("\n".join(lines) + "\nEND\n")
    pose = pyrosetta.pose_from_pdb(str(merged))
    sfxn = pyrosetta.get_fa_scorefxn()  # ref2015
    if relax:
        from pyrosetta.rosetta.protocols.relax import FastRelax
        fr = FastRelax(sfxn, 1)
        fr.apply(pose)
    total = sfxn(pose)
    # interface dG via InterfaceAnalyzerMover across P_R
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    iam = InterfaceAnalyzerMover("P_R")
    iam.set_scorefunction(sfxn)
    iam.apply(pose)
    data = iam.get_all_data()
    dG_sep = float(data.dG[1]) if hasattr(data, "dG") else float("nan")
    try:
        dG = float(iam.get_interface_dG())
    except Exception:
        dG = dG_sep
    return dict(ros_total=float(total), ros_ifdG=dG)


def main():
    pr = init_rosetta()
    e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
    b = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    out_path = Path("/tmp/rosetta_ref2015.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["pdb"] for r in out}
    for meta in b:
        pdb = meta["pdb"].upper()
        if pdb in done or pdb not in e0:
            continue
        r0 = e0[pdb]
        if not r0.get("pep_pdb") or not r0.get("poc_pdb"):
            continue
        try:
            s = score_complex(pr, r0["pep_pdb"], r0["poc_pdb"], relax=RELAX)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True); continue
        out.append(dict(pdb=pdb, y=meta["dg_exp"], **s))
        out_path.write_text(json.dumps(out))
        print(f"  {pdb}: ref2015 total={s['ros_total']:+.1f} ifdG={s['ros_ifdG']:+.2f} "
              f"(exp {meta['dg_exp']:.1f}) n={len(out)}", flush=True)

    # ---- correlate + LOO ----
    y = np.array([r["y"] for r in out])
    print(f"\n=== Rosetta ref2015 vs experimental ΔG (n={len(out)}) ===")
    for f in ["ros_total", "ros_ifdG"]:
        v = np.array([r[f] for r in out])
        if v.std() > 0:
            print(f"  raw corr({f}, ΔG) = {pearsonr(v, y).statistic:+.3f}")

    def loo1(x, y):
        p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            a, bb = np.polyfit(x[tr], y[tr], 1); p[i] = a * x[i] + bb
        return pearsonr(p, y).statistic, float(np.sqrt(np.mean((p - y) ** 2)))
    print("  --- fair LOO-fitted (like our model + Vina) ---")
    for f in ["ros_total", "ros_ifdG"]:
        v = np.array([r[f] for r in out])
        if v.std() > 0:
            r, rmse = loo1(v, y)
            print(f"  Rosetta {f:<10} LOO: r={r:+.3f} RMSE={rmse:.2f}")
    print(f"  guess-the-mean RMSE = {y.std():.2f}")
    print(f"  [reference] Vina-fit +0.527 | ours +0.576 | FlexPepDock lit r=0.59 (within-target)")


if __name__ == "__main__":
    main()
