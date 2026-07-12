"""E22 — Vina-score the real RAPiDock poses (top-5) for the ensemble, extract inter term.

For each crystal-65 complex: prepare receptor pdbqt (cached), prepare each of the top-5
RAPiDock pose ligands, Vina --score_only -> capture total AND intermolecular energy
(vina.score() returns [total, inter, intra, torsion, intra_best]). Mean over top-5.
Writes /tmp/e22_vina_real.json: {pdb: {vina_total, vina_inter}}. Crash-safe.

Reuses prepared receptors in runs/calibration_full/work/<PDB>/receptor.pdbqt when present.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GEN = ROOT / "logs/crystal65_n100"
CALIB = ROOT / "runs/calibration_full/work"
PY = "/home/igem/miniconda3/envs/score-env/bin/python"
OBABEL = "/home/igem/ADFRsuite_x86_64Linux_1.0/bin/obabel"


def prep_ligand(pose_pdb: Path, out_pdbqt: Path) -> bool:
    """Peptide PDB -> PDBQT via the project's production ligand prep (handles UNL/multi-ROOT)."""
    from hybridock_pep.prep.ligand import _prepare_single_ligand
    res = _prepare_single_ligand((0, pose_pdb, out_pdbqt.parent))
    if isinstance(res, Path) and res.exists():
        if res != out_pdbqt:
            out_pdbqt.write_text(res.read_text())
        return True
    return False


def prep_receptor(pdb: str, poc_pdb: Path, out_pdbqt: Path) -> Path | None:
    cached = CALIB / pdb / "receptor.pdbqt"
    if cached.exists():
        return cached
    r = subprocess.run([OBABEL, str(poc_pdb), "-O", str(out_pdbqt), "-xr", "--partialcharge",
                        "gasteiger"], capture_output=True, text=True)
    return out_pdbqt if out_pdbqt.exists() else None


def score_pose(v_receptor_pdbqt, lig_pdbqt, center):
    from vina import Vina
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(v_receptor_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    v.compute_vina_maps(center=[float(c) for c in center], box_size=[30, 30, 30])
    sc = v.score()
    # Vina 1.2: [0]=total [1]=inter [2]=intra [3]=torsion [4]=intra_best
    total = float(sc[0])
    inter = float(sc[1]) if len(sc) > 1 else float(sc[0])
    if total > 0:  # clash — local optimize a couple rounds
        try:
            v.optimize()
            sc = v.score(); total = float(sc[0]); inter = float(sc[1]) if len(sc) > 1 else total
        except Exception:
            pass
    return total, inter


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    out_path = Path("/tmp/e22_vina_real.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    tmp = Path("/tmp/e22_prep"); tmp.mkdir(exist_ok=True)
    for pdb, meta in bench.items():
        if pdb in out:
            continue
        posedir = GEN / f"cr_{pdb}" / "poses"
        if not posedir.exists():
            continue
        poc = (ROOT / meta["pocket_pdb"]).resolve()
        rec = prep_receptor(pdb, poc, tmp / f"{pdb}_rec.pdbqt")
        if rec is None:
            print(f"  {pdb} receptor prep FAIL", flush=True); continue
        totals, inters = [], []
        n_poses = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # rank1 by default (fast)
        for i in range(n_poses):
            pose = posedir / f"pose_{i}.pdb"
            if not pose.exists():
                continue
            lig = tmp / f"{pdb}_{i}.pdbqt"
            if not prep_ligand(pose, lig):
                continue
            xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                            for l in pose.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))])
            try:
                t, n = score_pose(rec, lig, xyz.mean(0))
                totals.append(t); inters.append(n)
            except Exception as e:  # noqa: BLE001
                print(f"  {pdb} pose{i} score FAIL {type(e).__name__}", flush=True)
        if not totals:
            print(f"  {pdb} no poses scored", flush=True); continue
        out[pdb] = dict(vina_total=float(np.mean(totals)), vina_inter=float(np.mean(inters)),
                        y=meta["dg_exp"], n_poses=len(totals))
        out_path.write_text(json.dumps(out))
        print(f"  {pdb}: vina_total={np.mean(totals):+.2f} vina_inter={np.mean(inters):+.2f} "
              f"(n={len(totals)}, {len(out)}/65)", flush=True)
    print(f"done: {len(out)}/65", flush=True)


if __name__ == "__main__":
    main()
