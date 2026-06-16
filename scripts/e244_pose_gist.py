"""E244 — POSE-AWARE GIST: the real WaterMap-for-a-ligand calculation. Per complex: run GIST on the apo
pocket (reuse e242 MD+GIST), then overlay the CRYSTAL peptide pose onto the per-voxel hydration free
energy and score the water THIS peptide displaces:

  disp_unhappy : sum of dG over displaced voxels with dG>0  (favorable: removing frustrated water)
  disp_happy   : sum of dG over displaced voxels with dG<0  (penalty: removing well-satisfied water)
  disp_total   : net displaced free energy  (the WaterMap binding contribution; ML learns the sign)
  disp_n_vox / disp_max : extent & sharpest displaced site

Unlike the apo-pocket descriptors (one value per receptor, can't tell peptides apart), this is PER-COMPLEX
and per-peptide. Target = the complex's own Kd. Writes data/e244_pose_gist.jsonl.

Run: python3 scripts/e244_pose_gist.py --manifest data/e244_diverse_manifest.json \
        --cache data/e244_pose_gist_a.jsonl --nshard 2 --shard 0 --equil-ps 100 --prod-ps 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e180_protdcal_925 as e180          # fetch
import e229_pocket_md_pilot as e229       # site_and_apo
import e242_gist_pipeline as e242         # solvate, write_min_eq_prod, run_gist, solvate_translation

WORK = ROOT / "runs" / "e244_gist"; WORK.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "e244_pose_gist.jsonl"
DISP_R = 2.8     # A: a voxel within this of any peptide heavy atom is "displaced" by the peptide
POCKET_R = 6.0


def peptide_heavy_xyz(pdb, pep_ch):
    """Crystal peptide heavy-atom coords (holo frame) from the cached RCSB structure."""
    from Bio.PDB import PDBParser
    f = e180.fetch(pdb)
    st = PDBParser(QUIET=True).get_structure(pdb, str(f))
    model = st[0]
    if pep_ch not in [c.id for c in model]:
        return None
    xyz = [a.coord for r in model[pep_ch] if r.id[0] == " " for a in r if a.element != "H"]
    return np.array(xyz, float) if xyz else None


def load_grid_dG(out: Path):
    """Per-voxel coords + BULK-REFERENCED free-energy density (kcal/mol/A^3) and voxel volume."""
    cols = out.read_text().splitlines()[1].split()
    data = np.genfromtxt(out, skip_header=2)
    idx = {c: i for i, c in enumerate(cols)}
    C = lambda k: data[:, idx[k]]
    xyz = data[:, 1:4]
    g_O = C("g_O")
    esw = C("Esw-dens(kcal/mol/A^3)"); eww_u = C("Eww-dens(kcal/mol/A^3)")
    dtt = C("dTStrans-dens(kcal/mol/A^3)"); dto = C("dTSorient-dens(kcal/mol/A^3)")
    # bulk Eww self-calibration uses distance from grid center, recomputed by caller; here use far-from-
    # densest as bulk proxy: voxels with normal g_O are bulk if their dG would otherwise blow up.
    bulk = (g_O > 0.8) & (g_O < 1.2)
    bulk_eww = float(np.median(eww_u[bulk] / g_O[bulk])) if bulk.sum() > 50 else -0.30
    deww = eww_u - bulk_eww * g_O
    vox = 0.5 ** 3
    dG_vox = (esw + deww - dtt - dto) * vox     # kcal/mol per voxel
    return xyz, dG_vox, vox


def displaced_descriptors(xyz, dG_vox, pep_xyz, gridcntr):
    """Water displaced by the peptide pose + the apo-pocket integral (for head-to-head)."""
    from scipy.spatial import cKDTree
    d_center = np.linalg.norm(xyz - np.asarray(gridcntr), axis=1)
    pk = d_center <= POCKET_R
    out = {
        "gist_dG_pocket": float(dG_vox[pk].sum()),
        "gist_unhappy_dG": float(dG_vox[pk][dG_vox[pk] > 0].sum()),
        "gist_n_pocket_vox": int(pk.sum()),
    }
    if pep_xyz is None or len(pep_xyz) == 0:
        out.update(disp_total=np.nan, disp_unhappy=np.nan, disp_happy=np.nan,
                   disp_n_vox=0, disp_max=np.nan)
        return out
    # voxels within DISP_R of any peptide heavy atom = displaced by this peptide
    tree = cKDTree(pep_xyz)
    near = tree.query_ball_point(xyz, DISP_R)
    disp = np.array([len(n) > 0 for n in near])
    dd = dG_vox[disp]
    out.update(
        disp_total=float(dd.sum()),
        disp_unhappy=float(dd[dd > 0].sum()),     # reward: frustrated water removed
        disp_happy=float(dd[dd < 0].sum()),       # penalty: good water removed
        disp_n_vox=int(disp.sum()),
        disp_max=float(dd.max()) if disp.any() else 0.0,
        disp_per_pepatom=float(dd.sum() / len(pep_xyz)),
    )
    return out


def run_one(c, equil_ps, prod_ps):
    pdb, seq, pep_ch = c["pdb"], c["seq"], c["pep_ch"]
    site, apo = e229.site_and_apo(pdb, seq, pep_ch)
    if apo is None:
        raise RuntimeError("no apo/site")
    rism_apo = ROOT / "runs" / "e230_rism" / pdb / "apo_amber.pdb"
    src = rism_apo if rism_apo.exists() else Path(apo)
    wd = WORK / pdb; wd.mkdir(exist_ok=True)
    apo2 = wd / "apo.pdb"; apo2.write_bytes(src.read_bytes())
    prm, crd = e242.solvate(apo2, wd)
    trans = e242.solvate_translation(apo2, prm, crd)
    gridcntr = np.asarray(site, float) + trans
    dcd = e242.write_min_eq_prod(prm, crd, wd, equil_ps, prod_ps)
    out = e242.run_gist(prm, crd, dcd, gridcntr, wd)
    xyz, dG_vox, _ = load_grid_dG(out)
    pep = peptide_heavy_xyz(pdb, pep_ch)
    pep_g = (pep + trans) if pep is not None else None      # peptide into grid frame
    desc = displaced_descriptors(xyz, dG_vox, pep_g, gridcntr)
    for f in (dcd, wd / "gist-gO.dx", wd / "gist-gH.dx"):
        Path(f).unlink(missing_ok=True)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "e244_diverse_manifest.json"))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--equil-ps", type=float, default=100.0)
    ap.add_argument("--prod-ps", type=float, default=1000.0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    comp = json.load(open(a.manifest))["complexes"]
    done = set()
    for c in ROOT.glob("data/e244_pose_gist*.jsonl"):
        for l in c.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["pdb"])
    comp = [c for c in comp if c["pdb"] not in done]
    comp = comp[a.shard::a.nshard]
    if a.limit:
        comp = comp[: a.limit]
    print(f"=== E244 pose-GIST shard {a.shard}/{a.nshard}: {len(comp)} complexes -> {cache.name} ===", flush=True)
    for c in comp:
        t0 = time.time()
        try:
            d = run_one(c, a.equil_ps, a.prod_ps)
            row = {"pdb": c["pdb"], "y": c["y"], "L": c["L"], "seq": c["seq"], **d}
            with open(cache, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  {c['pdb']} y={c['y']:+.1f} disp_total={d['disp_total']:+.1f} "
                  f"disp_unhappy={d['disp_unhappy']:+.1f} disp_n={d['disp_n_vox']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {c['pdb']} FAILED: {e}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
