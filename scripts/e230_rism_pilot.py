"""E230 — 3D-RISM pocket-hydration pilot (the OSI-clean WaterMap substitute). For each multi-binder
receptor: parameterize the apo receptor (pdb4amber → tleap/ff14SB), run 3D-RISM (rism3d.snglpnt, cSPCE/KH,
0.1 M-equivalent), dump the water-O distribution grid g_O(r), and integrate hydration descriptors over the
binding-pocket sphere:

  n_pocket   expected # waters in the pocket  (rho_bulk * integral of g_O)
  n_sites    # structured hydration sites (g_O > 3)  — WaterMap's "high-occupancy" waters
  max_g      peak g_O in pocket  (sharpest ordered water)
  mean_g     mean g_O in pocket
  exchem     global solute solvation free energy (KH excess chemical potential, kcal/mol)

3D-RISM fills cavities by the integral equation (no dry-pocket / bulk-washout problem that sank vanilla MD),
and is a single static-structure calculation — the "static image → hydration behaviour" model. Then test:
do these predict the receptor BASELINE past the static-sequence wall (0.15)?  Bar at n~23 is r≈0.41.

Usage:
  python3 scripts/e230_rism_pilot.py --smoke --pdb 4e34     # one receptor, validate the pipeline
  python3 scripts/e230_rism_pilot.py                        # full run over the manifest
  python3 scripts/e230_rism_pilot.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e229_pocket_md_pilot as e229  # noqa: E402  (reuse site_and_apo)

AMBER = Path("/home/igem/miniconda3/envs/ambertools")
RISM = AMBER / "bin" / "rism3d.snglpnt"
TLEAP = AMBER / "bin" / "tleap"
PDB4AMBER = AMBER / "bin" / "pdb4amber"
XVV = AMBER / "dat" / "rism1d" / "cSPCE" / "cSPCE_kh.xvv"
MANIFEST = ROOT / "data" / "e228_pilot_manifest.json"
CACHE = ROOT / "data" / "e230_rism.jsonl"
WORK = ROOT / "runs" / "e230_rism"; WORK.mkdir(parents=True, exist_ok=True)
RHO_BULK = 0.03333   # waters / A^3
WATER_R = 8.0
ENV = {**os.environ, "AMBERHOME": str(AMBER), "PATH": f"{AMBER/'bin'}:{os.environ.get('PATH','')}"}


def read_grid(path):
    """Parse a RISM MRC/CCP4 scalar grid → (g[nx,ny,nz], origin(3 A), voxel(3 A))."""
    hi = np.fromfile(path, dtype=np.int32, count=256)
    hf = np.fromfile(path, dtype=np.float32, count=256)
    nx, ny, nz = int(hi[0]), int(hi[1]), int(hi[2])
    nsymbt = int(hi[23])
    cella = hf[10:13]                       # cell dims (A)
    origin = hf[49:52].astype(float)        # MRC ORIGIN (words 50-52)
    voxel = cella / np.array([nx, ny, nz], float)
    raw = np.fromfile(path, dtype=np.float32, offset=1024 + nsymbt)[: nx * ny * nz]
    g = raw.reshape(nz, ny, nx).transpose(2, 1, 0)   # → [ix,iy,iz]
    return g, origin, voxel


def pocket_descriptors(dxfile, site):
    g, origin, d = read_grid(dxfile)
    nx, ny, nz = g.shape
    # voxel centers
    xs = origin[0] + d[0] * np.arange(nx)
    ys = origin[1] + d[1] * np.arange(ny)
    zs = origin[2] + d[2] * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = (X - site[0]) ** 2 + (Y - site[1]) ** 2 + (Z - site[2]) ** 2
    mask = r2 <= WATER_R ** 2
    gp = g[mask]
    if gp.size == 0:
        return None
    vox = float(d[0] * d[1] * d[2])
    n_pocket = float(RHO_BULK * gp.sum() * vox)
    return {"n_pocket": n_pocket, "n_sites": int((gp > 3.0).sum()),
            "max_g": float(gp.max()), "mean_g": float(gp.mean()), "n_vox": int(gp.size)}


def parse_exchem(stdout):
    val = np.nan
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("rism_excessChemicalPotential"):   # data line (header starts with "|")
            toks = [t for t in s.split()[1:] if _isnum(t)]
            if toks:
                val = float(toks[0])                       # Total excess chemical potential (kcal/mol)
    return val


def _isnum(t):
    try:
        float(t); return True
    except ValueError:
        return False


def run_one(pdb, seq, pep_ch, smoke=False):
    site, apo = e229.site_and_apo(pdb, seq, pep_ch)
    if apo is None:
        raise RuntimeError("no apo/site")
    wd = WORK / pdb; wd.mkdir(exist_ok=True)
    clean = wd / "apo_amber.pdb"
    subprocess.run([str(PDB4AMBER), "-i", str(apo), "-o", str(clean), "--nohyd", "--dry"],
                   env=ENV, capture_output=True, timeout=300, cwd=wd)
    if not clean.exists():
        raise RuntimeError("pdb4amber failed")
    leapin = wd / "leap.in"
    leapin.write_text(f"source leaprc.protein.ff14SB\nmol = loadpdb {clean.name}\n"
                      f"saveamberparm mol mol.prmtop mol.rst7\nquit\n")
    subprocess.run([str(TLEAP), "-f", "leap.in"], env=ENV, capture_output=True, timeout=600, cwd=wd)
    prm, rst = wd / "mol.prmtop", wd / "mol.rst7"
    if not (prm.exists() and rst.exists()):
        raise RuntimeError("tleap failed (nonstd residue / missing atoms)")
    buf, spc, tol = ("8", "1.0,1.0,1.0", "1e-3") if smoke else ("10", "0.75,0.75,0.75", "1e-4")
    p = subprocess.run([str(RISM), "--pdb", str(clean), "--prmtop", str(prm), "--rst", str(rst),
                        "--xvv", str(XVV), "--closure", "kh", "--buffer", buf, "--grdspc", spc,
                        "--guv", "guv", "--tolerance", tol],
                       env=ENV, capture_output=True, timeout=3000, cwd=wd, text=True)
    guv = next(iter(wd.glob("guv.O*.mrc")), None) or next(iter(wd.glob("guv.O*.dx")), None)
    if guv is None:
        raise RuntimeError(f"no guv grid (rism rc={p.returncode}): {p.stderr[-300:]}")
    desc = pocket_descriptors(guv, site)
    if desc is None:
        raise RuntimeError("no pocket voxels")
    desc["exchem"] = parse_exchem(p.stdout)
    # cleanup big grids
    for f in list(wd.glob("guv*.mrc")) + list(wd.glob("guv*.dx")):
        f.unlink(missing_ok=True)
    return desc


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--cache", default=str(CACHE))
    a = ap.parse_args()
    CACHE = Path(a.cache)
    if a.eval_only:
        return eval_only()

    man = json.load(open(a.manifest))
    recs = man["receptors"]
    if a.pdb:
        recs = [r for r in recs if r["peptides"][0]["pdb"] == a.pdb]
    if a.limit:
        recs = recs[: a.limit]
    done = {json.loads(l)["rep_pdb"] for l in CACHE.read_text().splitlines()} if CACHE.exists() else set()
    print(f"=== E230 3D-RISM pilot: {len(recs)} receptors (smoke={a.smoke}) ===", flush=True)
    for i, rc in enumerate(recs, 1):
        rep = rc["peptides"][0]; pdb = rep["pdb"]
        if pdb in done:
            continue
        t0 = time.time()
        try:
            d = run_one(pdb, rep["seq"], rep["pep_ch"], smoke=a.smoke)
            row = {"rep_pdb": pdb, "n_pep": rc["n_pep"], "y_mean": rc["y_mean"], "y_std": rc["y_std"], **d}
            with open(CACHE, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  [{i}/{len(recs)}] {pdb} n_pep={rc['n_pep']} n_pocket={d['n_pocket']:.1f} "
                  f"n_sites={d['n_sites']} max_g={d['max_g']:.1f} exchem={d['exchem']:.0f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(recs)}] {pdb} FAILED: {e}  ({time.time()-t0:.0f}s)", flush=True)
    eval_only()


def eval_only():
    if not CACHE.exists():
        print("no cache"); return
    rows = [json.loads(l) for l in CACHE.read_text().splitlines()]
    if len(rows) < 5:
        print(f"only {len(rows)} receptors — need >=5"); return
    y = np.array([r["y_mean"] for r in rows])
    feats = ["n_pocket", "n_sites", "max_g", "mean_g", "exchem"]
    print(f"\n=== 3D-RISM HYDRATION → RECEPTOR BASELINE (n={len(rows)}) ===")
    print(f"  baseline std={y.std():.2f}")
    for f in feats:
        x = np.array([r.get(f, np.nan) for r in rows], float)
        ok = ~np.isnan(x)
        if ok.sum() < 5 or np.nanstd(x) < 1e-9:
            continue
        print(f"  {f:<10} r={np.corrcoef(x[ok], y[ok])[0,1]:+.3f}")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r.get(f, np.nan) for f in feats] for r in rows], float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    pred = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        sc = StandardScaler().fit(X[tr])
        pred[i] = Ridge(alpha=2.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i+1]))[0]
    rmv = float(np.corrcoef(pred, y)[0, 1])
    bar = 0.41 if len(rows) <= 25 else 0.30
    print(f"\n  RISM multivariate (LOO Ridge): r={rmv:+.3f}   (bar at n={len(rows)}: r≈{bar:.2f})")
    print(f"  VERDICT: {'*** BREAKS THE WALL *** hydration beats static 0.15' if rmv > bar else 'does NOT clear bar — wall is FEP-absolute-bound'}")


if __name__ == "__main__":
    main()
