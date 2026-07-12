"""E174 — band-SPECIFIC physics features (Ram's architecture: global model stays, but band-appropriate NEW
features activate per band). The per-residue burial array (computed inside anchor_features but only max/sum
exposed) yields genuinely new binding-mode descriptors:

  VLONG (distributed binding):  n_buried, frac_buried, burial_entropy, top3_burial, burial_cv, salt_bridges
  SHORT (single-anchor binding): anchor_dominance(=burial_concentration), anchor_gap, anchor_isolation

These capture HOW the peptide binds (one deep anchor vs many distributed contacts) — physics the scalar
mean/max burial can't. Compute for crystal-925 from structures, report per-band correlation (do they beat the
base features?), then test the band-conditional model (base + band-gated NEW features) with multi-fold robustness.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hybridock_pep.scoring.anchor_features as af  # noqa: E402
e108 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e108", ROOT / "experiments/e108_ingest_pdbbind.py"))
importlib.util.spec_from_file_location("e108", ROOT / "experiments/e108_ingest_pdbbind.py").loader.exec_module(e108)
WORK = ROOT / "runs" / "e174_tmp"; WORK.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "e174_band_physics.jsonl"
_BUR = af._BURIAL_R


def band_physics(pep_pdb: Path, rec_pdb: Path):
    """Recompute the per-residue burial array and derive band-specific binding-mode features."""
    pep = af._parse_residues(pep_pdb)
    if not pep:
        return None
    rec_xyz, rec_charged = af._receptor(rec_pdb)
    burial, salt = [], 0
    for r in pep:
        rx = np.asarray(r["xyz"])
        nb = int((np.linalg.norm(rec_xyz - rx.mean(0), axis=1) < _BUR).sum()) if rx.size else 0
        burial.append(nb)
        cc = af._charge_center(r["rn"], r["at"])
        if cc is not None and rec_charged and any(
            cc[0] * sr < 0 and np.linalg.norm(cc[1] - xr) < af._SB_R for sr, xr in rec_charged
        ):
            salt += 1
    b = np.asarray(burial, float); L = len(b); tot = b.sum() + 1e-9
    p = b / tot
    srt = np.sort(b)[::-1]
    return {
        # VLONG — distributed binding
        "n_buried": float((b > 5).sum()),
        "frac_buried": float((b > 5).mean()),
        "burial_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
        "top3_burial": float(srt[:3].sum()),
        "burial_cv": float(b.std() / (b.mean() + 1e-9)),
        "salt_bridges": float(salt),
        # SHORT — single-anchor binding
        "anchor_dominance": float(b.max() / tot),
        "anchor_gap": float(b.max() - np.median(b)),
        "anchor_isolation": float((srt[0] - srt[1]) if L > 1 else srt[0]),
    }


NEWK = ["n_buried", "frac_buried", "burial_entropy", "top3_burial", "burial_cv", "salt_bridges",
        "anchor_dominance", "anchor_gap", "anchor_isolation"]


def build_cache():
    done = {json.loads(l)["pdb"] for l in CACHE.read_text().splitlines()} if CACHE.exists() else set()
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    n = 0
    for r in rows:
        if r["pdb"] in done:
            continue
        d = next((Path(p).parent for p in glob.glob(str(ROOT / f"data/drive_pull/pl/P-L/*/{r['pdb']}/{r['pdb']}_ligand.mol2"))), None)
        if d is None:
            continue
        try:
            pep = WORK / f"{r['pdb']}.pdb"; e108.mol2_to_pdb_seq(d / f"{r['pdb']}_ligand.mol2", pep)
            bp = band_physics(pep, (d / f"{r['pdb']}_protein.pdb").resolve())
        except Exception:  # noqa: BLE001
            bp = None
        if bp:
            with open(CACHE, "a") as fh:
                fh.write(json.dumps({"pdb": r["pdb"], **bp}) + "\n")
            n += 1
    print(f"  band-physics cache: +{n} (total {len(done)+n})", flush=True)


def main():
    if not CACHE.exists() or len(CACHE.read_text().splitlines()) < 800:
        print("building band-physics cache (CPU)...", flush=True)
        build_cache()
    bp = {json.loads(l)["pdb"]: json.loads(l) for l in CACHE.read_text().splitlines()}
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if json.loads(l)["pdb"] in bp]
    y = np.array([r["y"] for r in rows]); L = np.array([r["length"] for r in rows])
    short, vlong = L <= 8, L >= 17
    print(f"\n=== NEW band-physics feature correlations (short n={short.sum()}, vlong n={vlong.sum()}) ===")
    print(f"  {'feature':18s} {'short_r':>8s} {'vlong_r':>8s}   (base: max_burial short −0.34, mean_burial vlong −0.36)")
    for k in NEWK:
        v = np.array([float(bp[r["pdb"]][k]) for r in rows])
        rs = np.corrcoef(v[short], y[short])[0, 1] if np.std(v[short]) > 0 else np.nan
        rv = np.corrcoef(v[vlong], y[vlong])[0, 1] if np.std(v[vlong]) > 0 else np.nan
        print(f"  {k:18s} {rs:+8.3f} {rv:+8.3f}")


if __name__ == "__main__":
    main()
