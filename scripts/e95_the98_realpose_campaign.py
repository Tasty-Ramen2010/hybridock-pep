"""E95 — the-98 real-pose campaign (parallel to cr65). Reconstructs receptors from RCSB, RAPiDock N=100.

the-98 source PDBs were in wiped /tmp. We reconstruct: id = PDB_recChain_pepChain (e.g. 3IQQ_A_B),
seq + experimental ΔG are in the pooled benchmark CSV. Download each PDB from RCSB, extract the receptor
chain, run RAPiDock N=100 with the peptide sequence. Crash-safe / resumable. Runs on the SAME GPU as the
cr65 campaign (RAPiDock uses ~4GB; GPU has headroom for two).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CAMP = ROOT / "runs" / "e95_the98_campaign"
CAMP.mkdir(parents=True, exist_ok=True)
PDBCACHE = ROOT / "runs" / "e95_pdbcache"
PDBCACHE.mkdir(parents=True, exist_ok=True)
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"
RUNNER = ROOT / "src/hybridock_pep/sampling/run_rapidock.py"
RDIR = ROOT / "third_party/RAPiDock"
MODELDIR = RDIR / "train_models/CGTensorProductEquivariantModel"
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def load_the98():
    rows = []
    for f in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / f)):
            if r["dataset"] == "the98":
                rows.append(dict(id=r["pdb"], seq=r["seq"], y=float(r["y"])))
    return rows


def fetch_pdb(pdb):
    f = PDBCACHE / f"{pdb}.pdb"
    if f.exists() and f.stat().st_size > 0:
        return f
    for url in (f"https://files.rcsb.org/download/{pdb}.pdb",):
        try:
            urllib.request.urlretrieve(url, f)
            if f.stat().st_size > 0:
                return f
        except Exception:  # noqa: BLE001
            pass
    return None


def extract_chain(pdb_file, chain, out):
    """Write only ATOM records of the given chain (protein receptor)."""
    lines = []
    for ln in Path(pdb_file).read_text().splitlines():
        if ln.startswith(("ATOM", "TER")) and (len(ln) < 22 or ln[21] == chain):
            if ln.startswith("ATOM") and ln[17:20].strip() not in AA3:
                continue
            lines.append(ln)
    if not any(l.startswith("ATOM") for l in lines):
        return None
    out.write_text("\n".join(lines) + "\nEND\n")
    return out


def generate(complexes):
    for i, r in enumerate(complexes):
        cid = r["id"]
        outdir = CAMP / cid
        poses = outdir / "poses"
        if poses.exists() and len(list(poses.glob("pose_*.pdb"))) >= 90:
            print(f"  [{i+1}/{len(complexes)}] {cid}: have poses, skip", flush=True)
            continue
        # harvest existing nested ranks first
        raw = outdir / "poses_raw"
        if raw.exists():
            ranks = sorted([p for p in raw.rglob("rank*.pdb")],
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
            if len(ranks) >= 90:
                poses.mkdir(parents=True, exist_ok=True)
                for j, rp in enumerate(ranks):
                    (poses / f"pose_{j}.pdb").write_text(rp.read_text())
                print(f"  [{i+1}/{len(complexes)}] {cid}: harvested {len(ranks)}", flush=True)
                continue
        parts = cid.split("_")
        pdb, rec_ch = parts[0], parts[1]
        pf = fetch_pdb(pdb)
        if pf is None:
            print(f"  [{i+1}/{len(complexes)}] {cid}: RCSB download FAIL", flush=True)
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        rec = extract_chain(pf, rec_ch, outdir / "receptor.pdb")
        if rec is None:
            print(f"  [{i+1}/{len(complexes)}] {cid}: chain {rec_ch} extract FAIL", flush=True)
            continue
        t0 = time.time()
        cmd = [RAPIDOCK_PY, str(RUNNER), "--peptide", r["seq"], "--receptor", str(rec.resolve()),
               "--output-dir", str((outdir / "poses_raw").resolve()), "--n-samples", "100",
               "--rapidock-dir", str(RDIR.resolve()), "--model-dir", str(MODELDIR.resolve()),
               "--ckpt", "rapidock_local.pt", "--scoring-function", "none", "--seed", "42"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ranks = sorted([p for p in (outdir / "poses_raw").rglob("rank*.pdb")],
                           key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
            poses.mkdir(exist_ok=True)
            for j, rp in enumerate(ranks):
                (poses / f"pose_{j}.pdb").write_text(rp.read_text())
            print(f"  [{i+1}/{len(complexes)}] {cid}: {len(ranks)} poses ({time.time()-t0:.0f}s)", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  [{i+1}/{len(complexes)}] {cid}: TIMEOUT", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(complexes)}] {cid}: FAIL {str(e)[:60]}", flush=True)


def main():
    c = load_the98()
    print(f"=== E95 the-98 real-pose campaign ({len(c)} complexes), parallel to cr65 ===", flush=True)
    generate(c)
    print("=== E95 generation done ===", flush=True)


if __name__ == "__main__":
    main()
