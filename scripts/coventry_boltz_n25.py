#!/usr/bin/env python
"""Escalate the cells we get WRONG to 25 diffusion samples — a pre-registered sampling test.

THE PREDICTION BEING TESTED. `coventry_disagreement.py` found that the rows we rank correctly are
exactly the rows where the co-folded pose beats our docked pose by a wide margin at the cognate
cell (signed disagreement vs cognate rank, r = -0.745), and the rows we get wrong are the rows
where the two poses AGREE. Agreement here is not reassurance: our own docked interaction term
correlates with the co-folded one at +0.021 across all 324 cells, so "agreement" means neither
pose found anything pair-specific. The seven failing rows are the rows where one diffusion sample
found nothing better than a backwards-threaded docked pose.

So there are exactly two explanations and they make opposite predictions:

  SAMPLING   one trajectory was unlucky. 25 samples should find a deeper cognate pose, and should
             help the cognate MORE than the false positives currently beating it, because a real
             binding mode exists to be found for the cognate and does not for the others.
  SCORING    no number of samples helps, because ref2015-after-cancellation cannot tell the
             cognate pose from the decoy pose once both are physically reasonable. Then cognate
             and false positives improve by the same amount and no row flips.

Both arms are measured: every cell in a failing row gets the same 25 samples, cognate and false
positive alike, so the comparison is within-row and the main effects cancel as usual. An earlier
n=5 retry was null, but it retried BROKEN cells (fold-gate rejects, positive interface energies)
chosen for being unscorable -- a different population from these, which score fine and rank wrong.

Two phases so the GPU is never idle waiting on Rosetta: co-fold everything first, then score.

Usage: coventry_boltz_n25.py [--samples 25] [--phase cofold|score|both] [--workers 8]
Output: logs/coventry_boltz_n25.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
CELLS = ROOT / "data/coventry_disagree_cells.json"
POSES = ROOT / "runs/coventry_boltz_n25"
BINDERS = ROOT / "datasets/coventry/binders"
AF3_BINDERS = ROOT / "datasets/coventry/binders_af3"
OUT = ROOT / "logs/coventry_boltz_n25.jsonl"
COFOLD_LOG = ROOT / "logs/coventry_boltz_n25_cofold.jsonl"
TRANS_LOG = ROOT / "logs/coventry_boltz_n25_transplant.jsonl"

#: Binders where our ESMFold model is the proven outlier -- Boltz and AF3 agree against it by
#: 0.9-2.4 A while ours is 5.7-16.9 A off. Their cells are transplanted onto the AF3 model.
BROKEN = {"n3", "n7", "pc21", "pc26"}

_PR = None


def binder_dir(binder: str) -> Path:
    return AF3_BINDERS if binder in BROKEN else BINDERS


# ----------------------------------------------------------------------------- phase 1: GPU

def cofold_many(pep: str, binder: str, pep_seq: str, bind_seq: str,
                samples: int, timeout_s: int = 3600) -> dict:
    """One Boltz-2 call producing `samples` structures; returns every converted PDB path."""
    from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb

    name = f"{pep}__{binder}"
    dest = POSES / name
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"bz25_{name}_", dir="/tmp/claude-1000"))
    try:
        spec = work / f"{name}.yaml"
        spec.write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {bind_seq}\n      msa: empty\n"
            f"  - protein:\n      id: B\n      sequence: {pep_seq}\n      msa: empty\n"
        )
        t0 = time.time()
        proc = subprocess.run(
            [str(BOLTZ), "predict", str(spec), "--out_dir", str(work),
             "--recycling_steps", "3", "--diffusion_samples", str(samples),
             "--output_format", "mmcif", "--override"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if proc.returncode != 0:
            return {"error": f"boltz exit {proc.returncode}: {proc.stderr[-400:]}"}
        cifs = sorted(work.rglob("*_model_*.cif"))
        if not cifs:
            return {"error": "no structure produced"}
        models = []
        for cif in cifs:
            k = cif.stem.rsplit("_model_", 1)[-1]
            pdb = dest / f"cofolded_{k}.pdb"
            try:
                _cif_to_pdb(cif, pdb)
            except Exception as exc:  # noqa: BLE001 -- one bad sample must not kill the cell
                continue
            m = {"sample": int(k), "pdb": str(pdb)}
            conf = cif.parent / f"confidence_{cif.stem}.json"
            if conf.exists():
                try:
                    c = json.loads(conf.read_text())
                    m.update({key: c[key] for key in ("iptm", "ptm", "complex_plddt")
                              if key in c})
                except (OSError, ValueError):
                    pass
            models.append(m)
        return {"seconds": round(time.time() - t0, 1), "models": models}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def phase_cofold(cells: list[dict], samples: int) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_boltz_grid import load_grid

    peps, binders = load_grid()
    done = set()
    if COFOLD_LOG.exists():
        for line in COFOLD_LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("models"):
                    done.add(r["name"])
    todo = [c for c in cells if f"{c['peptide']}__{c['binder']}" not in done]
    print(f"phase 1 (GPU): {len(todo)} cells x {samples} samples "
          f"({len(done)} already co-folded)", flush=True)
    t0 = time.time()
    with COFOLD_LOG.open("a") as fh:
        for k, c in enumerate(todo, 1):
            p, b = c["peptide"], c["binder"]
            r = {"name": f"{p}__{b}", "peptide": p, "binder": b, "role": c["role"],
                 "cognate": int(p == b)}
            r.update(cofold_many(p, b, peps[p], binders[b], samples))
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"  [{k}/{len(todo)}] {r['name']:14s} {c['role']:<15} "
                  f"{len(r.get('models', []))} models  {r.get('seconds', 0):.0f}s  "
                  f"eta {(len(todo) - k) * el / k / 60:.0f} min", flush=True)
    print("N25_COFOLD_DONE", flush=True)


# --------------------------------------------------------------------------- phase 2: CPU

def _init() -> None:
    global _PR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()


def transplant_one(args: tuple) -> dict:
    """Move one co-folded sample onto our binder model. Needs hybridock_pep, so: score-env."""
    from hybridock_pep.sampling.cofold import transplant, write_transplanted_peptide
    from hybridock_pep.analysis.direction import axis_quality

    name, peptide, binder, sample, pdb = args
    out = {"name": name, "peptide": peptide, "binder": binder, "sample": sample,
           "cognate": int(peptide == binder)}
    rec = binder_dir(binder) / f"{binder}_1b1.pdb"
    pep_only = POSES / name / f"peptide_{sample:02d}.pdb"
    try:
        tr = transplant(Path(pdb), rec)
        write_transplanted_peptide(tr, pep_only)
    except (ValueError, OSError) as exc:
        out["error"] = f"transplant: {type(exc).__name__}: {exc}"
        return out
    out["fold_rmsd"] = round(float(tr.fold_rmsd), 3)
    out["accepted"] = bool(tr.accepted)
    out["peptide_pdb"] = str(pep_only)
    try:
        # axis_quality takes CA COORDINATES, not a path -- passing the Path silently reaches
        # float() on a PosixPath and takes down the whole pool
        ca = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                       for l in pep_only.read_text().splitlines()
                       if l.startswith("ATOM") and l[12:16].strip() == "CA"])
        if len(ca) >= 2:
            out["axis_quality"] = round(float(axis_quality(ca)), 3)
    except (ValueError, OSError):
        pass
    return out


def phase_transplant(workers: int) -> None:
    """PyRosetta is compiled for 3.10 and hybridock_pep needs 3.11, so the two halves of the
    scoring pass cannot share one interpreter. Transplanting is the half that needs 3.11."""
    done = set()
    if TRANS_LOG.exists():
        for line in TRANS_LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["name"], r["sample"]))
    jobs = []
    for line in COFOLD_LOG.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for m in r.get("models", []):
            if (r["name"], m["sample"]) in done or not Path(m["pdb"]).exists():
                continue
            jobs.append((r["name"], r["peptide"], r["binder"], m["sample"], m["pdb"]))
    print(f"phase 2a (transplant): {len(jobs)} poses, {workers} workers", flush=True)
    if not jobs:
        return
    with TRANS_LOG.open("a") as fh, ProcessPoolExecutor(workers) as ex:
        for k, row in enumerate(ex.map(transplant_one, jobs, chunksize=4), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 100 == 0:
                print(f"  [{k}/{len(jobs)}]", flush=True)
    print("N25_TRANSPLANT_DONE", flush=True)


def score_one(args: tuple) -> dict:
    """Repack and score one transplanted peptide. Needs PyRosetta, so: the rapidock env."""
    from coventry_refine import refine_one

    row = dict(args[0])
    rec = binder_dir(row["binder"]) / f"{row['binder']}_1b1.pdb"
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000") as td:
        try:
            r = refine_one(_PR, str(rec), row["peptide_pdb"], str(Path(td) / "refined.pdb"))
            row["best_iface"] = r.get("ref2015_interface")
        except Exception as exc:  # noqa: BLE001 -- one bad sample must not stop the sweep
            row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def phase_score(workers: int) -> None:
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["name"], r["sample"]))
    jobs = []
    for line in TRANS_LOG.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r["name"], r["sample"]) in done or not r.get("peptide_pdb"):
            continue
        if Path(r["peptide_pdb"]).exists():
            jobs.append((r,))
    print(f"phase 2b (score): {len(jobs)} poses, {workers} workers", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, row in enumerate(ex.map(score_one, jobs, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 50 == 0 or k == 1:
                el = time.time() - t0
                v = row.get("best_iface")
                print(f"  [{k}/{len(jobs)}] {row['name']:14s} s{row['sample']:<3} "
                      f"iface {v if v is not None else float('nan'):8.1f}  "
                      f"eta {(len(jobs) - k) * el / k / 60:.0f} min", flush=True)
    print("N25_SCORE_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--phase", choices=("cofold", "transplant", "score", "both"),
                default="both")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cells = json.loads(CELLS.read_text())
    POSES.mkdir(parents=True, exist_ok=True)
    if a.phase in ("cofold", "both"):
        phase_cofold(cells, a.samples)
    if a.phase in ("transplant", "both"):
        phase_transplant(a.workers)
    if a.phase in ("score", "both"):
        phase_score(a.workers)


if __name__ == "__main__":
    main()
