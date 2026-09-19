#!/usr/bin/env python
"""Score co-folded poses the moment they land, instead of waiting for all 324.

The Boltz grid takes ~50 s per cell on the 5070 and the scoring is CPU-only Rosetta, so the
two can run at the same time on one machine for free. This watches logs/coventry_boltz.jsonl
and refines each transplanted pose as it appears, which means the cognate row and the twelve
measured cross-reactivities have scores within minutes of being co-folded rather than hours.

The pose is scored through EXACTLY the same path as every docked cell -- interface repack,
constrained minimisation, ref2015 interface energy (scripts/coventry_refine.py) -- against the
same binder model. That is the whole point: same receptor, same scoring function, one number
per cell, and the only thing that differs is where the peptide came from. A co-folded pose
that scores better than our docked one is a statement about pose generation, not about
scoring.

Runs until the Boltz grid writes COVENTRY_BOLTZ_DONE, or until --once is given.

Usage: coventry_boltz_score_watch.py [workers] [--once]
Output: logs/coventry_boltz_refine.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
SRC = ROOT / "logs/coventry_boltz.jsonl"
OUT = ROOT / "logs/coventry_boltz_refine.jsonl"
BINDERS = ROOT / "datasets/coventry/binders"
AF3_BINDERS = ROOT / "datasets/coventry/binders_af3"
GRID_LOG = ROOT / "logs/coventry_boltz_grid.log"
POLL_S = 20

_PR = None


def _init():
    global _PR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()


def _min_contact(pdb: str) -> float:
    """Closest receptor-peptide heavy-atom distance after refinement."""
    import numpy as np
    from scipy.spatial import cKDTree
    rec, pep = [], []
    for l in open(pdb):
        if l.startswith("ATOM") and l[76:78].strip() != "H":
            xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            (pep if l[21] == "B" else rec).append(xyz)
    if not rec or not pep:
        return float("nan")
    return float(cKDTree(np.asarray(rec)).query(np.asarray(pep))[0].min())


def score_cell(rec: dict) -> dict:
    """Refine and score one co-folded cell."""
    import tempfile
    from coventry_refine import refine_one

    name = rec["name"]
    out = {"name": name, "peptide": rec["peptide"], "binder": rec["binder"],
           "cognate": rec["cognate"], "iptm": rec.get("iptm"), "ptm": rec.get("ptm"),
           "fold_rmsd": rec.get("fold_rmsd"), "accepted": rec.get("accepted"),
           "axis": rec.get("axis"), "axis_quality": rec.get("axis_quality")}
    # peptide-only file: refine_one appends it to its OWN copy of the receptor
    pose = rec.get("peptide_only")
    if not pose and rec.get("transplanted"):
        # older grid rows predate peptide_only -- derive it from the stored complex
        pose = str(Path(rec["transplanted"]).with_name("peptide.pdb"))
        if not Path(pose).exists():
            src = Path(rec["transplanted"])
            keep = [l for l in src.read_text().splitlines()
                    if l.startswith("ATOM") and l[21] == "B"]
            Path(pose).write_text("\n".join(keep) + "\nTER\nEND\n") if keep else None
    if not pose or not Path(pose).exists():
        out["error"] = rec.get("transplant_error") or rec.get("error") or "no transplanted pose"
        return out
    # The peptide was transplanted into a specific receptor's frame, so it MUST be scored
    # against that same receptor. Scoring an AF3-frame peptide against the ESMFold model puts
    # it several angstrom off and returns interface energies in the thousands of REU.
    rdir = AF3_BINDERS if rec.get("receptor_model") == "af3" else BINDERS
    receptor = rdir / f"{rec['binder']}_1b1.pdb"
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000") as td:
        dest = str(Path(td) / "refined.pdb")
        try:
            r = refine_one(_PR, str(receptor), pose, dest)
            out["best_iface"] = r.get("ref2015_interface")
            out.update({k: v for k, v in r.items() if k != "ref2015_interface"})
            out["min_contact"] = _min_contact(dest)
        except Exception as exc:  # noqa: BLE001 -- one bad pose must not stop the watch
            out["error"] = f"{type(exc).__name__}: {exc}"
    return out


#: Binders where our ESMFold model is the proven outlier (Boltz and AF3 agree against it).
#: Their cells are re-transplanted onto the AF3 model and scored from that file instead --
#: the ESMFold transplant for these four is geometrically meaningless, not merely worse.
AF3_SRC = ROOT / "logs/coventry_boltz_af3.jsonl"
BROKEN = {"n3", "n7", "pc21", "pc26"}


def _retransplant_af3() -> None:
    """Refresh the AF3 re-transplants for the broken columns. Cheap, resumable, no GPU."""
    import subprocess
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/coventry_boltz_retransplant_af3.py")],
                       capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        print("  (af3 re-transplant timed out; will retry next poll)", flush=True)


def _sources() -> list[Path]:
    """Every shard's co-fold log. Parallel workers each write their own file."""
    return [f for f in sorted(ROOT.glob("logs/coventry_boltz*.jsonl"))
            if "refine" not in f.name and "retry" not in f.name and "af3" not in f.name]


def pending() -> list[dict]:
    """Cells co-folded but not yet scored, preferring the AF3 transplant where it exists."""
    if not _sources():
        return []
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    af3 = {}
    if AF3_SRC.exists():
        for line in AF3_SRC.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                af3[r["name"]] = r
    todo, seen = [], set()
    for line in (l for f in _sources() for l in f.read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["name"] in done or r["name"] in seen:
            continue
        seen.add(r["name"])
        if r["binder"] in BROKEN:
            if r["name"] not in af3:
                continue          # wait for the re-transplant rather than score a bad frame
            r = af3[r["name"]]
        todo.append(r)
    return todo


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    once = "--once" in sys.argv
    print(f"watching {SRC.name} with {workers} workers", flush=True)
    n_total = 0
    while True:
        _retransplant_af3()
        todo = pending()
        if todo:
            with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
                for res in ex.map(score_cell, todo, chunksize=1):
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    n_total += 1
                    tag = "COGNATE" if res["cognate"] else "       "
                    v = res.get("best_iface")
                    print(f"  {tag} {res['name']:16s} iface "
                          f"{v if v is None else round(v, 1)!s:>8}  "
                          f"iptm {res.get('iptm') or float('nan'):.3f}"
                          f"{'  ' + res['error'][:60] if 'error' in res else ''}", flush=True)
        if once:
            break
        if GRID_LOG.exists() and "COVENTRY_BOLTZ_DONE" in GRID_LOG.read_text()[-4000:] \
                and not pending():
            break
        time.sleep(POLL_S)
    print(f"scored {n_total} cells; COVENTRY_BOLTZ_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
