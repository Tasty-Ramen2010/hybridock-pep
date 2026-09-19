#!/usr/bin/env python
"""Co-fold the whole Coventry 18x18 grid with Boltz-2 and score what comes out.

THREE QUESTIONS, ALL ON THE SAME RUN.

1. Can a co-folding model do the specificity grid at all?  Boltz reports ipTM per complex,
   which is its own estimate of how well the two chains fit together, so the 18x18 ipTM
   matrix is directly comparable to Fig. 2B without any physics from us.  If ipTM separates
   the 18 cognates from the 294 non-binders better than our docked-and-rescored grid does,
   that is the honest headline and we report it as such.

2. Does its pose beat ours?  Each co-folded complex is transplanted onto our binder model
   (sampling/cofold.py) so it lands in the same frame as every docked pose, then scored by
   the same repack-and-ref2015 path.  Same receptor, same scoring function, same 1 number
   per cell -- the only difference is where the peptide came from.

3. Does its DIRECTION rescue our pose pool?  This is the cheap one.  We thread the groove
   backwards in 79-81% of poses, and on 9CCE the co-folded direction identifies the forward
   ones with 98% precision.  Filtering our existing 24 poses per cell by agreement with the
   co-folded axis costs no GPU time and changes which pose the scorer is allowed to pick.

NO MSA, ON PURPOSE.  These binders are de novo designs; they have no natural homologues, so
an MSA search returns noise at best and costs minutes per cell.  `msa: empty` also keeps the
sequences off a third-party server, which matters for unpublished designs.

Usage: coventry_boltz_grid.py [--cells N] [--cognates-only] [--workers 1]
Output: logs/coventry_boltz.jsonl, one line per cell; poses under runs/coventry/boltz/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))

BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
OUT = ROOT / "logs/coventry_boltz.jsonl"
POSES = ROOT / "runs/coventry/boltz"
BINDERS = ROOT / "datasets/coventry/binders"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def load_grid() -> tuple[dict[str, str], dict[str, str]]:
    """Peptide and binder sequences, keyed by the paper's labels."""
    peps = {r["name"]: r["sequence"]
            for r in csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    # binders are named "<label>_1b1" in the structure set and "<label>B" in the paper's grid
    binders = {r["name"].removesuffix("_1b1"): r["sequence"]
               for r in csv.DictReader(open(ROOT / "data/coventry_binders.csv"))}
    return peps, binders


def cofold_cell(pep: str, binder: str, pep_seq: str, bind_seq: str,
                timeout_s: int = 1800) -> dict:
    """Run one Boltz-2 co-fold and return its confidences plus the converted PDB path.

    Args:
        pep: Peptide label, e.g. "n1".
        binder: Binder label, e.g. "pc28".
        pep_seq: Peptide sequence.
        bind_seq: Binder sequence.
        timeout_s: Hard limit on the Boltz subprocess.

    Returns:
        Dict with iptm/ptm/plddt and "pdb" pointing at the co-folded complex, or "error".
    """
    from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb

    name = f"{pep}__{binder}"
    dest = POSES / name
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "cofolded.pdb"
    work = Path(tempfile.mkdtemp(prefix=f"bz_{name}_", dir="/tmp/claude-1000"))
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
             "--recycling_steps", "3", "--diffusion_samples", "1",
             "--output_format", "mmcif", "--override"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        if proc.returncode != 0:
            return {"error": f"boltz exit {proc.returncode}: {proc.stderr[-400:]}"}
        cifs = sorted(work.rglob("*_model_0.cif"))
        if not cifs:
            return {"error": "no structure produced"}
        rec: dict = {"seconds": round(time.time() - t0, 1)}
        for j in cifs[0].parent.glob("confidence_*_model_0.json"):
            try:
                c = json.loads(j.read_text())
                rec.update({k: c[k] for k in
                            ("iptm", "ptm", "complex_plddt", "complex_iplddt",
                             "ligand_iptm", "protein_iptm") if k in c})
            except (OSError, ValueError):
                pass
            break
        _cif_to_pdb(cifs[0], final)
        rec["pdb"] = str(final)
        return rec
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def transplant_cell(cofolded: Path, binder: str, name: str) -> dict:
    """Move the co-folded peptide onto our binder model and record the gate diagnostics."""
    from hybridock_pep.sampling.cofold import (transplant, write_transplanted_peptide,
                                               write_transplanted_pose)
    from hybridock_pep.analysis.direction import axis_quality

    rec = BINDERS / f"{binder}_1b1.pdb"
    if not rec.exists():
        return {"transplant_error": f"no binder model at {rec}"}
    try:
        tr = transplant(cofolded, rec)
    except ValueError as exc:
        return {"transplant_error": f"{type(exc).__name__}: {exc}"}
    out = POSES / name / "transplanted.pdb"
    write_transplanted_pose(tr, rec, out)
    # peptide alone, for the refinement path -- it appends the peptide to its own copy of the
    # receptor, so handing it the complex would double the receptor
    pep_only = write_transplanted_peptide(tr, POSES / name / "peptide.pdb")
    return {
        "peptide_only": str(pep_only),
        "fold_rmsd": round(tr.fold_rmsd, 3),
        "n_aligned": tr.n_aligned,
        "accepted": tr.accepted,
        "axis": [round(float(v), 4) for v in tr.direction],
        "axis_quality": round(axis_quality(tr.peptide_ca), 3),
        "centroid": [round(float(v), 2) for v in tr.centroid],
        "transplanted": str(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=0, help="stop after N cells (0 = all 324)")
    ap.add_argument("--cognates-only", action="store_true",
                    help="only the 18 diagonal pairs, as a fast first pass")
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help=("run only shard I of N, so several workers can share the grid. "
                          "The split is applied AFTER the priority sort and takes every Nth "
                          "cell, which keeps each worker's queue in priority order rather "
                          "than giving one worker all the cognates and another all the blanks."))
    ap.add_argument("--out-suffix", default="", metavar="S",
                    help="append to the output JSONL name so parallel workers do not "
                         "interleave writes into one file")
    args = ap.parse_args()

    global OUT
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if args.out_suffix:
        OUT = ROOT / f"logs/coventry_boltz{args.out_suffix}.jsonl"

    peps, binders = load_grid()
    missing = [k for k in ORDER if k not in peps or k not in binders]
    if missing:
        sys.exit(f"missing sequences for {missing}")

    # A cell already done by ANY worker counts as done, so the shards never duplicate work
    # even if one is restarted or the split changes between runs.
    done = set()
    for f in sorted(ROOT.glob("logs/coventry_boltz*.jsonl")):
        if "refine" in f.name or "retry" in f.name:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])

    # Run the cells that carry information first. The 294 blank cells of Fig. 2B all say the
    # same thing ("no measurable binding"), so they can only ever confirm or deny a negative;
    # the 30 filled ones are where the paper actually measured a Kd, and they are what decides
    # whether co-folding reproduces the grid. Order: 18 cognates, then the 12 measured
    # cross-reactivities, then everything else -- so an interrupted run still has the answer.
    measured = {(r["peptide"], r["binder_label"][:-1])
                for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))
                if r["measured"] == "1" and r["cognate"] != "1"}

    def tier(p: str, b: str) -> int:
        return 0 if p == b else 1 if (p, b) in measured else 2

    jobs = [(p, b) for p in ORDER for b in ORDER if (not args.cognates_only or p == b)]
    jobs = [(p, b) for p, b in jobs if f"{p}__{b}" not in done]
    jobs.sort(key=lambda pb: (tier(*pb), ORDER.index(pb[0]), ORDER.index(pb[1])))
    n_by_tier = {t: sum(1 for pb in jobs if tier(*pb) == t) for t in (0, 1, 2)}
    print(f"priority order: {n_by_tier[0]} cognate, {n_by_tier[1]} measured cross-reactive, "
          f"{n_by_tier[2]} blank-cell", flush=True)
    if shard_n > 1:
        jobs = jobs[shard_i::shard_n]
    if args.cells:
        jobs = jobs[: args.cells]
    print(f"{len(jobs)} cells to co-fold in shard {shard_i}/{shard_n} "
          f"({len(done)} already done grid-wide) -> {OUT.name}", flush=True)

    POSES.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        for k, (p, b) in enumerate(jobs, 1):
            name = f"{p}__{b}"
            rec = {"name": name, "peptide": p, "binder": b,
                   "peptide_seq": peps[p], "cognate": int(p == b)}
            rec.update(cofold_cell(p, b, peps[p], binders[b]))
            if "pdb" in rec:
                rec.update(transplant_cell(Path(rec["pdb"]), b, name))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{k}/{len(jobs)}] {name:16s} "
                  f"iptm {rec.get('iptm', float('nan')):.3f}  "
                  f"fold {rec.get('fold_rmsd', float('nan')):.2f} A  "
                  f"{rec.get('seconds', 0):.0f}s"
                  f"{'  ERROR: ' + rec['error'] if 'error' in rec else ''}", flush=True)
    print("COVENTRY_BOLTZ_DONE")


if __name__ == "__main__":
    main()
