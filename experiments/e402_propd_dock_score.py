"""E402 — dock and score every ProP-PD pair from E400 against its E401 domain.

The expensive half of the specificity benchmark. One `hybridock-pep dock` per
(domain, peptide) pair; the pair's score is the best pose's ΔG. E403 turns those
into an AUC.

Design notes
------------
*Resumable.* Results are appended to a JSONL as each pair finishes and completed
pairs are skipped on restart. At roughly 1.5 min/pair this is a multi-day job on
one GPU, so it has to survive being interrupted.

*Sequential.* One dock at a time. Stage 1 sampling already saturates the GPU and
Stage 2 already uses a process pool for ligand prep; running pairs concurrently
mostly buys swap pressure.

*Run directories are deleted after the score is read.* 1,185 runs of poses,
PDBQTs and grids is tens of GB of intermediates for six numbers each. Pass
`--keep-runs` to retain them.

*Site and box come from E401's index*, which derives them from the cropped
domain — no per-pair site guessing.

Usage
-----
    python experiments/e402_propd_dock_score.py                # everything
    python experiments/e402_propd_dock_score.py --limit 20     # pilot
    python experiments/e402_propd_dock_score.py --n-samples 40 # more sampling

Because peptides here are 16-mers with no known pose, `--n-samples` is the main
accuracy/time lever. The default of 20 matches what the rest of this repo uses
for a quick run; raise it if the AUC looks sampling-limited.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "propd_specificity.jsonl"
INDEX = ROOT / "data" / "propd_structures" / "index.json"
RESULTS = ROOT / "data" / "propd_dock_scores.jsonl"
RUNDIR = ROOT / "runs" / "propd"

#: Where the CLI lives. The benchmark runs unattended, so resolve it the same
#: way the rest of the package does rather than trusting $PATH.
CLI = Path(sys.prefix) / "bin" / "hybridock-pep"


def pair_key(rec: dict) -> str:
    return f"{rec['domain_id']}|{rec['peptide']}"


def load_done() -> set[str]:
    if not RESULTS.is_file():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            try:
                done.add(pair_key(json.loads(line)))
            except (ValueError, KeyError):
                continue
    return done


def parse_score(out_dir: Path) -> dict | None:
    """Read the best pose's scores out of a finished run."""
    csv_path = out_dir / "ranked_poses.csv"
    if not csv_path.is_file():
        return None
    import csv as _csv

    with csv_path.open() as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return None

    def _f(row, *names):
        for n in names:
            v = row.get(n)
            if v not in (None, "", "nan"):
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    # ranked_poses.csv is sorted best-first by the pipeline.
    best = rows[0]
    return {
        "delta_g": _f(best, "delta_g", "dg", "affinity"),
        "vina": _f(best, "vina_score", "vina"),
        "n_poses_scored": len(rows),
    }


def run_pair(rec: dict, geo: dict, n_samples: int, keep: bool, timeout: int) -> dict:
    out_dir = RUNDIR / f"{rec['domain_id']}__{rec['peptide']}"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    receptor = ROOT / geo["path"]
    site = geo["site"]
    cmd = [
        str(CLI), "dock",
        "--peptide", rec["peptide"],
        "--receptor", str(receptor),
        "--site", str(site[0]), str(site[1]), str(site[2]),
        "--box", str(geo["box"]),
        "--n-samples", str(n_samples),
        "--output-dir", str(out_dir),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        err = None if proc.returncode == 0 else (proc.stderr or proc.stdout or "")[-600:]
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout}s"
    elapsed = round(time.time() - t0, 1)

    scores = parse_score(out_dir) or {}
    if not keep:
        shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "domain_id": rec["domain_id"],
        "peptide": rec["peptide"],
        "label": rec["label"],
        "plddt": geo.get("plddt"),
        "box": geo.get("box"),
        "n_samples": n_samples,
        "seconds": elapsed,
        "error": err if not scores.get("delta_g") else None,
        **scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="stop after N new pairs")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800, help="per-pair seconds")
    ap.add_argument("--keep-runs", action="store_true")
    args = ap.parse_args()

    if not PAIRS.is_file():
        print(f"missing {PAIRS} — run e400 first", file=sys.stderr)
        return 2
    if not INDEX.is_file():
        print(f"missing {INDEX} — run e401 first", file=sys.stderr)
        return 2
    if not CLI.exists():
        print(f"hybridock-pep not found at {CLI}", file=sys.stderr)
        return 2

    index = json.loads(INDEX.read_text())
    records = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    records = [r for r in records if r["domain_id"] in index]
    done = load_done()
    todo = [r for r in records if pair_key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print("=== E402 — dock + score ProP-PD specificity pairs ===")
    print(f"  {len(records)} pairs total · {len(done)} done · {len(todo)} this run")
    print(f"  n_samples={args.n_samples}  timeout={args.timeout}s\n")

    RUNDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    started = time.time()
    for i, rec in enumerate(todo, 1):
        res = run_pair(rec, index[rec["domain_id"]], args.n_samples, args.keep_runs, args.timeout)
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(res) + "\n")
        if res.get("delta_g") is not None:
            ok += 1
            status = f"dG {res['delta_g']:+7.2f}"
        else:
            failed += 1
            status = f"FAIL  {(res.get('error') or '')[:60]}"
        rate = (time.time() - started) / i
        eta_h = rate * (len(todo) - i) / 3600
        print(
            f"  [{i}/{len(todo)}] {rec['domain_id'][:34]:34s} {rec['peptide']} "
            f"L{rec['label']}  {status}  {res['seconds']:6.1f}s  ETA {eta_h:5.1f}h",
            flush=True,
        )

    print(f"\n  {ok} scored, {failed} failed → {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
