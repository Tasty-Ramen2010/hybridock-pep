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
    """Pairs that already have a score.

    A pair that *failed* is deliberately not counted as done: over a multi-day
    run the failures are the ones worth retrying, since they are usually
    environmental (a missing converter, a transient sampler crash) and get
    fixed while the job is still going. Re-running simply appends a newer row.
    """
    if not RESULTS.is_file():
        return set()
    done = set()
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("delta_g") is not None:
                done.add(pair_key(rec))
        except (ValueError, KeyError):
            continue
    return done


def parse_score(out_dir: Path) -> dict | None:
    """Read the pose ensemble's scores out of a finished run.

    Records more than the single best ΔG on purpose.

    *Best-of-N is an extreme-value statistic.* Taking min(ΔG) over N poses gives
    a pair with more successfully-converted poses more chances at a good score,
    independent of whether it binds. Pose counts do vary here (Meeko conversion
    yield differs run to run), so `mean_top3` and `median` are recorded as
    discriminants whose expectation does not drift with N, and `n_poses_scored`
    is kept so the bias can be tested for directly.

    *Convergence tells sampling-limited from scoring-limited.* RAPiDock samples
    blind and finds the right pocket roughly half the time on its own benchmark.
    If the AUC comes out low, that is ambiguous between "the scorer cannot
    discriminate" and "the sampler never found the site" — unless pose agreement
    is recorded. When many poses land in one cluster, sampling converged on a
    site; when they scatter, it did not. E403 stratifies on this.
    """
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

    dgs = [d for d in (_f(r, "delta_g", "dg", "affinity") for r in rows) if d is not None]
    if not dgs:
        return None
    ordered = sorted(dgs)  # most negative first

    # Cluster spread: how many distinct sites the poses landed on, and how much
    # of the ensemble the top cluster holds.
    clusters = [r.get("cluster_id") or r.get("cluster") for r in rows]
    clusters = [c for c in clusters if c not in (None, "")]
    n_clusters = len(set(clusters)) if clusters else None
    top_frac = None
    if clusters:
        from collections import Counter

        top_frac = round(Counter(clusters).most_common(1)[0][1] / len(clusters), 3)

    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    return {
        "delta_g": ordered[0],
        "mean_top3": round(sum(ordered[:3]) / min(3, len(ordered)), 4),
        "median_dg": round(median, 4),
        "dg_spread": round(ordered[-1] - ordered[0], 4),
        "vina": _f(rows[0], "vina_score", "vina"),
        "n_poses_scored": len(rows),
        "n_clusters": n_clusters,
        "top_cluster_frac": top_frac,
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
        # --blind is mandatory here, not a tuning choice. No pocket is known for
        # any pair, and for the negatives none exists. Without it Stage 1.7a
        # deletes every pose landing >35 Å from the domain centroid, which on
        # the elongated receptors in this set (armadillo/ankyrin repeats, coiled
        # coils) is where the real groove is — and it truncates more of the pose
        # set the larger the receptor, making domains non-comparable.
        "--blind",
        # Site/box still bound the Stage 2 grid: E401 sized them to cover each
        # cropped domain, which is cheaper than the blind default of 100 Å on
        # every receptor and reaches the same surface.
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
        "n_res": geo.get("n_res"),
        "n_samples": n_samples,
        "blind": True,
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
