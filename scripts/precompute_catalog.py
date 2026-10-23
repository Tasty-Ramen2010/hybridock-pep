#!/usr/bin/env python3
"""Run the catalog's peptide panel for real and write the preloaded results the home page shows.

For every protein in ``web/static/data/proteins.json`` and every peptide in its ``panel``, this runs the same
``hybridock-pep dock`` the page starts (25 poses, the protein's known site) and keeps the headline binding strength and
the five best poses (their peptide C-alpha coordinates, for the 3D view). Nothing is simulated or hand-edited: the file is
a record of runs. Finished runs are reused, so an interrupted job picks up where it stopped.

    conda activate score-env
    python3 scripts/precompute_catalog.py --timing src/hybridock_pep/web/static/data/timing.json

``--only mdm2`` runs one protein. ``--force`` redoes finished runs. Run it on the demo computer so the numbers and the
times come from the same machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "src" / "hybridock_pep" / "web" / "static"
POSES = 25
KEEP = 5  # poses kept per binder (their shapes ship inside catalog.json)
#: pairs the project's own notes describe as known binders; anything else is a candidate, and the page says so
KNOWN = {("mdm2", "ETFSDLWKLLPE"), ("mdm2", "SQETFSDLWKLLP"), ("bcl2", "LSECLKRIGDELDS")}


def first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        raw = (row.get(k) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def peptide_ca(pdb: Path, n: int) -> list[float]:
    """Cα coordinates of the last ``n`` residues of a pose file (the peptide is written after the protein), 1 decimal."""
    pts: list[float] = []
    for line in pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            pts += [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    return [round(v, 1) for v in pts[-3 * n:]]


def find_pose(out: Path, name: str) -> Path | None:
    for sub in ("poses_scored", "poses_minimized", "poses", "poses_raw", ""):
        cand = out / sub / name if sub else out / name
        if name and cand.is_file():
            return cand
    return None


def receptor_and_site(p: dict) -> tuple[Path, list[float], int]:
    """The receptor file, site and box the live page would use for this protein."""
    from hybridock_pep.web import server  # the server's own validated examples take precedence, as in the page

    for ex in server.available_examples():
        if ex["id"] == p.get("backendExample"):
            return Path(ex["receptor_path"]), list(ex["site"]), int(ex["box"])
    s = p["site"]
    return WEB / p["file"], [s["x"], s["y"], s["z"]], int(p.get("box", 30))


def note_time(history: Path | None, seq: str, poses: int, seconds: float) -> None:
    """Append one real run time to the history file the server reads (see hybridock_pep.web.server.record_run_time)."""
    if history is None:
        return
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"peptide": seq, "poses": poses, "seconds": round(seconds, 1)}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=WEB / "data" / "catalog.json")
    ap.add_argument("--timing", type=Path, help="timing.json from benchmark_timing.py, copied into the catalog")
    ap.add_argument("--machine", default="", help="name of this computer (default: taken from --timing)")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "catalog")
    ap.add_argument("--exe", default=shutil.which("hybridock-pep") or "hybridock-pep")
    ap.add_argument("--only", help="a protein key, e.g. mdm2")
    ap.add_argument("--poses", type=int, default=POSES, help=f"poses per run (default {POSES}, the largest size the demo offers)")
    ap.add_argument("--force", action="store_true", help="redo runs that already finished")
    ap.add_argument("--history", type=Path, help="timing history file to add each run's wall time to (the server quotes it back for the same peptide)")
    ap.add_argument("--time-sizes", default="", help="also run each peptide at these pose counts, only to record how long they take, e.g. 5,10")
    args = ap.parse_args()

    proteins = json.loads((WEB / "data" / "proteins.json").read_text(encoding="utf-8"))
    timing = json.loads(args.timing.read_text(encoding="utf-8")) if args.timing and args.timing.exists() else None
    prior = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    binders: dict[str, list] = dict(prior.get("binders", {}))
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    for p in proteins:
        if args.only and p["key"] != args.only:
            continue
        receptor, site, box = receptor_and_site(p)
        results = []
        for seq in p.get("panel", []):
            out = args.runs_dir / f"{p['key']}_{seq}"
            csv_path = out / "ranked_poses.csv"
            cmd = [args.exe, "dock", "--peptide", seq, "--receptor", str(receptor), "--site", *map(str, site),
                   "--box", str(box), "--n-samples", str(args.poses), "--output-dir", str(out)]
            if args.force or not csv_path.exists():
                shutil.rmtree(out, ignore_errors=True)
                print(f"[{p['key']}] {seq}: running ({args.poses} poses)...", flush=True)
                t0 = time.monotonic()
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0 or not csv_path.exists():
                    print(f"  FAILED ({proc.returncode}): {(proc.stdout + proc.stderr)[-400:]}", flush=True)
                    continue
                took = time.monotonic() - t0
                print(f"  done in {took:.0f} s", flush=True)
                note_time(args.history, seq, args.poses, took)
            for extra in [int(x) for x in args.time_sizes.split(",") if x.strip()]:
                tmp = args.runs_dir / f"_time_{p['key']}_{seq}_{extra}"
                shutil.rmtree(tmp, ignore_errors=True)
                t1 = time.monotonic()
                done = subprocess.run([*cmd[:cmd.index("--n-samples")], "--n-samples", str(extra), "--output-dir", str(tmp)], capture_output=True, text=True, check=False)
                if done.returncode == 0:
                    note_time(args.history, seq, extra, time.monotonic() - t1)
                    print(f"  timed {extra} poses: {time.monotonic() - t1:.0f} s", flush=True)
                shutil.rmtree(tmp, ignore_errors=True)
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            keep = []
            for i, row in enumerate(rows[:KEEP], 1):
                dg = first_float(row, ("pooled_affinity_dg", "delta_g", "mmgbsa_dg"))
                pose = find_pose(out, row.get("pose_filename", ""))
                if dg is None or pose is None:
                    continue
                ca = peptide_ca(pose, len(seq))
                if len(ca) == 3 * len(seq):
                    keep.append({"rank": int(row.get("rank") or i), "deltaG": round(dg, 2), "ca": ca})
            if not keep:
                print(f"  no usable poses for {seq}", flush=True)
                continue
            results.append({"seq": seq, "deltaG": keep[0]["deltaG"], "known": (p["key"], seq) in KNOWN, "poses": keep})
            print(f"  {p['key']} {seq}: {keep[0]['deltaG']} kcal/mol", flush=True)
        if results:
            binders[p["key"]] = results

    machine = args.machine or (timing or {}).get("machine", "") or prior.get("machine", "")
    catalog = {
        "version": 1, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "machine": machine, "poses": args.poses,
        "timing": ({k: timing[k] for k in ("grid", "refLength", "holdout", "measuredAt") if k in timing} if timing else prior.get("timing")),
        "binders": binders,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, separators=(",", ":")) + "\n", encoding="utf-8")
    n = sum(len(v) for v in binders.values())
    print(f"wrote {args.out}: {n} results for {len(binders)} proteins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
