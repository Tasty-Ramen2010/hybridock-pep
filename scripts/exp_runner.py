#!/usr/bin/env python3
"""Experiment runner for the two deployed hypotheses.

Reuses nsweep_coverage's ca_rmsd so numbers are directly comparable to the
existing baseline (4.79A mean / 11-of-15 <=5A on these same 15 complexes at N=24).

  --partial-mode fixed   : one conformation_partial for all (default '1:1:1' = upstream)
  --partial-mode oracle  : per-complex partial chosen from the CSV's ss_class
                           SHEET->Extended, HELIX->Helical, UNUSUAL->mix

TSR score rescaling is controlled by env vars read inside RAPiDock's sampling.py
(RAPIDOCK_TSR_K / RAPIDOCK_TSR_K_TOR), so it needs no plumbing here.

Also reports pose SPREAD (mean pairwise RMSD of the returned poses), because the
literature's counterargument to score up-scaling is that it contracts samples
toward local minima -- which would shrink spread and could hurt best-of-N even
while sharpening individual poses. We measure that rather than assume it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsweep_coverage import ca_rmsd, _load_coords  # identical metric

RAPIDOCK_PY = os.environ.get("RAPIDOCK_PY", sys.executable)

SS_TO_PARTIAL = {
    "SHEET": "0:1:0",     # Extended
    "HELIX": "1:0:0",     # Helical
    "UNUSUAL": "1:1:1",   # no prior -> keep upstream mix
}


def pose_spread(poses: list[str]) -> float:
    """Mean pairwise CA RMSD across returned poses (diversity proxy)."""
    import math
    coords = []
    for p in poses[:12]:                      # cap for cost
        c = _load_coords(p)
        if c:
            coords.append(c)
    if len(coords) < 2:
        return float("nan")
    tot, n = 0.0, 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a, b = coords[i], coords[j]
            m = min(len(a), len(b))
            if m == 0:
                continue
            tot += math.sqrt(sum(math.dist(a[t], b[t]) ** 2 for t in range(m)) / m)
            n += 1
    return tot / n if n else float("nan")


def assert_torsion_nonlinearity(infer_dir: str) -> None:
    """Refuse to start a benchmark on a tree whose torsion heads use the wrong nonlinearity.

    THIS HAS NOW COST TWO BENCHMARKS. The authors' code has used Tanh in tor_bb_final_layer and
    tor_sc_final_layer since their first commit, and rapidock_local.pt's torsion weights were
    trained under it. Our fork carried SiLU (Sep-13 finding: long-peptide <=5A went 27/77 -> 9/77
    for UNCHANGED pretrained weights). It was fixed, and then on Sep-18 a `git reset --hard
    origin/master` silently put SiLU back, along with deleting the comment that explained why it
    must not be. Everything docked locally for the next three days went through it: the whole
    387-complex length-balanced bench, the blind run, 42k poses. Nothing errored, nothing looked
    wrong in a log, and the only symptom was that placement stayed perfect while SHAPE degraded
    (best-of-24 Kabsch 1.53 -> 2.35 A).

    A source-level check is the right guard because the failure is a source-level revert, and it
    costs microseconds against the hours of GPU a silent rerun costs.
    """
    src = Path(infer_dir) / "models" / "diffusion.py"
    lines = src.read_text().splitlines()
    for head in ("tor_bb_final_layer", "tor_sc_final_layer"):
        try:
            i = next(k for k, l in enumerate(lines)
                     if f"self.{head} = nn.Sequential" in l)
        except StopIteration:
            raise SystemExit(f"REFUSING TO RUN: {head} not found in {src}")
        # collect the Sequential's own lines, dropping comments so the prose in them
        # ("Tanh, NOT SiLU") can never satisfy or trip the check
        depth, body = 0, []
        for l in lines[i:]:
            code = l.split("#", 1)[0]
            body.append(code)
            depth += code.count("(") - code.count(")")
            if depth <= 0 and len(body) > 1:
                break
        block = "\n".join(body)
        if "nn.Tanh()" not in block or "nn.SiLU()" in block:
            raise SystemExit(
                f"REFUSING TO RUN: {head} in {src} does not use nn.Tanh().\n"
                f"  This is the Sep-13 SiLU/Tanh fork bug. Every benchmark through this tree is\n"
                f"  invalid. Restore with:\n"
                f"    git checkout backup-master-preclean -- {src}")


def run_one(name, receptor, seq, n, steps, batch, out_root, infer_dir, model_dir,
            ckpt, partial):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(["complex_name", "protein_description", "peptide_description"])
        w.writerow([name, receptor, seq])
        csv_path = tf.name
    cmd = [
        RAPIDOCK_PY, str(Path(infer_dir) / "inference.py"),
        "--protein_peptide_csv", csv_path,
        "--out", str(out_root), "--output_dir", str(out_root),
        "--model_dir", str(model_dir), "--ckpt", ckpt,
        "--N", str(n), "--batch_size", str(batch),
        "--inference_steps", str(steps), "--actual_steps", str(steps),
        "--no_final_step_noise",
        "--conformation_partial", partial,
        "--scoring_function", "none", "--confidence_ckpt", "null",
        "--cpu", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(infer_dir), capture_output=True, text=True)
    dt = time.time() - t0
    os.unlink(csv_path)

    poses = sorted(glob.glob(str(Path(out_root) / name / "rank*.pdb")))
    if not poses:
        print(f"[{name}] NO POSES (dt={dt:.0f}s) partial={partial}\n{proc.stderr[-400:]}", flush=True)
        return None
    crystal = receptor.replace("_protein_pocket.pdb", "_peptide.pdb")
    pairs = [r for r in (ca_rmsd(crystal, p) for p in poses) if r is not None]
    if not pairs:
        return None
    direct = [d for d, _ in pairs]
    best = min(direct)
    spread = pose_spread(poses)
    print(f"[{name}] partial={partial} n={len(direct)} dt={dt:.0f}s "
          f"best_direct={best:.2f}A spread={spread:.2f}A nearnative={sum(1 for d in direct if d<=2)}/{len(direct)}",
          flush=True)
    return best, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--infer-dir", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--ckpt", default="rapidock_local.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial-mode", choices=["fixed", "oracle"], default="fixed")
    ap.add_argument("--partial", default="1:1:1")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    # CLAUDE.md 7: paths crossing into RAPiDock MUST be absolute -- inference.py runs
    # with cwd=infer_dir, so a relative --out silently writes poses somewhere this
    # process never looks and every complex reports "NO POSES".
    a.out = str(Path(a.out).expanduser().resolve())
    a.infer_dir = str(Path(a.infer_dir).expanduser().resolve())
    if a.model_dir:
        a.model_dir = str(Path(a.model_dir).expanduser().resolve())
    model_dir = a.model_dir or str(Path(a.infer_dir) / "train_models" / "CGTensorProductEquivariantModel")
    assert_torsion_nonlinearity(a.infer_dir)
    rows = list(csv.DictReader(open(a.bench)))[: a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)

    print(f"[{a.label}] {len(rows)} complexes N={a.n} steps={a.steps} "
          f"partial-mode={a.partial_mode} TSR_K={os.environ.get('RAPIDOCK_TSR_K','1.0')} "
          f"TSR_K_TOR={os.environ.get('RAPIDOCK_TSR_K_TOR','1.0')}", flush=True)

    bests, spreads = [], []
    for i, r in enumerate(rows):
        ss = r.get("ss_class", "UNUSUAL")
        partial = SS_TO_PARTIAL.get(ss, "1:1:1") if a.partial_mode == "oracle" else a.partial
        print(f"[{i+1}/{len(rows)}] {r['name']} ({ss}, {r.get('pep_len','?')}-mer)", flush=True)
        res = run_one(r["name"], r["receptor"], r["seq"], a.n, a.steps, a.batch,
                      a.out, a.infer_dir, model_dir, a.ckpt, partial)
        if res:
            bests.append(res[0])
            if res[1] == res[1]:
                spreads.append(res[1])

    n = len(bests)
    if not n:
        print("NO RESULTS"); return
    sb = sorted(bests)
    print(f"\n=== RESULT [{a.label}] ===")
    print(f"n={n} mean={sum(bests)/n:.2f}A median={sb[n//2]:.2f}A "
          f"le2A={sum(1 for b in bests if b<=2)}/{n} le5A={sum(1 for b in bests if b<=5)}/{n} "
          f"mean_spread={sum(spreads)/len(spreads) if spreads else float('nan'):.2f}A")


if __name__ == "__main__":
    main()
