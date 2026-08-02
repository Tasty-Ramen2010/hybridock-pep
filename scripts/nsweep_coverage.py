#!/usr/bin/env python3
"""nsweep_coverage.py — Does RAPiDock generation coverage scale with N?

The proven Stage-1 bottleneck is near-native GENERATION rate (~3.7% at N=100),
not ranking. This quantifies whether drawing MORE stochastic samples raises the
oracle near-native rate — the cheapest possible 'considerable improvement' lever.

Method (runs entirely in the rapidock conda env):
  For each benchmark complex, dock the peptide from SEQUENCE (blind: built
  conformations + randomized position, NO crystal coords) generating N_MAX poses
  in one shot. Poses are i.i.d. stochastic draws, so the first-N prefix is a valid
  N-sample. Compute receptor-frame Ca RMSD to the crystal peptide (docking RMSD,
  no superposition). Report, per N in {100,200,300,500}:
     - oracle best-of-N Ca RMSD (mean over complexes, per SS class)
     - Hit@2A = fraction of complexes with >=1 pose <= 2.0 A
     - near-native rate = fraction of ALL poses <= 2.0 A

Usage (rapidock env):
    python scripts/nsweep_coverage.py --smoke        # 1 complex, N=20, ~2 min
    python scripts/nsweep_coverage.py --n-max 500    # full 30-complex sweep
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAPIDOCK = ROOT / "third_party" / "RAPiDock"
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python"
MODEL_DIR = RAPIDOCK / "train_models" / "CGTensorProductEquivariantModel"
BENCH = ROOT / "data" / "benchmark30.csv"
PREFIXES = [100, 200, 300, 500]

# Configurable at runtime (via CLI) so we can eval ANY checkpoint — pretrained
# (base third_party/RAPiDock) OR the retrained finetuned model. Set in main().
_INFER_DIR = RAPIDOCK          # dir containing inference.py to invoke
_MODEL_DIR = MODEL_DIR         # dir containing model_parameters.yml + ckpt
_CKPT = "rapidock_local.pt"    # checkpoint filename within _MODEL_DIR


def _kabsch(P, Q):
    """Superposed RMSD: best-fit rotation+translation of P onto Q (conformation only)."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    V, S, W = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(W.T @ V.T))
    U = W.T @ np.diag([1.0, 1.0, d]) @ V.T
    return float(np.sqrt(((Qc - Pc @ U.T) ** 2).sum(1).mean()))


def ca_rmsd(ref_pdb: str, pose_pdb: str) -> tuple[float, float] | None:
    """Return (direct receptor-frame Ca RMSD [docking], superposed Ca RMSD [conformation])."""
    import MDAnalysis as mda
    import warnings
    warnings.filterwarnings("ignore")
    try:
        a = mda.Universe(ref_pdb).select_atoms("name CA").positions
        b = mda.Universe(pose_pdb).select_atoms("name CA").positions
    except Exception:
        return None
    if len(a) == 0 or len(a) != len(b):
        return None
    direct = float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
    return direct, _kabsch(b, a)


def run_one(name: str, receptor: str, seq: str, n: int, out_root: Path,
            steps: int, batch: int) -> list[tuple[float, float]]:
    """Run RAPiDock inference for one complex; return per-pose (direct, superposed) RMSDs."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(["complex_name", "protein_description", "peptide_description"])
        w.writerow([name, receptor, seq])
        csv_path = tf.name

    cmd = [
        RAPIDOCK_PY, str(_INFER_DIR / "inference.py"),
        "--protein_peptide_csv", csv_path,
        "--out", str(out_root), "--output_dir", str(out_root),
        "--model_dir", str(_MODEL_DIR), "--ckpt", _CKPT,
        "--N", str(n), "--batch_size", str(batch),
        "--inference_steps", str(steps), "--actual_steps", str(steps),
        "--no_final_step_noise",
        "--conformation_partial", "1:1:1",
        "--scoring_function", "none", "--confidence_ckpt", "null",
        "--cpu", "4",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_INFER_DIR), capture_output=True, text=True)
    dt = time.time() - t0
    os.unlink(csv_path)

    pose_dir = out_root / name
    poses = sorted(glob.glob(str(pose_dir / "rank*.pdb")),
                   key=lambda p: int("".join(c for c in Path(p).stem.split("_")[0][4:] if c.isdigit()) or 0))
    if not poses:
        print(f"[{name}] NO POSES (dt={dt:.0f}s). stderr tail:\n{proc.stderr[-800:]}", flush=True)
        return []

    crystal_pep = receptor.replace("_protein_pocket.pdb", "_peptide.pdb")
    pairs = [ca_rmsd(crystal_pep, p) for p in poses]
    pairs = [r for r in pairs if r is not None]
    if pairs:
        direct = [d for d, _ in pairs]
        best = min(direct)
        nn = sum(1 for d in direct if d <= 2.0)
        print(f"[{name}] {len(pairs)} poses dt={dt:.0f}s best_direct={best:.2f}A "
              f"nearnative={nn}/{len(pairs)}", flush=True)
    else:
        print(f"[{name}] poses saved but RMSD failed (crystal={crystal_pep})", flush=True)
    return pairs


def summarize(results: dict[str, tuple[str, list[tuple[float, float]]]]) -> None:
    print("\n" + "=" * 78)
    print("COVERAGE SUMMARY  (DIRECT=docking placement | SUP=superposed conformation)")
    print("=" * 78)
    classes = sorted({ss for ss, _ in results.values()}) + ["ALL"]
    for N in PREFIXES:
        any_row = any(len(p) >= 1 for _, p in results.values())
        if not any_row:
            continue
        print(f"\n--- N = {N} ---")
        for cls in classes:
            bd, bs, hitd, tot_nn, tot_p, ncx = [], [], 0, 0, 0, 0
            for ss, pairs in results.values():
                if cls != "ALL" and ss != cls:
                    continue
                pref = pairs[:N]
                if not pref:
                    continue
                ncx += 1
                direct = [d for d, _ in pref]
                sup = [s for _, s in pref]
                bd.append(min(direct))
                bs.append(min(sup))
                hitd += 1 if min(direct) <= 2.0 else 0
                tot_nn += sum(1 for d in direct if d <= 2.0)
                tot_p += len(direct)
            if ncx == 0:
                continue
            print(f"  {cls:8s} n={ncx:2d}  best_DIRECT={np.mean(bd):5.2f}A  "
                  f"best_SUP={np.mean(bs):5.2f}A  Hit@2A_direct={hitd/ncx:4.0%}  "
                  f"near-native={tot_nn/max(tot_p,1):5.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="1 complex, N=20")
    ap.add_argument("--n-max", type=int, default=500)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("/tmp/claude-1000/nsweep"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench", type=Path, default=BENCH)
    ap.add_argument("--infer-dir", type=Path, default=RAPIDOCK,
                    help="dir with inference.py (use RAPiDock_finetuned to eval retrained ckpt)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="dir with model_parameters.yml + checkpoint (default: <infer-dir>/train_models/...)")
    ap.add_argument("--ckpt", type=str, default="rapidock_local.pt",
                    help="checkpoint filename within --model-dir")
    args = ap.parse_args()

    global _INFER_DIR, _MODEL_DIR, _CKPT
    _INFER_DIR = args.infer_dir.resolve()
    _MODEL_DIR = ((args.model_dir if args.model_dir is not None
                   else args.infer_dir / "train_models" / "CGTensorProductEquivariantModel")).resolve()
    _CKPT = args.ckpt

    rows = list(csv.DictReader(open(args.bench)))
    if args.smoke:
        rows, args.n_max = rows[:1], 20
    elif args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[nsweep] {len(rows)} complexes x N={args.n_max}, steps={args.steps}, "
          f"batch={args.batch} -> {args.out}", flush=True)

    results: dict[str, tuple[str, list[tuple[float, float]]]] = {}
    rmsd_csv = args.out / "per_pose_rmsd.csv"
    with open(rmsd_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "ss_class", "pose_idx", "rmsd_direct", "rmsd_superposed"])
        for i, r in enumerate(rows):
            name, ss = r["name"], r.get("ss_class", "NA")
            print(f"\n[{i+1}/{len(rows)}] {name} ({ss}, {r.get('pep_len','?')}-mer)", flush=True)
            pairs = run_one(name, r["receptor"], r["seq"], args.n_max, args.out,
                            args.steps, args.batch)
            results[name] = (ss, pairs)
            for j, (d, s) in enumerate(pairs):
                wr.writerow([name, ss, j, f"{d:.3f}", f"{s:.3f}"])
            fh.flush()

    summarize(results)
    print(f"\n[nsweep] per-pose RMSDs -> {rmsd_csv}", flush=True)


if __name__ == "__main__":
    main()
