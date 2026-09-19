#!/usr/bin/env python
"""Did the distillation actually move docking? Paired, same receptors, direct RMSD.

WHY THIS AND NOT THE VAL LOSS. The run ended with val 0.405 and a "new best" banner, and that
number means nothing here. Score-matching loss is dominated by the high-noise regimes where the
target is nearly isotropic, so a model can fit it well while placing peptides 11 A from the
answer -- which is exactly the state we are trying to fix. Worse, the gate-collapse channel makes
val loss ANTI-correlated with pose quality, and every diffusion checkpoint this project has ever
selected on val loss has been worse at docking. So selection happens here, on placement.

THE COMPARISON IS THE SAME ONE WE ALREADY RAN. The receptor-flexibility test docked these
fourteen designed complexes against the paper's own binder chains and the shipped checkpoint
scored median 11.73 A, 0/14 under 5 A. That is the baseline, it was measured with identical
receptors, identical sequences, identical sampler settings and identical pose counts, and this
re-runs exactly it with the distilled weights. One variable changes: the checkpoint.

THIS BENCHMARK IS CLEAN FOR THIS RUN. Every Coventry receptor was excluded from the training
corpus precisely so that these fourteen stay held out. That was not free -- it cost 82 of the 130
high-ipTM complexes -- and it is what makes this number interpretable rather than a memorisation
score like longgeo's.

Poses are in the paper frame already, because the receptor handed to the sampler IS the paper's
binder chain, so no superposition is needed and none is done -- superimposing here would hide
placement error, which is the entire quantity of interest.

Usage: distill_eval.py --ckpt <name> --model-dir <dir> --label <tag> [--dock]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
WORK = ROOT / "datasets/distill_eval"
GOOD = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc12", "pc21", "pc26", "pc28",
        "pc34", "pc43", "pc44", "pc46"]
#: the shipped arm on this exact bench, from receptor_flexibility_test.py's holo column
BASELINE = {"n1": 14.02, "n2": 5.72, "n3": 17.98, "n4": 24.48, "n7": 27.11, "pc2": 11.18,
            "pc12": 20.95, "pc21": 17.00, "pc26": 12.29, "pc28": 5.14, "pc34": 10.83,
            "pc43": 5.02, "pc44": 10.54, "pc46": 8.30}


def ca(p: Path) -> np.ndarray:
    out, seen = [], set()
    for l in p.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen:
                continue
            seen.add(k)
            out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(out)


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dock", action="store_true")
    a = ap.parse_args()

    poses = ROOT / f"runs/distill_eval/{a.label}"
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in GOOD}

    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in GOOD:
        f = PAPER / f"{p}_1b1.pdb"
        if not f.exists():
            continue
        d = WORK / p
        d.mkdir(exist_ok=True)
        (d / "receptor.pdb").write_text(f.read_text())
        rows.append({"name": p, "receptor": str(d / "receptor.pdb"),
                     "peptide_pdb": str(d / "receptor.pdb"), "seq": seqs[p]})
    bench = ROOT / "data/distill_eval_bench.csv"
    with bench.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    if a.dock:
        e = dict(os.environ); e["OMP_NUM_THREADS"] = "1"
        subprocess.run([os.environ.get("HDP_RAPIDOCK_PY",
                        str(Path.home() / "miniconda3/envs/rapidock/bin/python")),
                        str(ROOT / "scripts/exp_runner.py"), "--bench", str(bench),
                        "--limit", "9999", "--n", "24", "--steps", "16",
                        "--infer-dir", str(ROOT / "third_party/RAPiDock_finetuned"),
                        "--model-dir", a.model_dir, "--ckpt", a.ckpt,
                        "--out", str(poses), "--partial-mode", "fixed", "--partial", "1:1:1",
                        "--label", a.label], env=e, check=False)

    print(f"\n{a.label} vs the shipped arm — same receptors, same peptides, direct RMSD\n")
    print(f"  {'target':<8}{'shipped':>10}{'this run':>11}{'delta':>9}")
    new, old = [], []
    for p in GOOD:
        cands = sorted((poses / p).glob("rank*.pdb")) if (poses / p).exists() else []
        if not cands:
            continue
        r = rmsd(ca(cands[0]), ca(PAPER / f"{p}_peptide.pdb"))
        if not np.isfinite(r):
            continue
        b = BASELINE[p]
        new.append(r); old.append(b)
        print(f"  {p:<8}{b:>10.2f}{r:>11.2f}{r - b:>+9.2f}")

    if not new:
        print("\n  no poses yet — rerun with --dock")
        return
    new, old = np.array(new), np.array(old)
    from scipy import stats
    d = new - old
    print(f"\n  shipped:   median {np.median(old):5.2f} A   <=5A {(old <= 5).sum()}/{len(old)}"
          f"   <=10A {(old <= 10).sum()}/{len(old)}")
    print(f"  this run:  median {np.median(new):5.2f} A   <=5A {(new <= 5).sum()}/{len(new)}"
          f"   <=10A {(new <= 10).sum()}/{len(new)}")
    better = int((new < old).sum())
    print(f"\n  paired difference: median {np.median(d):+.2f} A   mean {d.mean():+.2f} A")
    print(f"  better on {better}/{len(d)}"
          f"   sign p={stats.binomtest(better, len(d), 0.5).pvalue:.3f}"
          f"   Wilcoxon p={stats.wilcoxon(new, old).pvalue:.3f}")
    print("\n  -> " + (
        "the distillation IMPROVED placement on designed grooves."
        if np.median(d) < -1.0 and stats.wilcoxon(new, old).pvalue < 0.05 else
        "NULL or noise on the target class. Report it as such — this is the fifth retrain and "
        "four of the previous four also looked plausible before they were measured."))


if __name__ == "__main__":
    main()
