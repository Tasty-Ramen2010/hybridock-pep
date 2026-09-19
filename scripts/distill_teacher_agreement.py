#!/usr/bin/env python
"""Did the training take at all? Measured on held-out complexes, where we have the power to see it.

THE PROBLEM WITH THE NULL WE JUST GOT. Both distilled checkpoints came back null on the fourteen
designed complexes: epoch 3 better on 7/14 (p=1.000), epoch 15 better on 5/14 (p=0.241). But the
paired differences have sd 5.29 A, so at n=14 the minimum detectable effect is 3.96 A -- a third
of the entire error we are trying to remove. That bench cannot see a 1 A improvement, or a 2 A
one, and it never will: fourteen is every designed complex for which a structure exists. A null
measured with that much noise is weak evidence, and reporting it as "distillation does not work"
would be overclaiming.

THE QUESTION THIS ANSWERS INSTEAD, and why it is worth asking separately. Distillation has an
explicit training objective -- move our poses toward the co-folded ones -- and whether that
objective was achieved is a different question from whether achieving it helps. On held-out
complexes we can measure it with n in the hundreds, because the target does not require a crystal
structure. So:

  DID WE MOVE TOWARD THE TEACHER?   agreement with the co-folded pose, held-out complexes
  DID THAT MAKE US MORE CORRECT?    the n=14 ground-truth bench, already run

Reading the two together is the point. Moved-and-no-gain means the teacher's poses are not
actually better on this class, or that agreement does not transfer to accuracy. Did-not-move
means the training never took and the corpus or learning rate is the problem, not the idea.
Neither conclusion is available from either measurement alone.

HELD OUT MEANS HELD OUT. The comparison set is the gap-check complexes whose receptors were
excluded from training, so no groove here was seen during the run. The shipped arm's poses on
exactly these complexes already exist from the gap check, which makes this a clean paired
comparison with no baseline to re-derive.

THIS IS AGREEMENT WITH A TEACHER, NOT CORRECTNESS. The targets are Boltz predictions; ipTM >= 0.85
co-folds are right about 80% of the time on the one set where we can check. Moving toward them is
evidence the optimisation worked, never evidence the poses became right.

Usage: distill_teacher_agreement.py --ckpt <name> --model-dir <dir> --label <tag> [--dock]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
WORK = ROOT / "datasets/distill_dock"
BASE_POSES = ROOT / "runs/distill_gap"
GAP_BENCH = ROOT / "data/distill_gap_bench.csv"


def ca_of(p: Path) -> np.ndarray:
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

    trained = {r["complex_name"] for r in csv.DictReader(open(ROOT / "data/distill2_train.csv"))
               if r["source"] == "cofold_distill"}
    trained |= {r["complex_name"] for r in csv.DictReader(open(ROOT / "data/distill2_val.csv"))}
    rows = [r for r in csv.DictReader(GAP_BENCH.open()) if r["name"] not in trained]
    print(f"{len(rows)} held-out complexes (receptors never seen in training)\n")

    bench = ROOT / "data/distill_heldout_bench.csv"
    with bench.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    poses = ROOT / f"runs/distill_heldout/{a.label}"
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

    print("AGREEMENT WITH THE CO-FOLDED POSE — held out, paired against the shipped arm\n")
    new, old = [], []
    for r in rows:
        n = r["name"]
        tgt = WORK / n / "cofold_ca.npy"
        if not tgt.exists():
            continue
        cof = np.load(tgt)
        bc = sorted((BASE_POSES / n).glob("rank*.pdb")) if (BASE_POSES / n).exists() else []
        nc = sorted((poses / n).glob("rank*.pdb")) if (poses / n).exists() else []
        if not bc or not nc:
            continue
        b, c = rmsd(ca_of(bc[0]), cof), rmsd(ca_of(nc[0]), cof)
        if np.isfinite(b) and np.isfinite(c):
            old.append(b); new.append(c)

    if len(new) < 10:
        print(f"  only {len(new)} paired — rerun with --dock")
        return
    new, old = np.array(new), np.array(old)
    d = new - old
    from scipy import stats
    print(f"  {'':<22}{'n':>5}{'median':>9}{'mean':>8}{'<5A':>7}{'<10A':>7}")
    print(f"  {'shipped arm':<22}{len(old):>5}{np.median(old):>9.2f}{old.mean():>8.2f}"
          f"{int((old < 5).sum()):>7}{int((old < 10).sum()):>7}")
    print(f"  {a.label:<22}{len(new):>5}{np.median(new):>9.2f}{new.mean():>8.2f}"
          f"{int((new < 5).sum()):>7}{int((new < 10).sum()):>7}")
    better = int((new < old).sum())
    w = stats.wilcoxon(new, old)
    print(f"\n  paired: median {np.median(d):+.2f} A   mean {d.mean():+.2f} A   sd {d.std(ddof=1):.2f}")
    print(f"  closer to the teacher on {better}/{len(d)}"
          f"   sign p={stats.binomtest(better, len(d), 0.5).pvalue:.4f}"
          f"   Wilcoxon p={w.pvalue:.4f}")
    mde = (1.96 + 0.84) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"  minimum detectable effect at this n: {mde:.2f} A "
          f"(the n=14 ground-truth bench sits at 3.96 A)")

    moved = w.pvalue < 0.05 and np.median(d) < 0
    print("\n  -> " + (
        "the training DID take: on grooves it never saw, our poses moved toward the teacher. "
        "Read with the flat ground-truth result, that means agreement did not buy correctness."
        if moved else
        "NO measurable move toward the teacher on held-out grooves."))
    if not moved:
        # DO NOT read this as "the training did not take" without checking the weights -- an
        # earlier version of this line said exactly that and it was wrong for run 2, whose
        # convolutions moved 1.186e-01 (15x run 1) on a train loss that fell 0.4518 -> 0.3645.
        # A null here with weights that demonstrably moved is a real negative about the method
        # at this corpus size; a null with weights that did not move says only that the
        # optimiser was too gentle. distill_conv_check.py separates the two in seconds.
        print("     Whether that is a negative about DISTILLATION or about the OPTIMISER depends "
              "on\n     whether the placement weights actually moved -- run distill_conv_check.py "
              "before\n     concluding either. Weights that moved + no effect = a real negative.")


if __name__ == "__main__":
    main()
