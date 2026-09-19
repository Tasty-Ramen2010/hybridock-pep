#!/usr/bin/env python
"""Option 2, tested before it is built: how much does a rigid, slightly-wrong receptor cost us?

THE PROPOSAL was to let RAPiDock denoise the receptor as well as the peptide -- flexible-receptor
docking. That is a real change to the diffusion state and the training loop, so it is worth
knowing first whether receptor conformation is actually costing us anything, because there is a
cheap experiment that answers it and a month of work that would otherwise answer it slowly.

THE EXPERIMENT. Dock the same peptide, with the same checkpoint, against two receptors:

  PREDICTED   our ESMFold model of the binder -- an unbound prediction, which is what the tool
              uses at inference and what every Coventry number so far was produced against
  HOLO        the receptor chain lifted out of the paper's own designed complex, i.e. the
              receptor in exactly the conformation the real peptide binds to

The peptide, the sequence, the sampler, the checkpoint and the pose count are identical. The only
thing that differs is whether the receptor is in its bound conformation. So the difference in
peptide RMSD is precisely the price of receptor rigidity -- an upper bound on it, in fact, since
HOLO is the perfect answer a flexible model could only approach.

HOW TO READ IT. A large gap means receptor conformation is a real bottleneck and flexibility is
worth building. A small gap means the receptor was never the problem, our peptide placement is
wrong for other reasons, and adding receptor degrees of freedom would add search space without
adding accuracy -- which on a model this size would likely make things worse.

Ground truth is the designed peptide pose from the same complex. The four identity-broken columns
are excluded.

Usage: receptor_flexibility_test.py [--dock]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Derive the repo root instead of hardcoding it: this script also runs on the DGX, where
# the tree lives at ~/Ram_Work/hybridock-pep. A hardcoded /home/igem path has broken a
# DGX run three times before.
ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
ESM = ROOT / "datasets/coventry/binders"
WORK = ROOT / "datasets/flexrec"
POSES = ROOT / "runs/flexrec"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
BAD_ID = {"pc11", "pc17", "pc18", "pc35"}
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def best_offset(a, b):
    best = (0, -1.0)
    for off in range(0, 10):
        x = a[off:]
        n = min(len(x), len(b))
        if n < 40:
            continue
        idt = sum(1 for u, v in zip(x[:n], b[:n]) if u == v) / n
        if idt > best[1]:
            best = (off, idt)
    return best[0]


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dock", action="store_true")
    a = ap.parse_args()

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}
    good = [p for p in ORDER if p not in BAD_ID]
    WORK.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm, src in (("predicted", ESM), ("holo", PAPER)):
        for p in good:
            f = src / f"{p}_1b1.pdb"
            if not f.exists():
                continue
            d = WORK / f"{arm}_{p}"
            d.mkdir(exist_ok=True)
            (d / "receptor.pdb").write_text(f.read_text())
            rows.append({"name": f"{arm}_{p}", "receptor": str(d / "receptor.pdb"),
                         "peptide_pdb": str(d / "receptor.pdb"), "seq": seqs[p],
                         "arm": arm, "target": p})
    bench = ROOT / "data/flexrec_bench.csv"
    with bench.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} docking runs ({len(good)} targets x 2 receptor states)\n")

    if a.dock:
        e = dict(os.environ); e["OMP_NUM_THREADS"] = "1"
        subprocess.run([os.environ.get("HDP_RAPIDOCK_PY",
                        str(Path.home() / "miniconda3/envs/rapidock/bin/python")),
                        str(ROOT / "scripts/exp_runner.py"), "--bench", str(bench),
                        "--limit", "9999", "--n", "24", "--steps", "16",
                        "--infer-dir", str(ROOT / "third_party/RAPiDock_finetuned"),
                        "--model-dir", str(ROOT / "third_party/RAPiDock_finetuned/longft_tanh"),
                        "--ckpt", "rapidock_finetuned_epoch010.pt",
                        "--out", str(POSES), "--partial-mode", "fixed", "--partial", "1:1:1",
                        "--label", "flexrec"], env=e, check=False)

    print(f"  {'target':<8}{'predicted rec':>15}{'holo rec':>11}{'gain':>9}")
    pr, ho = [], []
    for p in good:
        tb, pseq = ca_seq(PAPER / f"{p}_1b1.pdb")
        tp, _ = ca_seq(PAPER / f"{p}_peptide.pdb")
        vals = {}
        for arm, src in (("predicted", ESM), ("holo", PAPER)):
            cands = sorted((POSES / f"{arm}_{p}").glob("rank*.pdb")) \
                if (POSES / f"{arm}_{p}").exists() else []
            if not cands:
                vals[arm] = None; continue
            q, _ = ca_seq(cands[0])
            if arm == "holo":
                vals[arm] = rmsd(q, tp)          # already in the paper frame
            else:
                m, mseq = ca_seq(src / f"{p}_1b1.pdb")
                off = best_offset(mseq, pseq)
                mm = m[off:]
                n = min(len(mm), len(tb))
                R, t = kabsch(mm[:n], tb[:n])
                vals[arm] = rmsd((R @ q.T).T + t, tp)
        if vals.get("predicted") is None or vals.get("holo") is None:
            continue
        pr.append(vals["predicted"]); ho.append(vals["holo"])
        print(f"  {p:<8}{vals['predicted']:>15.2f}{vals['holo']:>11.2f}"
              f"{vals['predicted'] - vals['holo']:>+9.2f}")

    if not pr:
        print("\n  no poses yet — rerun with --dock")
        return
    from scipy import stats
    pr, ho = np.array(pr), np.array(ho)
    print(f"\n  predicted receptor: median {np.median(pr):5.2f} A   <=5A {(pr <= 5).sum()}/{len(pr)}")
    print(f"  holo receptor:      median {np.median(ho):5.2f} A   <=5A {(ho <= 5).sum()}/{len(ho)}")

    # THE TEST IS PAIRED, so the statistic must be too. An earlier version of this script
    # reported median(predicted) - median(holo) = +3.17 A and called the effect real. That is
    # the difference of two medians computed over DIFFERENT targets' orderings, not the median
    # of the per-target differences, and here the two disagree wildly: +3.17 against +0.27.
    # Every target is docked in both arms, so pair them.
    d = pr - ho
    better = int((ho < pr).sum())
    print(f"\n  paired difference (predicted - holo), the statistic that actually applies:")
    print(f"    median {np.median(d):+5.2f} A    mean {d.mean():+5.2f} A    sd {d.std(ddof=1):5.2f} A")
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"    95% CI on the mean {d.mean() - 1.96 * se:+.2f} to {d.mean() + 1.96 * se:+.2f} A")
    print(f"    holo better on {better}/{len(pr)} targets"
          f"   sign test p={stats.binomtest(better, len(pr), 0.5).pvalue:.3f}"
          f"   Wilcoxon p={stats.wilcoxon(pr, ho).pvalue:.3f}")

    real = stats.wilcoxon(pr, ho).pvalue < 0.05 and np.median(d) > 2.0
    print("\n  -> " + (
        "receptor conformation IS costing us; flexible-receptor docking is worth building."
        if real else
        f"NULL. A PERFECT receptor moves the median by {np.median(d):+.2f} A and wins on "
        f"{better}/{len(pr)} — a coin flip. Note {(pr <= 5).sum()}/{len(pr)} and "
        f"{(ho <= 5).sum()}/{len(ho)} under 5 A: handing the model the exact bound conformation "
        f"does not get a single target right that was not already right. Receptor rigidity is "
        f"not the bottleneck; adding receptor degrees of freedom would enlarge the search "
        f"without fixing placement."))


if __name__ == "__main__":
    main()
