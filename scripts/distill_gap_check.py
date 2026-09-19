#!/usr/bin/env python
"""Is there anything to distil? Dock the corpus ourselves and measure the gap we would be closing.

THE QUESTION THAT DECIDES WHETHER RETRAIN #5 HAPPENS. We now hold 454 co-folded designed-groove
complexes. Training on them is only worth GPU time if our own sampler puts the peptide somewhere
DIFFERENT. If RAPiDock already produces what Boltz produces on these, distillation would be
training the model toward where it already is, and the four previous finetunes all died at exactly
this step -- nobody checked whether the target differed from the starting point.

THE RECEPTOR IS HELD FIXED SO THE COMPARISON IS ABOUT PLACEMENT. Each co-folded complex is split,
and its receptor chain -- Boltz's own -- is handed to RAPiDock as the docking target. Both methods
therefore see the identical receptor and the only thing that differs is where the peptide went.
Docking against our separately-folded monomer instead would confound placement error with a
receptor-modelling difference.

WHAT DISAGREEMENT DOES AND DOES NOT MEAN. These are synthetic pairs with no crystal, so a large
RMSD does not by itself prove we are wrong. But on the fourteen designed complexes where we DO
have the paper's structure, our poses sit at median 10.88 A and co-folded ones at 3.42 A, with
0/14 versus 9/14 under 5 A. On this geometry class, when the two disagree, the co-folded pose is
the one that has been right. That prior is what licenses reading disagreement as headroom -- and
it is a prior, so it is stated rather than assumed.

Reports the distribution, not just a mean: a corpus where we match on most and miss badly on a few
implies something very different for training than a uniform moderate gap.

Usage: distill_gap_check.py [--n 150] [--dock]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import os
from pathlib import Path

import numpy as np

# Derive the repo root instead of hardcoding it: this script also runs on the DGX, where
# the tree lives at ~/Ram_Work/hybridock-pep. A hardcoded /home/igem path has broken a
# DGX run three times before.
ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
CORPUS = ROOT / "logs/distill_corpus.jsonl"
WORK = ROOT / "datasets/distill_dock"
BENCH = ROOT / "data/distill_gap_bench.csv"
POSES = ROOT / "runs/distill_gap"
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def split_complex(p: Path):
    """(receptor lines, peptide CA coords) — peptide is the shorter chain."""
    ch, cur = {}, None
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM"):
            continue
        ch.setdefault(l[21], []).append(l)
    if len(ch) < 2:
        return None, None
    def nres(ls):
        return len({x[22:27] for x in ls})
    ks = sorted(ch, key=lambda c: -nres(ch[c]))
    rec, pep = ch[ks[0]], ch[ks[1]]
    ca = np.array([(float(x[30:38]), float(x[38:46]), float(x[46:54]))
                   for x in pep if x[12:16].strip() == "CA"])
    return rec, ca


def ca_of(pdb: Path) -> np.ndarray:
    out, seen = [], set()
    for l in pdb.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen:
                continue
            seen.add(k)
            out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(out)


def rmsd(A, B):
    n = min(len(A), len(B))
    if n < 3:
        return float("nan")
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dock", action="store_true", help="run the docking (else analyse only)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    usable = [r for r in rows if r.get("usable") and r.get("pdb") and Path(r["pdb"]).exists()]
    usable.sort(key=lambda r: -(r.get("iptm") or 0))
    sel = usable[:a.n]
    print(f"{len(usable)} usable co-folded complexes; taking the {len(sel)} highest-ipTM "
          f"(median {st.median(r['iptm'] for r in sel):.2f})\n")

    WORK.mkdir(parents=True, exist_ok=True)
    bench = []
    for r in sel:
        rec, ca = split_complex(Path(r["pdb"]))
        if rec is None or ca is None or len(ca) < 4:
            continue
        d = WORK / r["name"]
        d.mkdir(parents=True, exist_ok=True)
        rp = d / "receptor.pdb"
        rp.write_text("\n".join(rec) + "\nTER\nEND\n")
        np.save(d / "cofold_ca.npy", ca)
        bench.append({"name": r["name"], "receptor": str(rp), "seq": r["pep_seq"],
                      "peptide_pdb": str(d / "receptor.pdb"), "pep_len": r["pep_len"],
                      "iptm": r["iptm"]})
    with BENCH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bench[0]))
        w.writeheader(); w.writerows(bench)
    print(f"  wrote {len(bench)} docking targets -> {BENCH.relative_to(ROOT)}")

    if a.dock:
        print("\n  docking with the shipped checkpoint (longft_tanh ep010)...", flush=True)
        env = {"HDP_INFER_DIR": "third_party/RAPiDock_finetuned",
               "COVENTRY_MODEL_DIR": "third_party/RAPiDock_finetuned/longft_tanh",
               "COVENTRY_CKPT": "rapidock_finetuned_epoch010.pt"}
        import os
        e = dict(os.environ); e.update(env); e["OMP_NUM_THREADS"] = "1"
        subprocess.run([os.environ.get("HDP_RAPIDOCK_PY",
                        str(Path.home() / "miniconda3/envs/rapidock/bin/python")),
                        str(ROOT / "scripts/exp_runner.py"), "--bench", str(BENCH),
                        "--limit", "9999", "--n", "24", "--steps", "16",
                        "--infer-dir", str(ROOT / "third_party/RAPiDock_finetuned"),
                        "--model-dir", str(ROOT / "third_party/RAPiDock_finetuned/longft_tanh"),
                        "--ckpt", "rapidock_finetuned_epoch010.pt",
                        "--out", str(POSES), "--partial-mode", "fixed", "--partial", "1:1:1",
                        "--label", "distill_gap"], env=e, check=False)

    print("\nTHE GAP — our pose vs the co-folded pose, same receptor\n")
    gaps, iptms, lens = [], [], []
    for b in bench:
        d = WORK / b["name"]
        cof = np.load(d / "cofold_ca.npy")
        cands = sorted((POSES / b["name"]).glob("rank*.pdb")) if (POSES / b["name"]).exists() \
            else []
        if not cands:
            continue
        ours = ca_of(cands[0])
        g = rmsd(ours, cof)
        if np.isfinite(g):
            gaps.append(g); iptms.append(b["iptm"]); lens.append(b["pep_len"])
    if not gaps:
        print("  no docked poses yet — rerun with --dock")
        return
    gaps = np.array(gaps)
    print(f"  {len(gaps)} complexes compared")
    print(f"  CA RMSD between our pose and the co-folded pose:")
    print(f"    median {np.median(gaps):.2f} A   mean {gaps.mean():.2f} A   "
          f"range {gaps.min():.2f}-{gaps.max():.2f}")
    for lo, hi, lbl in ((0, 2, "agree closely   (<2A)"), (2, 5, "roughly agree   (2-5A)"),
                        (5, 10, "disagree        (5-10A)"), (10, 999, "disagree badly  (>10A)")):
        n = int(((gaps >= lo) & (gaps < hi)).sum())
        print(f"    {lbl:<24}{n:>5}  ({100 * n / len(gaps):>5.1f}%)")
    print(f"\n  corr(gap, co-fold ipTM)      {np.corrcoef(gaps, iptms)[0, 1]:+.3f}")
    print(f"  corr(gap, peptide length)    {np.corrcoef(gaps, lens)[0, 1]:+.3f}")
    big = gaps >= 5
    print(f"\n  -> " + (
        f"{100 * big.mean():.0f}% of the corpus disagrees by 5A or more. There IS a gap to close."
        if big.mean() > 0.4 else
        f"only {100 * big.mean():.0f}% disagrees by 5A+. We largely already produce what Boltz "
        f"produces here, so distillation would be training toward where we already are."))


if __name__ == "__main__":
    main()
