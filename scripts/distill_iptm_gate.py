#!/usr/bin/env python
"""Before training on Boltz's answers: does ipTM actually pick the ones Boltz got RIGHT?

THE ASSUMPTION THE WHOLE DISTILLATION RESTS ON. The corpus is 454 co-folded complexes and the
plan filters them by ipTM, keeping the confident ones as training targets. That filter is only
meaningful if ipTM separates correct co-folds from wrong ones. Nobody has checked. Four previous
finetunes died at exactly this class of unexamined premise -- a training signal that was assumed
rather than measured -- so it is measured here first.

THE ONLY TEST SET THAT CAN ANSWER IT. For the eighteen designed complexes in the Coventry grid we
hold the paper's own structures, and we co-folded the cognate pairs. So for those we know both
what Boltz predicted and what the answer was. Co-folding sits at 3.42 A median there against our
10.88 A, but it is not uniformly right: roughly nine of fourteen land under 5 A and the rest do
not. If ipTM ranks the nine above the rest, the filter is real and a high-ipTM corpus is a clean
training target. If it does not, then filtering by ipTM selects confident mistakes, and the
distillation would teach our generator to reproduce them.

Four columns with broken binder identity (pc11, pc17, pc18, pc35) are excluded -- for those the
sequence we co-folded is not the protein the paper measured, so their RMSD says nothing about
Boltz.

Reported as a rank correlation and as the separation between the under-5 A group and the rest,
because the filter is a THRESHOLD decision and a correlation alone would not tell us where to put
it.

Usage: distill_iptm_gate.py
"""
from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
BOLTZ = ROOT / "runs/coventry/boltz"
LOGS = ["logs/coventry_boltz.jsonl", "logs/coventry_boltz_w1.jsonl",
        "logs/coventry_boltz_w2.jsonl", "logs/coventry_boltz_retry5.jsonl"]
GOOD = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc12", "pc21", "pc26", "pc28",
        "pc34", "pc43", "pc44", "pc46"]


AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path, chain: str | None = None):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        if chain is not None and l[21] != chain:
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def ca(p: Path) -> np.ndarray:
    return ca_seq(p)[0]


def best_offset(a: str, b: str) -> int:
    """Offset into `a` that maximises identity against `b` — handles the MSG tag."""
    best = (0, -1.0)
    for off in range(0, 10):
        x = a[off:]
        n = min(len(x), len(b))
        if n < 30:
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
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def main() -> None:
    conf = {}
    for lg in LOGS:
        f = ROOT / lg
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            nm = r.get("name") or r.get("pair") or ""
            if r.get("iptm") is not None:
                conf[nm] = float(r["iptm"])

    print(f"{len(conf)} co-folds with a recorded ipTM\n")
    print(f"  {'target':<8}{'ipTM':>8}{'RMSD to paper':>16}{'':>4}")
    rows = []
    for p in GOOD:
        pdb = BOLTZ / f"{p}__{p}" / "transplanted.pdb"
        if not pdb.exists():
            pdb = BOLTZ / f"{p}__{p}" / "cofolded.pdb"
        truth = PAPER / f"{p}_peptide.pdb"
        if not (pdb.exists() and truth.exists()):
            continue
        key = next((k for k in conf if k.startswith(f"{p}__{p}") or k == f"{p}__{p}B"), None)
        if key is None:
            print(f"  {p:<8}{'-':>8}{'(no ipTM logged)':>16}")
            continue
        # THE CO-FOLD IS IN BOLTZ'S OWN FRAME, so it must be superimposed before any RMSD is
        # taken. A first version of this script compared raw coordinates and reported 0/14 under
        # 5 A, against an established 9/14 -- the contradiction is what exposed the omission.
        # Superimpose the co-folded RECEPTOR onto the paper's binder (sequence-aligned, because
        # our models carry an MSG tag the paper's do not), then carry the peptide through the
        # same transform. That measures peptide error in the receptor frame, which is the
        # quantity that matters.
        chains: dict[str, list[str]] = {}
        for l in pdb.read_text().splitlines():
            if l.startswith("ATOM"):
                chains.setdefault(l[21], []).append(l)
        if len(chains) < 2:
            continue
        pepch = min(chains, key=lambda c: len({x[22:27] for x in chains[c]}))
        recch = max(chains, key=lambda c: len({x[22:27] for x in chains[c]}))
        q, _ = ca_seq(pdb, pepch)
        mrec, mseq = ca_seq(pdb, recch)
        prec, pseq = ca_seq(PAPER / f"{p}_1b1.pdb")
        if len(mrec) < 30 or len(prec) < 30:
            continue
        off = best_offset(mseq, pseq)
        mm = mrec[off:]
        n = min(len(mm), len(prec))
        R, t = kabsch(mm[:n], prec[:n])
        r = rmsd((R @ q.T).T + t, ca(truth))
        if not np.isfinite(r):
            continue
        rows.append((p, conf[key], r))
        print(f"  {p:<8}{conf[key]:>8.3f}{r:>16.2f}{'  <5A' if r < 5 else '':>6}")

    if len(rows) < 6:
        print("\n  too few paired points to test the gate")
        return

    ip = np.array([r[1] for r in rows])
    rm = np.array([r[2] for r in rows])
    from scipy import stats
    sp = stats.spearmanr(ip, rm)
    pe = stats.pearsonr(ip, rm)
    print(f"\n  n = {len(rows)}")
    print(f"  Spearman(ipTM, RMSD)  {sp.statistic:+.3f}   p = {sp.pvalue:.3f}")
    print(f"  Pearson (ipTM, RMSD)  {pe.statistic:+.3f}   p = {pe.pvalue:.3f}")
    print("     (NEGATIVE is what we want: higher confidence, lower error)")

    hit, miss = rm < 5.0, rm >= 5.0
    if hit.sum() and miss.sum():
        print(f"\n  mean ipTM where Boltz was RIGHT  (<5 A, n={hit.sum()}):  {ip[hit].mean():.3f}")
        print(f"  mean ipTM where Boltz was WRONG (>=5 A, n={miss.sum()}):  {ip[miss].mean():.3f}")
        print(f"  separation {ip[hit].mean() - ip[miss].mean():+.3f}   "
              f"Mann-Whitney p = {stats.mannwhitneyu(ip[hit], ip[miss]).pvalue:.3f}")
        auc = float(np.mean([(x > ip[miss]).mean() + 0.5 * (x == ip[miss]).mean()
                             for x in ip[hit]]))
        print(f"  AUC of ipTM as a correctness classifier: {auc:.3f}")
        works = auc > 0.70 and sp.statistic < 0
    else:
        works = False
        auc = float("nan")

    print("\n  -> " + (
        "ipTM DOES separate Boltz's right answers from its wrong ones. Filtering the corpus on it "
        "selects correct targets, and the distillation set is sound."
        if works else
        "ipTM does NOT reliably separate right from wrong on this geometry class. A high-ipTM "
        "filter would select CONFIDENT co-folds, not correct ones -- so the corpus must be cut "
        "another way, or the training target is partly Boltz's mistakes."))


if __name__ == "__main__":
    main()
