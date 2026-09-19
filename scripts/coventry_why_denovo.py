#!/usr/bin/env python
"""What, specifically, about these de novo binders do we fail to model?

Four questions, each answered with a measurement rather than an argument.

  A  THE COMPRESSION.  Ram: "it almost seems like a cubic curve where the edges are hard to guess
     and so off, but our inside values fit very well."  Test it: regress experiment on our
     prediction, read the slope, and check whether the error is a function of how extreme the true
     affinity is.  A slope below 1 with error growing toward the edges IS shrinkage toward the
     mean; a genuine cubic would need the quadratic and cubic terms to earn their place.

  B  DO WE MODEL THE RECEPTOR AT ALL?  E221/E222 found that ~75% of affinity variance is a
     per-receptor baseline -- how tightly a pocket binds peptides in general -- and that no static
     representation predicts it (pocket composition 0.049, ProtDCal 0.149, ESM-2 0.154).  The grid
     lets us test this directly for the first time on designed binders: each binder is scored
     against all 18 peptides, so its COLUMN MEAN is our model's estimate of its general
     stickiness.  If that tracks the binder's measured cognate affinity, we have baseline signal.
     If it does not, we have none, and every cognate number is a peptide guess on a blind pocket.

  C  HOW FAR OUTSIDE OUR WORLD ARE THESE MOLECULES?  Compare the 18 peptides and 18 binders
     against the 8,130-complex training corpus on pose-independent descriptors -- length, charge,
     hydrophobicity, and sequence complexity.  The answer decides whether this is a data problem
     we can fix by training or a physics problem we cannot.

  D  WHAT r IS EVEN REACHABLE HERE?  The 18 cognate affinities span 2.2 nM to 230 nM -- a total
     spread of 2.76 kcal/mol and a standard deviation of 0.76.  Our model's own MAE on the data it
     was built for is ~1.2 kcal/mol.  When the noise is larger than the signal, correlation is
     capped by arithmetic before any modelling question is asked.  Compute that cap.

Usage: coventry_why_denovo.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
RT = 0.0019872041 * 298.0
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
POS, NEG = set("KR"), set("DE")
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def entropy(seq: str) -> float:
    """Shannon entropy of the residue composition, in bits. Low = repetitive."""
    n = len(seq)
    c = Counter(seq)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def descriptors(seq: str) -> dict:
    n = max(len(seq), 1)
    q = sum(1 for a in seq if a in POS) - sum(1 for a in seq if a in NEG)
    return {
        "length": len(seq),
        "net_charge_per_res": q / n,
        "abs_charge_per_res": sum(1 for a in seq if a in POS | NEG) / n,
        "hydropathy": sum(KD.get(a, 0) for a in seq) / n,
        "entropy_bits": entropy(seq),
        "max_repeat_frac": max(Counter(seq).values()) / n,
    }


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")


def seq_of_pdb(p: Path) -> str:
    return "".join(AA3.get(l[17:20].strip(), "")
                   for l in p.read_text().splitlines()
                   if l.startswith("ATOM") and l[12:16].strip() == "CA")


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    pep_seq = {r["name"]: r["sequence"]
               for r in csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    bind_seq = {r["name"].removesuffix("_1b1"): r["sequence"]
                for r in csv.DictReader(open(ROOT / "data/coventry_binders.csv"))}

    def load(path, field="best_iface"):
        M = {}
        for line in (ROOT / path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "name" in r:
                    a, b = r["name"].split("__")
                    b = b[:-1] if b.endswith("B") else b
                else:
                    a, b = r["peptide"], r["binder"]
                if r.get(field) is not None:
                    M[(a, b)] = r[field]
        return M

    dg = load("logs/coventry_dg.jsonl")
    ref = load("logs/coventry_refine_fixed.jsonl")
    exp = {p: RT * math.log(float(truth[(p, p)]["kd_M"])) for p in ORDER}

    y = np.array([exp[p] for p in ORDER])
    x = np.array([dg[(p, p)] for p in ORDER])

    # ---------------- A. the compression ------------------------------------
    print("=" * 78)
    print("A.  THE COMPRESSION — is it shrinkage toward the mean, and is it cubic?")
    print("=" * 78)
    slope, intercept = np.polyfit(x, y, 1)
    print(f"  regress experiment on our prediction:  slope {slope:+.3f}  "
          f"(1.000 would be calibrated)")
    print(f"  dynamic range   experiment {y.max() - y.min():.2f} kcal/mol   "
          f"ours {x.max() - x.min():.2f}   ratio {(x.max() - x.min()) / (y.max() - y.min()):.2f}")
    print(f"  sd              experiment {y.std(ddof=1):.2f}   ours {x.std(ddof=1):.2f}")
    err = x - y
    dev = np.abs(y - y.mean())
    print(f"\n  corr(|error|, how extreme the TRUE affinity is) = {pearson(np.abs(err), dev):+.3f}")
    print(f"  corr(signed error, true dG)                     = {pearson(err, y):+.3f}"
          f"   <- positive means we under-bind the tight ones")
    lo = [p for p in ORDER if exp[p] <= np.median(y)]
    hi = [p for p in ORDER if exp[p] > np.median(y)]
    print(f"  MAE on the TIGHT half (n={len(lo)}): "
          f"{st.mean([abs(dg[(p, p)] - exp[p]) for p in lo]):.2f} kcal/mol")
    print(f"  MAE on the WEAK  half (n={len(hi)}): "
          f"{st.mean([abs(dg[(p, p)] - exp[p]) for p in hi]):.2f} kcal/mol")

    print("\n  is a curve needed, or is a straight line enough?")
    for deg in (1, 2, 3):
        c = np.polyfit(x, y, deg)
        resid = y - np.polyval(c, x)
        ss = 1 - resid.var() / y.var()
        # adjusted for the extra parameters, because n=18 rewards any added term
        adj = 1 - (1 - ss) * (len(y) - 1) / (len(y) - deg - 1)
        print(f"    degree {deg}: R2 {ss:+.3f}   adjusted R2 {adj:+.3f}")
    print("  (a cubic that does not beat the line on ADJUSTED R2 is fitting noise, not curvature)")

    # ---------------- B. do we model the receptor baseline? -----------------
    print("\n" + "=" * 78)
    print("B.  DO WE MODEL THE RECEPTOR AT ALL? — column mean vs measured cognate affinity")
    print("=" * 78)
    print("  Each binder is scored against all 18 peptides, so its column mean is our estimate")
    print("  of how sticky that pocket is in general. Does it track the real cognate affinity?\n")
    for label, M in (("our calibrated dG", dg), ("ref2015 interface", ref)):
        colmu = np.array([st.mean([M[(p, b)] for p in ORDER]) for b in ORDER])
        rowmu = np.array([st.mean([M[(p, b)] for b in ORDER]) for p in ORDER])
        print(f"  {label}")
        print(f"     corr(binder column mean, its cognate dG) = {pearson(colmu, y):+.3f}"
              f"   <- the receptor baseline")
        print(f"     corr(peptide row mean,  its cognate dG) = {pearson(rowmu, y):+.3f}"
              f"   <- the peptide main effect")
        inter = np.array([M[(p, p)] for p in ORDER]) - rowmu - colmu + st.mean(M.values())
        print(f"     corr(interaction term,   its cognate dG) = {pearson(inter, y):+.3f}"
              f"   <- what specificity actually is")

    # ---------------- C. how far outside our world -------------------------
    print("\n" + "=" * 78)
    print("C.  HOW FAR OUTSIDE THE TRAINING CORPUS ARE THESE MOLECULES?")
    print("=" * 78)
    corpus = []
    seen = set()
    for r in csv.DictReader(open(ROOT / "data/longft_train_longgeo.csv")):
        if r["complex_name"] in seen:
            continue
        seen.add(r["complex_name"])
        p = Path(r["peptide_description"])
        if p.exists():
            s = seq_of_pdb(p)
            if 5 <= len(s) <= 30:
                corpus.append(descriptors(s))
        if len(corpus) >= 4000:
            break
    cov = [descriptors(pep_seq[p]) for p in ORDER]
    print(f"  corpus peptides sampled: {len(corpus)}   Coventry peptides: {len(cov)}\n")
    print(f"  {'descriptor':<22}{'corpus median':>15}{'Coventry median':>17}"
          f"{'z':>7}{'percentile':>12}")
    for k in ("length", "net_charge_per_res", "abs_charge_per_res", "hydropathy",
              "entropy_bits", "max_repeat_frac"):
        cv = [d[k] for d in corpus]
        vv = [d[k] for d in cov]
        mu, sd = st.mean(cv), (st.pstdev(cv) or 1e-9)
        z = (st.median(vv) - mu) / sd
        pct = 100 * sum(1 for c in cv if c <= st.median(vv)) / len(cv)
        print(f"  {k:<22}{st.median(cv):>15.2f}{st.median(vv):>17.2f}{z:>+7.2f}{pct:>11.0f}%")
    print("\n  per-peptide sequence complexity (2.0 bits ~ a 4-letter repeat):")
    for p in ORDER:
        d = descriptors(pep_seq[p])
        print(f"     {p:>6} {pep_seq[p]:<20} entropy {d['entropy_bits']:.2f} bits   "
              f"charge/res {d['net_charge_per_res']:+.2f}")
    print("\n  binders:")
    bd = [descriptors(bind_seq[b]) for b in ORDER if b in bind_seq]
    for k in ("length", "net_charge_per_res", "abs_charge_per_res", "hydropathy",
              "entropy_bits"):
        print(f"     {k:<22} median {st.median([d[k] for d in bd]):.2f}")

    # ---------------- D. the reachable ceiling ------------------------------
    print("\n" + "=" * 78)
    print("D.  WHAT CORRELATION IS EVEN REACHABLE ON A 2.8 kcal/mol WINDOW?")
    print("=" * 78)
    sig = y.std(ddof=1)
    print(f"  true signal sd on these 18 cognates: {sig:.2f} kcal/mol "
          f"(span {y.max() - y.min():.2f})")
    for noise in (0.6, 0.9, 1.2, 1.5):
        cap = sig / math.sqrt(sig ** 2 + noise ** 2)
        print(f"  if our per-complex error were {noise:.1f} kcal/mol, the BEST attainable "
              f"r is {cap:.2f}")
    print(f"\n  our measured MAE here is 1.74, and ~1.2 on the data the model was built for.")
    print(f"  so even a perfectly calibrated version of this model tops out near r = "
          f"{sig / math.sqrt(sig ** 2 + 1.2 ** 2):.2f} on this window.")
    print(f"  measured: r = {pearson(x, y):+.3f}")


if __name__ == "__main__":
    main()
