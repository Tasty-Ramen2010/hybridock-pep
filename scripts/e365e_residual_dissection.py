"""E365e — where do we STILL fail on the external Wang set (bungarotoxin removed, n=43)?

Dissect the residuals: worst offenders, and which peptide property (length, charge, hydrophobicity, aromatic /
proline content, true affinity) the error concentrates in. Goal: name the single biggest remaining failure mode.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e365e_residual_dissection.py
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
KD = set("DE"); KB = set("KR"); HYD = set("AILMFWVY"); ARO = set("FWY")


def main():
    feat = {}
    for l in (ROOT / "data/ppikb_features.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); feat[r["pdb"].upper()] = r
    rows = []
    for r in csv.DictReader(open(ROOT / "data/hybridock_wang2024_external_regular.csv")):
        if "bungarotoxin" in r["protein"].lower():
            continue
        s = r["peptide"]; aa = [c for c in s if c.isalpha()]; n = max(len(aa), 1)
        exp = float(r["exp"]); pred = float(r["pred"])
        rows.append(dict(pdb=r["pdb"], protein=r["protein"], peptide=s, exp=exp, pred=pred,
                         signed=pred - exp, abs=abs(pred - exp), length=len(aa),
                         net_charge=sum(c in KB for c in aa) - sum(c in KD for c in aa),
                         abs_charge=abs(sum(c in KB for c in aa) - sum(c in KD for c in aa)),
                         f_hyd=sum(c in HYD for c in aa) / n, f_aro=sum(c in ARO for c in aa) / n,
                         f_pro=sum(c == "P" for c in aa) / n))
    df = pd.DataFrame(rows)
    print(f"n={len(df)}  MAE={df['abs'].mean():.2f}  r={pearsonr(df.pred, df.exp)[0]:+.2f}\n")

    print("=== correlation of |error| with each property ===")
    for c in ["exp", "length", "abs_charge", "net_charge", "f_hyd", "f_aro", "f_pro"]:
        r, p = pearsonr(df[c], df["abs"])
        print(f"  |err| vs {c:11s}: r={r:+.3f} (p={p:.2f})")
    print("\n=== correlation of SIGNED error (pred-exp; + = under-binding) ===")
    for c in ["exp", "net_charge", "f_pro", "f_hyd"]:
        r, p = pearsonr(df[c], df["signed"])
        print(f"  signed vs {c:11s}: r={r:+.3f} (p={p:.2f})")

    def binned(col, edges, labels):
        print(f"\n=== MAE by {col} ===")
        b = pd.cut(df[col], edges, labels=labels, include_lowest=True)
        g = df.groupby(b, observed=True).agg(n=("abs", "size"), MAE=("abs", "mean"), mean_signed=("signed", "mean"))
        print(g.round(2).to_string())

    binned("exp", [-13, -9, -7, -3], ["tight (≤-9)", "mid (-9..-7)", "weak (>-7)"])
    binned("length", [0, 6, 12, 30], ["short (≤6)", "mid (7-12)", "long (13+)"])
    binned("abs_charge", [-1, 1, 3, 10], ["neutral (0-1)", "charged (2-3)", "hi-charge (4+)"])
    binned("f_pro", [-0.01, 0.001, 0.15, 1.0], ["no-Pro", "some-Pro", "Pro-rich (>15%)"])

    print("\n=== 8 worst remaining ===")
    print(df.sort_values("abs", ascending=False).head(8)[
        ["pdb", "protein", "peptide", "length", "net_charge", "exp", "pred", "signed", "abs"]
    ].to_string(index=False, max_colwidth=22))


if __name__ == "__main__":
    main()
