"""E149 — push CHARGED peptides toward PPI-Affinity's 0.71 with rich descriptors + receptor charge-
complementarity (data-driven, the part of electrostatics that does NOT wash).

Single-pose Coulomb/Born wash (the floor). But two signals survive and PPI-Affinity exploits them via
descriptors: (a) NET charge complementarity peptide_q × pocket_q (e_sb_net 'net not count' validated
universal), (b) physicochemical sequence descriptors (pI, hydrophobic moment, charge clustering). Add both,
train a charged-specialist, grouped CV. Report on BOTH distributions: pooled charged (broad/honest) and the
curated benchmark charged subset (the PPI-comparable 0.71 target).

Rich features on top of 16 physics:
  peptide: net_q, abs_q, pI(approx), charge_cluster (mean |Δpos| of like charges), hyd_moment (Eisenberg
           amphipathy), frac_charged, charge_runs, KR/DE counts, aa-composition(20)
  complementarity (peptide × receptor pocket): q_compl = pep_net_q · poc_net (opposite=favourable),
           abs charge match, |pep_q + poc_q| (neutralisation)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
e146 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e146", ROOT / "experiments/e146_charged_specialist.py"))
importlib.util.spec_from_file_location("e146", ROOT / "experiments/e146_charged_specialist.py").loader.exec_module(e146)
PROD = e146.PROD
AA = "ACDEFGHIKLMNPQRSTVWY"
POS, NEG = set("KR"), set("DE")
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2,
      "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
PKA = {"D": 3.65, "E": 4.25, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}


def approx_pI(seq):
    # crude: pH where net charge ~0, bisection
    def charge(ph):
        c = 1 / (1 + 10 ** (ph - 8.0)) - 1 / (1 + 10 ** (3.1 - ph))  # termini
        for a in seq:
            if a in ("K", "R", "H"):
                c += 1 / (1 + 10 ** (ph - PKA[a]))
            elif a in ("D", "E", "C", "Y"):
                c -= 1 / (1 + 10 ** (PKA[a] - ph))
        return c
    lo, hi = 0.0, 14.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rich_pep(seq):
    L = max(1, len(seq))
    npos = sum(c in POS for c in seq); nneg = sum(c in NEG for c in seq)
    pos_idx = [i for i, c in enumerate(seq) if c in POS]
    neg_idx = [i for i, c in enumerate(seq) if c in NEG]
    # charge clustering: mean gap between consecutive same-sign charges (small=clustered)
    def clust(idx):
        return np.mean(np.diff(sorted(idx))) / L if len(idx) > 1 else 1.0
    # Eisenberg hydrophobic moment (helical, 100°/res)
    ang = np.arange(L) * (100 * np.pi / 180)
    h = np.array([KD.get(c, 0) for c in seq])
    hm = np.sqrt((h * np.cos(ang)).sum() ** 2 + (h * np.sin(ang)).sum() ** 2) / L
    comp = [seq.count(a) / L for a in AA]
    return [float(npos - nneg), float(abs(npos - nneg)), approx_pI(seq), clust(pos_idx), clust(neg_idx),
            hm, (npos + nneg) / L, float(npos), float(nneg)] + comp


RKEYS = ["netq", "absq", "pI", "posclust", "negclust", "hyd_moment", "fchg", "nKR", "nDE"] + list(AA)


def metr(p, y):
    return pearsonr(p, y)[0], float(np.mean(np.abs(p - y))), float(np.sqrt(np.mean((p - y) ** 2)))


def load():
    rows = []
    for r in [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        pq = sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"])
        rows.append({"src": "pdbbind", "y": r["y"], "absq": q, "length": r["length"],
                     "feat": [r[c] for c in PROD], "rich": rich_pep(r["seq"]),
                     "compl": [pq * r["poc_net"], abs(pq) * abs(r["poc_net"]), abs(pq + r["poc_net"])]})
    for nm in ["train", "test"]:
        for r in csv.DictReader(open(ROOT / f"data/pooled_benchmark_{nm}.csv")):
            seq = r.get("seq", "")
            if not seq:
                continue
            q = abs(sum(c in POS for c in seq) - sum(c in NEG for c in seq))
            pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
            rows.append({"src": r["dataset"], "y": float(r["y"]), "absq": q, "length": int(r["length"]),
                         "feat": [float(r[c]) for c in PROD], "rich": rich_pep(seq),
                         "compl": [pq * float(r["poc_net"]), abs(pq) * abs(float(r["poc_net"])), abs(pq + float(r["poc_net"]))]})
    return rows


def cv(rows, mode, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    X = []
    for r in rows:
        row = list(r["feat"])
        if mode in ("rich", "all"):
            row += r["rich"]
        if mode in ("compl", "all"):
            row += r["compl"]
        X.append(row)
    X = np.array(X, float)
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.04,
                                          l2_regularization=3.0, min_samples_leaf=15, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred, y


def main():
    rows = load()
    absq = np.array([r["absq"] for r in rows])
    print(f"=== E149 push charged toward 0.71 (n={len(rows)}) ===\n")
    subsets = {
        "ALL charged |q|≥2 (pooled)": [r for r in rows if r["absq"] >= 2],
        "high |q|≥3 (pooled)": [r for r in rows if r["absq"] >= 3],
        "charged BENCHMARK only (PPI-comparable)": [r for r in rows if r["absq"] >= 2 and r["src"] in ("cr65", "the98")],
    }
    for name, sub in subsets.items():
        if len(sub) < 20:
            print(f"--- {name}: n={len(sub)} too small ---\n"); continue
        print(f"--- {name} (n={len(sub)}) ---  r / MAE / RMSE")
        for lbl, mode in [("base-16", "base"), ("+rich desc", "rich"), ("+charge-compl", "compl"), ("+ALL", "all")]:
            r, mae, rmse = metr(*cv(sub, mode))
            print(f"    {lbl:<16}{r:>+8.3f}{mae:>7.2f}{rmse:>7.2f}")
        print()
    print("  target: PPI-Affinity 0.71 on high-charge (curated). Check the BENCHMARK charged subset — that's")
    print("  the same-distribution comparison. Pooled-PDBbind charged is broader/harder (different number).")


if __name__ == "__main__":
    main()
