#!/usr/bin/env python
"""The 2x2: is the deficit the POSES or the FEATURES? Same grid, same cancellation, both answers.

Rows are which structure was scored, columns are what was read off it. Everything is median
polished the same way and the Coventry grid is never trained on; the learned column applies the
model already fitted on docked all-by-all blocks, unchanged, so this is transfer and not a new fit.

A feature set that works on good poses and fails on ours is a pose problem. A feature set that
fails on both is a feature problem. The answer decides where the next month of work goes, and it
is not currently known -- which is the only reason this script exists.

Usage: sel_pose_vs_features.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
ENERGY = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
          "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb", "i_total"]
OURS = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
        "c_aromatic_frac", "c_mixed_frac", "hydro_pocket", "hydro_product", "hydro_absdiff",
        "charge_product", "pocket_charge_per_res", "n_contact_per_res",
        "rec_atoms_touched_per_res", "burial_mean", "burial_sd", "burial_cv", "burial_max",
        "frac_res_contacting"]
ALL = ENERGY + OURS


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def cube(path: Path, key: str = "best") -> np.ndarray:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if key in r:
                rows[(r["peptide"], r["binder"])] = r[key]
    miss = [(p, b) for p in ORDER for b in ORDER if (p, b) not in rows]
    if miss:
        print(f"  warning: {len(miss)} cells missing from {path.name}, filled with 0")
    A = np.nan_to_num(np.array([[[float(rows.get((p, b), {}).get(f, 0.0)) for f in ALL]
                                 for b in ORDER] for p in ORDER]), nan=0.0,
                      posinf=0.0, neginf=0.0)
    return np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)


def main() -> None:
    import joblib

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    keep = np.zeros((N, N), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            keep[i, j] = (i == j) or truth.get((p, b), {}).get("measured") == "0"

    def grade(S):
        cog = np.array([S[i, i] for i in range(N)])
        non = np.array([S[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        rr = [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]
        return a, float(np.mean(rr)), sum(r <= 3 for r in rr)

    C_dock = cube(ROOT / "logs/sel_coventry_features_hard.jsonl")
    C_cof = cube(ROOT / "logs/sel_cofold_features.jsonl")

    m = joblib.load(ROOT / "data/sel_model_docked.joblib")
    # the saved model is whichever set won; rebuild an ours-only model on the same blocks so
    # both cells of the 2x2 column are strictly Rosetta-free
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    grid = defaultdict(dict)
    for line in (ROOT / "logs/sel_dock_features.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                grid[r["block"]][(r["pep_idx"], r["rec_idx"])] = r["best"]
    X, y = [], []
    for cells in grid.values():
        n = max(max(k) for k in cells) + 1
        if len(cells) != n * n or n < 5:
            continue
        A = np.nan_to_num(np.array([[[float(cells[(i, j)].get(f, 0.0)) for f in OURS]
                                     for j in range(n)] for i in range(n)]), nan=0.0,
                          posinf=0.0, neginf=0.0)
        Cb = np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)
        X.append(Cb.reshape(-1, len(OURS))); y.append(np.eye(n, dtype=int).reshape(-1))
    X, y = np.vstack(X), np.concatenate(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced").fit(sc.transform(X), y)
    oi = [ALL.index(f) for f in OURS]

    def learned(C):
        Z = C[:, :, oi].reshape(-1, len(OURS))
        return -clf.predict_proba(sc.transform(Z))[:, 1].reshape(N, N)

    print("THE 2x2.  Same 18x18 grid, same median polish, Coventry never trained on.")
    print("The learned column is the docked-block model applied unchanged -- transfer, not a fit.\n")
    print(f"  {'pose scored':<34}{'ref2015 total':>16}{'our 18 features':>18}")
    for label, C in (("our docked poses", C_dock), ("co-folded poses", C_cof)):
        a1 = grade(C[:, :, ALL.index("i_total")])
        a2 = grade(learned(C))
        print(f"  {label:<34}{a1[0]:>16.3f}{a2[0]:>18.3f}")
    print(f"\n  {'':<34}{'(rank, top3)':>16}")
    for label, C in (("our docked poses", C_dock), ("co-folded poses", C_cof)):
        a1 = grade(C[:, :, ALL.index("i_total")])
        a2 = grade(learned(C))
        print(f"  {label:<34}{f'{a1[1]:.2f}, {a1[2]}':>16}{f'{a2[1]:.2f}, {a2[2]}':>18}")

    print("\nSINGLE FEATURE, co-folded poses vs our docked poses (sign taken from the co-folded")
    print("side, so a value below 0.500 in the docked column means it points the other way there).\n")
    print(f"  {'feature':<28}{'co-folded':>11}{'docked':>9}{'lift':>8}")

    def sauc(C, k, sign=None):
        v = C[:, :, k]
        cog = np.array([v[i, i] for i in range(N)])
        non = np.array([v[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        if sign is None:
            return (a, 1.0) if a >= 0.5 else (1 - a, -1.0)
        return (a if sign > 0 else 1 - a), sign

    out = []
    for f in ALL:
        k = ALL.index(f)
        ac, sign = sauc(C_cof, k)
        ad, _ = sauc(C_dock, k, sign)
        out.append((f, ac, ad))
    for f, ac, ad in sorted(out, key=lambda t: -t[1]):
        tag = "  ref2015" if f in ENERGY else ""
        print(f"  {f:<28}{ac:>11.3f}{ad:>9.3f}{ac - ad:>+8.3f}{tag}")

    ours_c = [t[1] for t in out if t[0] in OURS]
    ours_d = [t[2] for t in out if t[0] in OURS]
    print(f"\n  our features, mean single-feature AUC: co-folded {np.mean(ours_c):.3f}, "
          f"docked {np.mean(ours_d):.3f}")
    print(f"  best single feature of ours on co-folded poses: "
          f"{max((t for t in out if t[0] in OURS), key=lambda t: t[1])[0]} "
          f"{max(t[1] for t in out if t[0] in OURS):.3f}")


if __name__ == "__main__":
    main()
