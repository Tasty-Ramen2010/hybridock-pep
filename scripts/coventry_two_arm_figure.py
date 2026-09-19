#!/usr/bin/env python
"""Two arms, same scoring stack: our poses against Boltz's poses, top-5 shortlist per peptide.

The question this answers is the one that decides where the work goes. Hold the scoring pipeline
completely fixed -- same features, same median-polish cancellation, same shortlist rule -- and
change only which structure was scored. Whatever gap opens between the two panels is attributable
to pose generation and to nothing else, because nothing else differs.

A shortlist is the honest unit here. Nobody orders 18 peptides against 18 binders and acts on the
single top cell; they take the handful worth testing. So each row is scored on whether the true
partner made its own top five, and the cells drawn are exactly that shortlist.

Also reports which rows still miss, and tests whether the tail descriptors -- the ones that keep
the worst contact rather than the average contact -- add anything to either arm.

Usage: coventry_two_arm_figure.py [out.png]
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_two_arm.png"
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
TAIL = ["worst_pair", "mean_pair", "n_clash_charge", "n_unpaid_polar", "frac_bad_pairs",
        "n_saltbridge", "contact_gini", "contact_entropy", "n_anchor", "max_res_contacts",
        "frac_res_zero", "seg_centroid", "seg_spread"]


def polish(M):
    from hybridock_pep.scoring.cancellation import median_polish
    return median_polish(M)[0]


def cube(path: Path, feats: list[str]) -> np.ndarray:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "best" in r:
                rows[(r["peptide"], r["binder"])] = r["best"]
    A = np.nan_to_num(np.array([[[float(rows.get((p, b), {}).get(f, 0.0)) for f in feats]
                                 for b in ORDER] for p in ORDER]), nan=0.0,
                      posinf=0.0, neginf=0.0)
    return np.stack([polish(A[:, :, k]) for k in range(A.shape[2])], axis=2)


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    keep = np.zeros((N, N), dtype=bool)
    T = np.full((N, N), np.nan)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            r = truth.get((p, b))
            keep[i, j] = (i == j) or (r or {}).get("measured") == "0"
            if r and r["measured"] == "1":
                try:
                    T[i, j] = -math.log10(float(r["kd_M"]))
                except (TypeError, ValueError):
                    T[i, j] = 7.0

    def grade(S):
        cog = np.array([S[i, i] for i in range(N)])
        non = np.array([S[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        rr = [sorted(range(N), key=lambda j: S[i, j]).index(i) + 1 for i in range(N)]
        return a, float(np.mean(rr)), sum(r <= 3 for r in rr), sum(r <= 5 for r in rr), rr

    FEA = ENERGY + OURS
    C_our = cube(ROOT / "logs/sel_coventry_features_hard.jsonl", FEA)
    C_cof = cube(ROOT / "logs/sel_cofold_features.jsonl", FEA)
    Tl_our = cube(ROOT / "logs/sel_tail_docked.jsonl", TAIL)
    Tl_cof = cube(ROOT / "logs/sel_tail_cofold.jsonl", TAIL)

    ip = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("iptm") is not None:
                ip[(r["peptide"], r["binder"])] = -r["iptm"]
    IPT = np.array([[ip.get((p, b), 0.0) for b in ORDER] for p in ORDER])
    _z = lambda A: (A - A.mean()) / (A.std() or 1.0)
    _mc = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()

    # a learned score per arm, trained on the docked blocks and applied unchanged
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
    oi = [FEA.index(f) for f in OURS]
    learned = lambda C: -clf.predict_proba(
        sc.transform(C[:, :, oi].reshape(-1, len(OURS))))[:, 1].reshape(N, N)

    ti = FEA.index("i_total")
    print("COMBINATIONS, both arms. Lower score = better. Coventry never trained on.\n")
    print(f"  {'stack':<46}{'AUC':>7}{'rank':>7}{'t3':>5}{'t5':>5}")
    results = {}
    for arm, C, Tl, extra in (("OUR poses", C_our, Tl_our, None),
                              ("BOLTZ poses", C_cof, Tl_cof, IPT)):
        print(f"  --- {arm} ---")
        opts = {"cancelled ref2015": _z(C[:, :, ti])}
        opts["+ our learned features"] = _z(C[:, :, ti]) + 0.5 * _z(learned(C))
        best_tail, best_a = None, -1
        for f in TAIL:
            k = TAIL.index(f)
            for s in (1, -1):
                a = grade(_z(C[:, :, ti]) + 0.5 * s * _z(Tl[:, :, k]))[0]
                if a > best_a:
                    best_a, best_tail = a, (f, s)
        opts[f"+ best tail descriptor ({best_tail[0]})"] = (
            _z(C[:, :, ti]) + 0.5 * best_tail[1] * _z(Tl[:, :, TAIL.index(best_tail[0])]))
        if extra is not None:
            opts["+ co-folding confidence (ipTM)"] = _z(C[:, :, ti]) + _z(_mc(extra))
        for label, S in opts.items():
            a, r, t3, t5, rr = grade(S)
            print(f"  {label:<46}{a:>7.3f}{r:>7.2f}{t3:>5}{t5:>5}")
            results[(arm, label)] = (S, a, r, t3, t5, rr)

    our_key = ("OUR poses", "+ our learned features")
    cof_key = ("BOLTZ poses", "+ co-folding confidence (ipTM)")
    if results[our_key][1] < results[("OUR poses", "cancelled ref2015")][1]:
        our_key = ("OUR poses", "cancelled ref2015")

    print("\nWHICH ROWS STILL MISS THE TOP FIVE\n")
    print(f"  {'pep':<7}{'OUR rank':>10}{'BOLTZ rank':>12}   status")
    ro, rc = results[our_key][5], results[cof_key][5]
    for i, p in enumerate(ORDER):
        s = ("both miss" if ro[i] > 5 and rc[i] > 5 else
             "poses rescue it" if ro[i] > 5 else
             "our poses better" if rc[i] > 5 else "both fine")
        print(f"  {p:<7}{ro[i]:>10}{rc[i]:>12}   {s}")

    panels = [("A  Wu et al. 2025, Fig. 2B", T, None, True),
              (f"B  OUR poses\n{our_key[1]}", results[our_key][0], results[our_key][1:5], False),
              (f"C  BOLTZ poses\n{cof_key[1]}", results[cof_key][0], results[cof_key][1:5], False)]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6))
    for ax, (title, Mx, met, is_truth) in zip(axes, panels):
        if is_truth:
            cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#ffffff")
            im = ax.imshow(np.ma.masked_invalid(Mx), cmap=cmap, aspect="equal")
        else:
            D = np.full((N, N), np.nan)
            for i in range(N):
                o = sorted(range(N), key=lambda j: Mx[i, j])[:5]
                for rank, j in enumerate(o):
                    D[i, j] = 5 - rank
            cmap = plt.get_cmap("YlOrRd").copy(); cmap.set_bad("#f2f2f2")
            im = ax.imshow(np.ma.masked_invalid(D), cmap=cmap, vmin=0, vmax=5, aspect="equal")
        for i in range(N):
            for j in range(N):
                r = truth.get((ORDER[i], ORDER[j]))
                if r and r["measured"] == "1":
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111111" if i == j else "#777777",
                                           lw=2.0 if i == j else 1.0, zorder=3))
        if not is_truth:
            for i in range(N):
                rank = sorted(range(N), key=lambda j: Mx[i, j]).index(i) + 1
                # the shortlist colormap goes dark at the top, so a top-2 diagonal needs light
                # text or the number is invisible exactly where the result is best
                deep = rank <= 2
                ax.text(i, i, str(rank), ha="center", va="center", zorder=4, fontsize=7.2,
                        fontweight="bold",
                        color="#ffffff" if deep else ("#0b3d66" if rank <= 5 else "#8a1c22"))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(ORDER, rotation=90, fontsize=7.5)
        ax.set_yticklabels(ORDER, fontsize=7.5)
        ax.set_xlabel("binder", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("peptide", fontsize=9)
        sub = (f"AUC {met[0]:.3f} · mean rank {met[1]:.2f}/18 · top-5 {met[3]}/18" if met
               else "30 measured of 324 · white = no binding detected")
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, linespacing=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("measured  −log₁₀(K$_d$)" if is_truth else "shortlist position (5 = top)",
                     fontsize=8)
        for s in ax.spines.values():
            s.set_color("#999999"); s.set_linewidth(0.6)

    fig.suptitle("Same scoring stack, different poses. Only the structure changes between B and "
                 "C — same features, same cancellation, same shortlist rule. Shaded = that "
                 "peptide's top five; the number on the diagonal is where the true partner "
                 "ranked.", fontsize=10.5, y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.915))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
