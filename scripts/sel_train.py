#!/usr/bin/env python
"""Train and validate a selectivity-specific scorer on the synthetic all-by-all blocks.

THE HYPOTHESIS BEING TESTED, STATED BEFORE THE NUMBERS.

  H1  Double-centring alone helps.  score(i,j) = grand + peptide(i) + binder(j) + interaction(i,j).
      Everything we systematically cannot model -- a peptide's desolvation offset, its length, a
      binder's overall stickiness -- is a MAIN EFFECT, constant down a row or column.  Specificity
      IS the interaction term by definition.  So subtracting both main effects from ANY score,
      including plain ref2015, should improve cognate ranking without any learning at all.

  H2  Learning a term reweighting beats the fixed ref2015 weights.  ref2015's weights were fitted
      for structure prediction, not for discrimination.  We have a specific reason to expect a
      different weighting is better: fa_atr alone separates cognates more cleanly than the total,
      because fa_rep and fa_sol both scale with how buried the peptide is and cancel it.

  H3  Typed chemistry and complementarity add signal beyond the energy terms.  Chemistry-typed
      contacts are what made hydrophobic complementarity work where chemistry-blind BSA did not.

TWO MODELS, BECAUSE THEY HAVE DIFFERENT DEPLOYMENT SCOPES.
  raw       trained on the pair features as they are.  Scores ONE peptide against ONE receptor,
            so it works for an isolated query.
  centred   trained on double-centred features.  Needs a panel -- several peptides against
            several receptors -- because the centring estimates the main effects from the panel
            itself.  That is not a restriction in practice: a selectivity screen IS a panel.

VALIDATION IS LEAVE-BLOCK-OUT.  A block is 10 complexes and 100 pairs; holding one out entirely
means no receptor and no peptide from it was ever seen.  The metric is the one the task actually
asks for -- where does the true partner rank among the 10 candidates -- not regression r, because
a scorer can correlate well overall and still never put the right answer first.

READ COLUMN-WISE, NOT ROW-WISE.  There is a backbone artifact built into this construction and it
only affects one direction, so the two directions are not equally trustworthy:

  ROW-wise (fix peptide i, vary receptor j) is CONTAMINATED.  Cell (i,j) threads peptide i onto
    receptor j's crystal backbone, so the diagonal cell (i,i) is the only one in the row sitting
    on the backbone that was actually crystallised with that peptide.  A perfect native fit is
    handed to the right answer for free.  Row-wise numbers here are an upper bound, not a result.

  COLUMN-wise (fix receptor j, vary peptide i) is CLEAN.  Every cell in a column uses the SAME
    receptor and the SAME backbone; only the threaded sequence changes.  No cell has a fit
    advantage the others lack, so the only thing that can separate them is interface chemistry.

So the headline number from this script is mean_rank_col, and it maps onto the Coventry
column-wise task (fix a binder, rank the 18 peptides).  Row-wise is reported for completeness
with the artifact called out.  The real row-wise test is the Coventry grid itself, where every
candidate gets a docked pose and nobody gets a free crystal backbone.

Usage: sel_train.py [--min-blocks 10]
Output: data/sel_model.joblib, logs/sel_train_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
import os
FEAT = ROOT / os.environ.get("SEL_FEAT", "logs/sel_features.jsonl")
MODEL = ROOT / os.environ.get("SEL_MODEL", "data/sel_model.joblib")
REPORT = ROOT / os.environ.get("SEL_REPORT", "logs/sel_train_report.json")

#: features that are properties of the PAIR. Anything peptide-only or receptor-only is excluded
#: by construction -- that is the entire point of this model.
ENERGY = ["i_fa_atr", "i_fa_rep", "i_fa_sol", "i_fa_elec", "i_lk_ball_wtd",
          "i_hbond_bb_sc", "i_hbond_sc", "i_hbond_sr_bb", "i_hbond_lr_bb", "i_total"]
CHEM = ["c_apolar_frac", "c_polar_frac", "c_opp_charge_frac", "c_like_charge_frac",
        "c_aromatic_frac", "c_mixed_frac"]
COMPL = ["hydro_pocket", "hydro_product", "hydro_absdiff", "charge_product",
         "pocket_charge_per_res"]
BURIAL = ["n_contact_per_res", "rec_atoms_touched_per_res", "burial_mean", "burial_sd",
          "burial_cv", "burial_max", "frac_res_contacting"]
#: OUR OWN terms only -- zero Rosetta. Typed contact chemistry, charge and hydrophobic
#: complementarity against the pocket the peptide actually touches, and the SHAPE of its burial
#: along the chain. If a selectivity score built from these works, it is ours end to end.
NO_ROSETTA = CHEM + COMPL + BURIAL
ALL_FEATS = ENERGY + CHEM + COMPL + BURIAL

import os as _os
if _os.environ.get("SEL_NO_ROSETTA"):
    ALL_FEATS = NO_ROSETTA


def load() -> tuple[dict, list[int]]:
    """Grid of feature dicts per block: grid[block][(i, j)] = row."""
    grid: dict[int, dict] = defaultdict(dict)
    for line in FEAT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or "i_total" not in r:
            continue
        grid[r["block"]][(r["pep_idx"], r["rec_idx"])] = r
    complete = []
    for b, cells in grid.items():
        n = max(max(k) for k in cells) + 1 if cells else 0
        if len(cells) == n * n and n >= 5:
            complete.append(b)
    return grid, sorted(complete)


def matrices(grid: dict, block: int, feats: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """(raw features, double-centred features, n) for one block, indexed [i, j, feature]."""
    cells = grid[block]
    n = max(max(k) for k in cells) + 1
    X = np.zeros((n, n, len(feats)))
    for (i, j), r in cells.items():
        X[i, j] = [float(r.get(f, 0.0)) for f in feats]
    # double-centre: subtract row and column means, add back the grand mean. What remains is
    # the interaction -- exactly the quantity a specificity assay measures.
    C = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean((0, 1), keepdims=True)
    return X, C, n


def rank_metrics(score: np.ndarray, axis: str = "row") -> dict:
    """Where does the true partner rank? score[i, j], lower = predicted better binding."""
    n = score.shape[0]
    ranks = []
    for k in range(n):
        v = score[k, :] if axis == "row" else score[:, k]
        ranks.append(int(np.argsort(np.argsort(v))[k]) + 1)
    return {"ranks": ranks, "mean": float(np.mean(ranks)),
            "top1": int(sum(r == 1 for r in ranks)), "top3": int(sum(r <= 3 for r in ranks)),
            "n": n}


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum(1.0 if p < q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def evaluate(scores: dict[int, np.ndarray]) -> dict:
    """Pool rank metrics and AUC across blocks."""
    rows, cols, pos, neg = [], [], [], []
    t1 = t3 = t1r = tot = 0
    for b, S in scores.items():
        rm = rank_metrics(S, "row")
        cm = rank_metrics(S, "col")
        rows += rm["ranks"]; cols += cm["ranks"]
        # top-k is reported COLUMN-wise: that is the direction with no backbone artifact
        t1 += cm["top1"]; t3 += cm["top3"]; t1r += rm["top1"]; tot += cm["n"]
        n = S.shape[0]
        for i in range(n):
            for j in range(n):
                (pos if i == j else neg).append(float(S[i, j]))
    return {"mean_rank_row": round(st.mean(rows), 3), "mean_rank_col": round(st.mean(cols), 3),
            "top1": t1, "top3": t3, "top1_row": t1r, "n": tot,
            "top1_pct": round(100 * t1 / tot, 1), "top3_pct": round(100 * t3 / tot, 1),
            "auc": round(auc(pos, neg), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-blocks", type=int, default=10)
    args = ap.parse_args()

    grid, blocks = load()
    if len(blocks) < args.min_blocks:
        print(f"only {len(blocks)} complete blocks; need {args.min_blocks}. "
              f"Let sel_features.py finish.")
        return
    n_per = matrices(grid, blocks[0], ALL_FEATS)[2]
    print(f"{len(blocks)} complete blocks, {n_per}x{n_per} each "
          f"= {len(blocks) * n_per * n_per} pairs")
    print(f"random baseline: mean rank {(n_per + 1) / 2:.1f}, "
          f"top1 {100 / n_per:.0f}%, top3 {300 / n_per:.0f}%\n")

    report: dict = {"blocks": len(blocks), "n_per_block": n_per, "features": ALL_FEATS}

    # ---- H1: does centring alone help a fixed score? --------------------------
    for label, centred in (("ref2015 i_total  RAW", False),
                           ("ref2015 i_total  DOUBLE-CENTRED", True)):
        S = {}
        for b in blocks:
            X, C, n = matrices(grid, b, ["i_total"])
            S[b] = (C if centred else X)[:, :, 0]
        m = evaluate(S)
        report[f"baseline_{'centred' if centred else 'raw'}"] = m
        print(f"{label:34s} COL {m['mean_rank_col']:.2f}  (row {m['mean_rank_row']:.2f}*)  "
              f"top1 {m['top1']:3d}/{m['n']} ({m['top1_pct']:.0f}%)  AUC {m['auc']:.3f}")

    # single-term baselines: is any one ref2015 term better than the total?
    print()
    best_single = None
    for t in ENERGY:
        S = {}
        for b in blocks:
            _, C, n = matrices(grid, b, [t])
            S[b] = C[:, :, 0]
        m = evaluate(S)
        if best_single is None or m["mean_rank_col"] < best_single[1]["mean_rank_col"]:
            best_single = (t, m)
        print(f"  centred single term {t:16s} COL {m['mean_rank_col']:.2f}  "
              f"row {m['mean_rank_row']:.2f}*  AUC {m['auc']:.3f}")
    report["best_single_term"] = {"term": best_single[0], **best_single[1]}

    # ---- H2/H3: learned models, leave-block-out ------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for mode, feats, label in (("raw", ALL_FEATS, "LEARNED raw features"),
                               ("centred", ALL_FEATS, "LEARNED double-centred"),
                               ("centred", ENERGY, "LEARNED centred, energy terms only"),
                               ("centred", ENERGY + CHEM + COMPL,
                                "LEARNED centred, no burial")):
        S = {}
        for held in blocks:
            Xtr, ytr = [], []
            for b in blocks:
                if b == held:
                    continue
                X, C, n = matrices(grid, b, feats)
                M = C if mode == "centred" else X
                for i in range(n):
                    for j in range(n):
                        Xtr.append(M[i, j]); ytr.append(int(i == j))
            Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(sc.transform(Xtr), ytr)
            X, C, n = matrices(grid, held, feats)
            M = C if mode == "centred" else X
            P = clf.predict_proba(sc.transform(M.reshape(-1, len(feats))))[:, 1]
            # higher P(cognate) = better binding, and the metrics expect lower = better
            S[held] = -P.reshape(n, n)
        m = evaluate(S)
        report[label] = m
        print(f"{label:34s} COL {m['mean_rank_col']:.2f}  (row {m['mean_rank_row']:.2f}*)  "
              f"top1 {m['top1']:3d}/{m['n']} ({m['top1_pct']:.0f}%)  "
              f"top3 {m['top3']:3d} ({m['top3_pct']:.0f}%)  AUC {m['auc']:.3f}")

    # ---- shuffle control ------------------------------------------------------
    # Permute which cell counts as cognate, inside each block, and retrain. If the pipeline is
    # measuring real chemistry this collapses to chance; if it stays high, something in the
    # construction leaks the answer and every number above is worthless.
    rng = np.random.default_rng(0)
    S = {}
    for held in blocks:
        Xtr, ytr = [], []
        for b in blocks:
            if b == held:
                continue
            X, C, n = matrices(grid, b, ALL_FEATS)
            perm = rng.permutation(n)
            for i in range(n):
                for j in range(n):
                    Xtr.append(C[i, j]); ytr.append(int(perm[i] == j))
        Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        X, C, n = matrices(grid, held, ALL_FEATS)
        P = clf.predict_proba(sc.transform(C.reshape(-1, len(ALL_FEATS))))[:, 1]
        S[held] = -P.reshape(n, n)
    m = evaluate(S)
    report["SHUFFLE CONTROL"] = m
    print(f"\n{'SHUFFLE CONTROL (labels permuted)':34s} COL {m['mean_rank_col']:.2f}  "
          f"top1 {m['top1']:3d}/{m['n']} ({m['top1_pct']:.0f}%)  AUC {m['auc']:.3f}"
          f"   <- must be at chance")

    # ---- final model on everything, for deployment ---------------------------
    import joblib
    Xtr, ytr = [], []
    for b in blocks:
        X, C, n = matrices(grid, b, ALL_FEATS)
        for i in range(n):
            for j in range(n):
                Xtr.append(C[i, j]); ytr.append(int(i == j))
    Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    joblib.dump({"scaler": sc, "clf": clf, "features": ALL_FEATS, "mode": "centred"}, MODEL)
    order = np.argsort(-np.abs(clf.coef_[0]))
    print(f"\nsaved {MODEL}\ntop weights (double-centred, standardised):")
    for k in order[:12]:
        print(f"   {ALL_FEATS[k]:26s} {clf.coef_[0][k]:+.3f}")
    report["weights"] = {ALL_FEATS[k]: round(float(clf.coef_[0][k]), 4) for k in order}
    REPORT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
