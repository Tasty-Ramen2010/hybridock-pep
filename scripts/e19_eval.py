"""E19 eval — test the three baseline-wall attacks honestly.

H1  BASELINE RECOVERY (the new one): can pocket descriptors predict the per-family
    baseline alpha_j? Fit alpha_grp = mean ΔG per group; regress on pocket features
    (leave-group-out). If R>0 cross-validated, we can reconstruct absolute ΔG on a
    NOVEL target. Then: absolute pred = recovered_alpha + within-interface term.
H2  LEARNED FAVORABILITY: decomposed sasa_hb/sasa_sb/sasa_apolar/sasa_unsat as
    SEPARATE features (regression learns signs) vs the v2 hand-weighted de_strength.
H3  LIGAND EFFICIENCY: per-residue (_pr) features to kill size bias.

Honest tests: cross-dataset transfer + leave-group-out within-target (same harness).
Verdict on whether ANY beats the standing baseline (hb+aromatic, within +0.45 / cross -0.54).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def _load():
    cr = json.loads(Path("/tmp/e19_cr.json").read_text())
    pb = json.loads(Path("/tmp/e19_pb.json").read_text())
    return cr, pb


def _fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w


def _mat(recs, feats):
    return np.array([[r.get(f, 0.0) for f in feats] for r in recs], float)


def transfer(cr, pb, feats):
    Xc, Xp = _mat(cr, feats), _mat(pb, feats)
    yc = np.array([r["y"] for r in cr]); yp = np.array([r["y"] for r in pb])
    p_cp = _fit_predict(Xc, yc, Xp); p_pc = _fit_predict(Xp, yp, Xc)
    return (pearsonr(p_cp, yp).statistic, pearsonr(p_pc, yc).statistic,
            float(np.sqrt(np.mean((p_cp - yp) ** 2))))


def logo(recs, feats):
    groups = {}
    for i, r in enumerate(recs):
        groups.setdefault(r["grp"], []).append(i)
    multi = {g: idx for g, idx in groups.items() if len(idx) >= 4}
    X = _mat(recs, feats); y = np.array([r["y"] for r in recs])
    pp, py = [], []
    for gid, te in multi.items():
        tr = [i for i in range(len(recs)) if recs[i]["grp"] != gid]
        pred = _fit_predict(X[tr], y[tr], X[te])
        if np.std(pred) > 0:
            pp.append(pred - pred.mean()); py.append(y[te] - y[te].mean())
    pr = pearsonr(np.concatenate(pp), np.concatenate(py)).statistic if pp else float("nan")
    return pr, len(multi)


# ---- H1: baseline recovery ----
POCKET = ["poc_n", "poc_f_hyd", "poc_f_pos", "poc_f_neg", "poc_net", "poc_f_arom",
          "poc_f_pol", "poc_eis"]


def baseline_recovery(recs, label):
    """Leave-group-out: predict per-group MEAN ΔG from pocket descriptors."""
    groups = {}
    for r in recs:
        groups.setdefault(r["grp"], []).append(r)
    multi = {g: rs for g, rs in groups.items() if len(rs) >= 4}
    if len(multi) < 4:
        # crystal-65 is all singletons; treat each pdb as its own target (alpha = its y)
        multi = {g: rs for g, rs in groups.items()}
    gids = list(multi.keys())
    # one pocket descriptor per group = mean over members; alpha = mean y
    Xg = np.array([[np.mean([r.get(f, 0.0) for r in multi[g]]) for f in POCKET] for g in gids])
    ag = np.array([np.mean([r["y"] for r in multi[g]]) for g in gids])
    if len(gids) < 5:
        return None
    preds = np.zeros(len(gids))
    for i in range(len(gids)):
        tr = [j for j in range(len(gids)) if j != i]
        preds[i] = _fit_predict(Xg[tr], ag[tr], Xg[i:i+1])[0]
    r = pearsonr(preds, ag).statistic
    rmse = float(np.sqrt(np.mean((preds - ag) ** 2)))
    print(f"  [{label}] baseline-recovery LOGO: r={r:+.3f} rmse={rmse:.2f} "
          f"(n_groups={len(gids)}, alpha spread={ag.std():.2f})")
    return r


FEATSETS = {
    "baseline hb+arom": ["hb_count", "arom_cc"],
    "H2 decomposed favorability": ["sasa_hb", "sasa_sb", "sasa_apolar", "sasa_unsat", "bsa_hyd"],
    "H3 ligand-efficiency (per-res)": ["bsa_hyd_pr", "sasa_hb_pr", "sasa_sb_pr",
                                       "sasa_unsat_pr", "hb_count_pr", "arom_cc_pr"],
    "H2+H3 combined": ["sasa_hb", "sasa_sb", "sasa_unsat", "bsa_hyd",
                       "bsa_hyd_pr", "sasa_hb_pr", "hb_count_pr"],
    "+pocket (H1 features inline)": ["hb_count", "arom_cc", "bsa_hyd"] + POCKET,
}


def main():
    cr, pb = _load()
    print(f"crystal={len(cr)} pepbi={len(pb)}\n")

    print("=== H1) BASELINE RECOVERY — predict per-family alpha from pocket structure ===")
    baseline_recovery(cr, "crystal")
    r_pb = baseline_recovery(pb, "pepbi")
    print("  (if r>0.3 cross-validated, absolute ΔG on novel targets becomes possible)\n")

    print("=== A) CROSS-DATASET TRANSFER (Pearson r) ===")
    print(f"{'featset':<34}{'cr->pb':>9}{'pb->cr':>9}{'RMSE':>8}")
    for name, fs in FEATSETS.items():
        r_cp, r_pc, rmse = transfer(cr, pb, fs)
        print(f"{name:<34}{r_cp:>9.3f}{r_pc:>9.3f}{rmse:>8.2f}")

    print("\n=== B) WITHIN-TARGET LEAVE-GROUP-OUT (PEPBI) ===")
    print(f"{'featset':<34}{'pooled r':>10}{'n_grp':>7}")
    base_pr = None
    for name, fs in FEATSETS.items():
        pr, ng = logo(pb, fs)
        if name == "baseline hb+arom":
            base_pr = pr
        print(f"{name:<34}{pr:>10.3f}{ng:>7}")

    print("\n=== VERDICT ===")
    h2_pr, _ = logo(pb, FEATSETS["H2 decomposed favorability"])
    h3_pr, _ = logo(pb, FEATSETS["H3 ligand-efficiency (per-res)"])
    comb_pr, _ = logo(pb, FEATSETS["H2+H3 combined"])
    print(f"  within-target baseline {base_pr:+.3f}")
    for nm, v in [("H2 learned favorability", h2_pr), ("H3 ligand-efficiency", h3_pr),
                  ("H2+H3", comb_pr)]:
        print(f"    {nm:<26}{v:+.3f}  ({'HELPS' if v > base_pr + 0.03 else 'no gain'})")
    print(f"  H1 baseline-recovery on PEPBI families: r={r_pb if r_pb is not None else float('nan'):+.3f}")
    print("  >> absolute ΔG unlocked?" ,
          "MAYBE" if (r_pb is not None and r_pb > 0.3) else "NO (per-family demean remains the ceiling)")


if __name__ == "__main__":
    main()
