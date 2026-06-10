"""E18 — combine the 3 stages, train weights, and test HONESTLY (the verdict).

Loads Stage1+2 features (/tmp/e18_{cr,pb}.json) + ESM coupling
(/tmp/e18_esm_coupling.json). Builds:
  de_sasa          (Stage 1, SASA x Eisenberg)
  tds_conf         (Stage 2, Σ ln n_basin · kT  ; no cooperativity)
  tds_conf_esm     (Stage 2+3, Σ ln n_basin·(1-λ·coupling_i) · kT ; ESM-discounted)
plus the H-bond+aromatic baseline geometry for ablation.

Honest tests (per plan):
  A) cross-dataset transfer (fit crystal -> predict PEPBI and reverse), absolute + sign
  B) leave-one-binding-group-out on PEPBI: per-group Spearman distribution
  C) ablation: combined vs each part vs hb+aromatic baseline
  D) absolute RMSE vs mean-baseline
Falsifiable: does the trained SYNTHESIS generalize better than its parts / the baseline?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e18_hybrid_features import N_BASIN, KB_T  # noqa: E402

LAMBDA = 0.7  # ESM coupling discount strength (the prompt's λ); fixed a-priori


def tds_conf_esm(seq, coupling):
    lnW = 0.0
    for i, aa in enumerate(seq):
        c = coupling[i] if coupling and i < len(coupling) else 0.0
        lnW += np.log(N_BASIN.get(aa, 4)) * (1.0 - LAMBDA * c)
    return KB_T * lnW


def build(records, esm):
    X, meta = [], []
    for r in records:
        seq = r["seq"]
        coup = esm.get(seq)
        te = tds_conf_esm(seq, coup) if coup else r["tds_conf"]
        X.append([r["de_sasa"], r["tds_conf"], te,
                  r.get("hb_count") or 0.0, r.get("aromatic_cc") or 0.0])
        meta.append(dict(y=r["y"], grp=r["grp"], seq=seq))
    return np.array(X, float), meta


FEAT = ["de_sasa", "tds_conf", "tds_conf_esm", "hb_count", "aromatic_cc"]
MODELS = {
    "Stage1 de_sasa": ["de_sasa"],
    "Stage2 tds_conf": ["tds_conf"],
    "Stage1+2 (SASA+entropy)": ["de_sasa", "tds_conf"],
    "Stage1+2+3 (+ESM coop)": ["de_sasa", "tds_conf_esm"],
    "baseline hb+aromatic": ["hb_count", "aromatic_cc"],
    "ALL combined": ["de_sasa", "tds_conf_esm", "hb_count", "aromatic_cc"],
}


def cols(idx_names):
    return [FEAT.index(n) for n in idx_names]


def fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Ate = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    return Ate @ w


def demean_by_group(X, meta):
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["grp"], []).append(i)
    return groups


def transfer(Xtr, mtr, Xte, mte, names):
    c = cols(names)
    ytr = np.array([m["y"] for m in mtr])
    yte = np.array([m["y"] for m in mte])
    pred = fit_predict(Xtr[:, c], ytr, Xte[:, c])
    return pearsonr(pred, yte).statistic, np.sqrt(np.mean((pred - yte) ** 2))


def logo_pergroup(X, meta, names):
    c = cols(names)
    groups = demean_by_group(X, meta)
    multi = {g: idx for g, idx in groups.items() if len(idx) >= 4}
    rhos = []
    pooled_pred, pooled_y = [], []
    for gid, te_idx in multi.items():
        tr_idx = [i for i in range(len(meta)) if meta[i]["grp"] != gid]
        ytr = np.array([meta[i]["y"] for i in tr_idx])
        # within-group demean train+test for slope estimation
        pred = fit_predict(X[np.ix_(tr_idx, c)], ytr, X[np.ix_(te_idx, c)])
        yte = np.array([meta[i]["y"] for i in te_idx])
        if np.std(pred) > 0:
            rhos.append(spearmanr(pred, yte).statistic)
            pooled_pred.append(pred - pred.mean()); pooled_y.append(yte - yte.mean())
    rhos = np.array(rhos)
    pr = pearsonr(np.concatenate(pooled_pred), np.concatenate(pooled_y)).statistic \
        if pooled_pred else float("nan")
    return rhos, pr


def main():
    cr = json.loads(Path("/tmp/e18_cr.json").read_text())
    pb = json.loads(Path("/tmp/e18_pb.json").read_text())
    esm_path = Path("/tmp/e18_esm_coupling.json")
    esm = json.loads(esm_path.read_text()) if esm_path.exists() else {}
    print(f"crystal={len(cr)} pepbi={len(pb)} esm_seqs={len([k for k in esm if k!='_mean'])}")
    Xc, mc = build(cr, esm)
    Xp, mp = build(pb, esm)

    print("\n=== A) CROSS-DATASET TRANSFER (fit A -> predict B), Pearson r ===")
    print(f"{'model':<26}{'cr->pb r':>10}{'pb->cr r':>10}{'cr->pb RMSE':>13}")
    for name, names in MODELS.items():
        r_cp, rmse_cp = transfer(Xc, mc, Xp, mp, names)
        r_pc, _ = transfer(Xp, mp, Xc, mc, names)
        print(f"{name:<26}{r_cp:>10.3f}{r_pc:>10.3f}{rmse_cp:>13.2f}")

    print("\n=== B) LEAVE-GROUP-OUT on PEPBI: per-group Spearman ===")
    print(f"{'model':<26}{'pooled r':>10}{'median ρ':>10}{'%correct':>10}{'n_grp':>7}")
    for name, names in MODELS.items():
        rhos, pr = logo_pergroup(Xp, mp, names)
        if len(rhos):
            print(f"{name:<26}{pr:>10.3f}{np.median(rhos):>10.2f}"
                  f"{np.mean(rhos>0):>9.0%}{len(rhos):>7}")

    print("\n=== D) ABSOLUTE kcal/mol (5-fold-ish: fit crystal, test pepbi) ===")
    ybar = np.array([m["y"] for m in mp])
    print(f"  PEPBI mean-baseline RMSE = {ybar.std():.2f}")
    for name, names in MODELS.items():
        _, rmse = transfer(Xc, mc, Xp, mp, names)
        print(f"  {name:<26} cross-dataset RMSE={rmse:.2f}")

    print("\n=== VERDICT ===")
    base_cp, _ = transfer(Xc, mc, Xp, mp, ["hb_count", "aromatic_cc"])
    all_cp, _ = transfer(Xc, mc, Xp, mp, ["de_sasa", "tds_conf_esm", "hb_count", "aromatic_cc"])
    s12_cp, _ = transfer(Xc, mc, Xp, mp, ["de_sasa", "tds_conf_esm"])
    print(f"  baseline hb+aromatic cr->pb: {base_cp:+.3f}")
    print(f"  Stage1+2+3 (no geometry):    {s12_cp:+.3f}")
    print(f"  ALL combined:                {all_cp:+.3f}")
    print(f"  >> synthesis beats baseline? {'YES' if all_cp > base_cp + 0.03 else 'NO'}")
    print(f"  >> SASA+entropy+ESM alone beats baseline? {'YES' if s12_cp > base_cp + 0.03 else 'NO'}")


if __name__ == "__main__":
    main()
