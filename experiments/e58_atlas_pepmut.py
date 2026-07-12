"""E58 — ATLAS peptide-mutation ΔΔG: independent validation of the Δphys maturation ranker.

ATLAS (real TCR-pMHC SPR Kd) has 131 PEPTIDE-mutation rows with measured Delta_DeltaG. This is an
INDEPENDENT peptide-side mutation benchmark (different assay, different system from SKEMPI). Tests:
  (1) Does the Δphys sequence prior (Δvol/Δhyd/Δchg/progly) that beat FlexPepDock on SKEMPI generalize?
  (2) ATLAS's OWN finding: their backrub ensemble (r=0.473) was WORSE than single-structure (r=0.630) —
      corroborating e57's premise that backbone ensembles don't add discrimination on interface muts.

No PyRosetta — pure sequence features vs experimental ddG, leave-system-out by PDB template.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ATLAS = Path("/tmp/ATLAS_repo/www/tables/ATLAS.tsv")
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
       "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
       "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
CHG = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}


def parse_pepmut(tok):
    """ 'P4G' -> ('P',4,'G'); handle multi like 'P4G,M5A' (take single only). """
    tok = tok.strip()
    if "," in tok or "/" in tok or len(tok) < 3:
        return None
    wt, mut = tok[0], tok[-1]
    num = tok[1:-1]
    if not num.isdigit() or wt not in KD or mut not in KD:
        return None
    return wt, int(num), mut


def main():
    rows = list(csv.DictReader(open(ATLAS), delimiter="\t"))
    data = []
    for r in rows:
        pm = r.get("PEP_mut", "WT")
        if pm in ("WT", "", "nan"):
            continue
        # ONLY peptide-side mutations (TCR and MHC wild-type) to isolate the peptide signal
        if r.get("TCR_mut") not in ("WT",) or r.get("MHC_mut") not in ("WT", "nan", ""):
            continue
        ddg = r.get("Delta_DeltaG_kcal_per_mol", "\\N")
        if ddg in ("\\N", "", "nan", "n.d."):
            continue
        p = parse_pepmut(pm)
        if p is None:
            continue
        wt, pos, mut = p
        try:
            ddg = float(ddg)
        except ValueError:
            continue
        sys_id = r.get("template_PDB") or r.get("true_PDB") or r.get("MHCname")
        data.append(dict(
            sys=sys_id, mut=pm, ddg_exp=ddg,
            d_hyd=KD[mut] - KD[wt], d_vol=VOL[mut] - VOL[wt],
            d_chg=CHG.get(mut, 0) - CHG.get(wt, 0),
            abs_dchg=abs(CHG.get(mut, 0) - CHG.get(wt, 0)),
            charged=1.0 if (wt in "DEKRH" or mut in "DEKRH") else 0.0,
            progly=1.0 if (mut in "PG" or wt in "PG") else 0.0))
    print(f"=== E58 ATLAS peptide-mutation ΔΔG. n={len(data)} peptide-only muts ===\n")
    if len(data) < 10:
        print("too few")
        return
    e = np.array([d["ddg_exp"] for d in data])

    print("=== single-feature Spearman vs experimental ΔΔG (sign-aware) ===")
    for f in ["d_hyd", "d_vol", "abs_dchg", "d_chg", "charged", "progly"]:
        x = np.array([d[f] for d in data])
        if np.std(x) < 1e-9:
            continue
        print(f"  {f:<10} Spearman={spearmanr(x, e).statistic:+.3f}  Pearson={pearsonr(x, e).statistic:+.3f}")

    # Δphys combined ridge, leave-SYSTEM-out (by template PDB)
    feats = ["d_hyd", "d_vol", "abs_dchg", "progly"]
    syss = sorted({d["sys"] for d in data})
    preds, exps = [], []
    percx = []
    for s in syss:
        tr = [d for d in data if d["sys"] != s]
        te = [d for d in data if d["sys"] == s]
        if len(te) < 4 or len(tr) < 20:
            continue
        X = np.array([[d[f] for f in feats] for d in tr])
        y = np.array([d["ddg_exp"] for d in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        R = 1.0 * np.eye(A.shape[1])
        R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[d[f] for f in feats] for d in te])
        pr = np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w
        ye = [d["ddg_exp"] for d in te]
        preds += list(pr)
        exps += ye
        if len(te) >= 5:
            percx.append(spearmanr(pr, ye).statistic)
    print(f"\n=== Δphys ridge, leave-system-out ===")
    print(f"  pooled    n={len(preds)}  Spearman={spearmanr(preds, exps).statistic:+.3f}")
    if percx:
        print(f"  per-system mean Spearman = {np.nanmean(percx):+.3f}  (n_systems={len(percx)})")
    print("\n  >> SKEMPI Δphys was +0.42 leave-complex-out. Does it transfer to ATLAS pep-muts?")
    print("  >> ATLAS own scoring: single-structure r=0.630 BEAT backrub r=0.473 (corroborates e57).")


if __name__ == "__main__":
    main()
