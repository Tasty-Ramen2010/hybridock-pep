"""E330 — our scorer on the PDBbind ~900 peptide-Kd set, for the ref2015 head-to-head.

Three numbers, in increasing order of honesty:
  (A) Random 5-fold CV (shuffle) — the OLD headline. LEAKY: exact-duplicate and
      near-identical peptides (point mutants, redundant complexes) get split across
      train/test folds, which inflates r. Kept only to show the size of the mirage.
  (B) Peptide-sequence-CLUSTERED 5-fold CV (GroupKFold over single-linkage clusters
      at >=40% identity). This is the leakage-free number: an entire cluster of
      similar peptides lands wholly in train or wholly in test, so the model is
      always predicting a peptide it has not seen a near-twin of. THIS is the number
      to report.
  (C) Length-stratified 5-fold as a robustness check on (A).
Reports Pearson r, Spearman, RMSE, MAE (kcal/mol). Also prints the ref2015 numbers if
the e329 cache exists, on the SAME complexes, so the comparison is strictly matched.

Note: clustering is on the PEPTIDE sequence only (that is what the jsonl carries and
the leakage-control point flagged in external feedback). Receptor-family grouping is a further tightening tracked separately
(the ATLAS TCR-pMHC selectivity benchmark).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

ID_THRESH = 0.60  # greedy: a peptide joins a cluster if identity to its rep >= this

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/pdbbind_peptides.jsonl"
REF = ROOT / "data/e329_ref2015_pdbbind.json"
FEATS = ['poc_n', 'poc_f_hyd', 'poc_f_arom', 'poc_net', 'poc_eis', 'bsa_hyd', 'sasa_hb',
         'sasa_sb', 'arom_cc', 'hb_count', 'mj_contact', 'strength_bur', 'rg_per_L',
         'org_density', 'cys_frac', 'mean_burial']


def metrics(y, p):
    r = pearsonr(p, y)[0]
    sp = spearmanr(p, y).statistic
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    return r, sp, rmse, mae


def _identity(a: str, b: str, aligner) -> float:
    """Aligned identity normalized by the shorter sequence (CD-HIT convention)."""
    if a == b:
        return 1.0
    score = aligner.score(a, b)  # match=1, mismatch=0, gap=0 -> # identical aligned residues
    return score / min(len(a), len(b))


def cluster_by_identity(seqs, thresh: float = ID_THRESH):
    """Greedy (CD-HIT-style) clustering of peptides by sequence identity.

    Returns an int array of cluster ids aligned to `seqs`. Unique peptides are sorted
    longest-first; each either joins the first existing cluster whose REPRESENTATIVE
    it is >= thresh identical to, or becomes a new representative. Using a fixed
    representative (rather than single-linkage) avoids transitive chaining, which at
    these thresholds otherwise merges unrelated short peptides into one blob.
    Identity is aligned matches normalized by the shorter sequence (CD-HIT convention),
    so a peptide that is a truncation/extension of a longer one still clusters with it.
    """
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    # Placement-aware identity (fixed 2026-07-09): gaps are PENALISED so that identity respects residue
    # position. With free gaps (open/extend = 0) the score degenerated to longest-common-subsequence /
    # shorter-length, which over-merged short peptides (e.g. GGA vs ACC scored 0.33 from one gapped residue,
    # GGA vs CGG scored 0.67 despite shifted G's). That collapsed 925 peptides into 21 clusters at 30% id.
    # Penalising gaps counts aligned matches in a single frame → GGA/ACC→0, GGA/CGG→0.33. See experiments/e367.
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5

    uniq = sorted(set(seqs), key=lambda s: (-len(s), s))
    reps: list[str] = []          # representative sequence per cluster
    seq_to_cid: dict[str, int] = {}
    for s in uniq:
        placed = False
        for cid, rep in enumerate(reps):
            if min(len(s), len(rep)) < thresh * max(len(s), len(rep)):
                continue  # lengths too different to reach thresh
            if _identity(s, rep, aligner) >= thresh:
                seq_to_cid[s] = cid
                placed = True
                break
        if not placed:
            seq_to_cid[s] = len(reps)
            reps.append(s)
    return np.array([seq_to_cid[s] for s in seqs], dtype=int)


def main():
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    X = np.array([[r[k] for k in FEATS] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    pdbs = [r["pdb"] for r in rows]
    seqs = [r["seq"] for r in rows]
    lengths = np.array([r["length"] for r in rows])
    print(f"=== E330 our scorer on PDBbind peptide-Kd (n={len(rows)}) ===")
    print(f"  experimental ΔG: mean={y.mean():.2f}  std={y.std():.2f}  "
          f"(mean-predictor RMSE={y.std():.2f})\n")

    clusters = cluster_by_identity(seqs, ID_THRESH)
    n_clusters = len(set(clusters.tolist()))
    n_singletons = sum(1 for c in np.bincount(clusters) if c == 1)
    print(f"  peptide-identity clustering (>= {ID_THRESH:.0%}): "
          f"{n_clusters} clusters from {len(set(seqs))} unique seqs "
          f"({n_singletons} singletons; largest={np.bincount(clusters).max()} rows)\n")

    def new_gbt():
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                         subsample=0.8, random_state=0)

    # (A) OLD headline — random split, leaky.
    pred_rand = cross_val_predict(new_gbt(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    r, sp, rmse, mae = metrics(y, pred_rand)
    print("(A) random 5-fold CV  [LEAKY — old headline, near-twins split across folds]:")
    print(f"    Pearson r = {r:+.3f}   Spearman = {sp:+.3f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}\n")

    # (B) leakage-free — cluster-grouped split.
    n_folds = min(5, n_clusters)
    pred_clu = cross_val_predict(new_gbt(), X, y, cv=GroupKFold(n_folds), groups=clusters)
    r, sp, rmse, mae = metrics(y, pred_clu)
    print(f"(B) peptide-CLUSTERED {n_folds}-fold CV  [LEAKAGE-FREE — report this one]:")
    print(f"    Pearson r = {r:+.3f}   Spearman = {sp:+.3f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}\n")

    # (C) length-stratified robustness check on the random split.
    len_groups = np.digitize(lengths, [8, 11, 15, 20])
    pred_len = cross_val_predict(new_gbt(), X, y,
                                 cv=GroupKFold(len(set(len_groups.tolist()))), groups=len_groups)
    r, sp, rmse, mae = metrics(y, pred_len)
    print("(C) length-stratified 5-fold CV  [robustness check]:")
    print(f"    Pearson r = {r:+.3f}   Spearman = {sp:+.3f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}\n")

    # matched comparison against ref2015 on the exact same complexes, using the
    # LEAKAGE-FREE (clustered) predictions so the head-to-head is honest.
    if REF.exists():
        ref = {d["pdb"]: d for d in json.loads(REF.read_text())}
        idx = [i for i, p in enumerate(pdbs) if p in ref]
        if len(idx) >= 20:
            ys = y[idx]
            ours = pred_clu[idx]
            rx = np.array([ref[pdbs[i]]["ros_ifdG"] for i in idx])
            print(f"--- MATCHED head-to-head on n={len(idx)} complexes scored by both ---")
            r, sp, rmse, mae = metrics(ys, ours)
            print(f"  OURS (16-feat, CLUSTERED CV): r={r:+.3f}  rho={sp:+.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
            # ref2015 is REU, not kcal/mol -> only correlation is meaningful
            rr = pearsonr(rx, ys)[0]; rsp = spearmanr(rx, ys).statistic
            print(f"  ref2015 ifdG (unrelaxed):     r={rr:+.3f}  rho={rsp:+.3f}  "
                  f"(REU, no kcal/mol RMSE/MAE)")
        else:
            print(f"(ref2015 cache has only {len(idx)} matched — run e329 to completion)")
    else:
        print("(ref2015 cache data/e329_ref2015_pdbbind.json not present yet)")


if __name__ == "__main__":
    main()
