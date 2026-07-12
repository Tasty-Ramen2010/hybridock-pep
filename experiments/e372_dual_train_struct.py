"""E372 — dual-training in the HEADLINE 16-STRUCT feature space: does PDBbind+PPIKB beat PDBbind-only?

Pools data/pdbbind_peptides.jsonl (16 STRUCT) with data/ppikb_struct_features.jsonl (same 16, from e371).
Leakage-free: cluster the WHOLE pool at 60% identity, GroupKFold, score only PDBbind members (so a PPIKB
near-duplicate of a PDBbind test peptide is held out in the same cluster). Reports MAE/RMSE/r vs baseline, and
re-scores the Wang external-43 holdout under each training set. The question: does dual-training beat MAE 1.40?

RESULT (2026-07-10): NO on MAE. PDBbind-only 1.40 -> +PPIKB 1.41 (in-dist), 1.64 -> 1.66 (Wang external n=40).
Only movement is external ranking r +0.056 (0.460 -> 0.516). PPIKB's noisier labels (IC50/EC50 mix, cross-source
Kd disagreement up to 10.8 kcal) dilute absolute-error calibration even as diversity slightly helps rank order.
Verdict: keep the PDBbind-only headline model; dual-training is not worth the MAE cost. Confirms E301 in the
strong STRUCT feature space, not just desc3d.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python experiments/e372_dual_train_struct.py
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import FEATS, cluster_by_identity, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GB = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)


def load_pdb():
    rows = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()]
    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows], float))
    y = np.array([float(r["y"]) for r in rows]); seq = [r["seq"] for r in rows]
    return X, y, seq


def load_ppikb():
    rows = [json.loads(l) for l in (ROOT / "data/ppikb_struct_features.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "poc_n" in r and r.get("y") is not None and r.get("seq")]
    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATS] for r in rows], float))
    y = np.array([float(r["y"]) for r in rows]); seq = [r["seq"] for r in rows]
    return X, y, seq


def wang43():
    rows = list(csv.DictReader(open(ROOT / "data/hybridock_wang2024_external43.csv")))
    # need STRUCT feats for these; pull from pdbbind jsonl by pdb if present, else skip
    feat = {json.loads(l)["pdb"].upper(): json.loads(l)
            for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines() if l.strip()}
    keep = [r for r in rows if r["pdb"].upper() in feat]
    if not keep:
        return None
    X = np.nan_to_num(np.array([[float(feat[r["pdb"].upper()][k]) for k in FEATS] for r in keep], float))
    y = np.array([float(r["exp"]) for r in keep])
    return X, y


def main():
    Xp, yp, sp = load_pdb()
    Xk, yk, sk = load_ppikb()
    print(f"PDBbind STRUCT={len(yp)}   PPIKB STRUCT (e371)={len(yk)}\n")

    def pooled_cv(extra_X, extra_y, extra_seq, tag):
        X = np.vstack([Xp] + ([extra_X] if len(extra_y) else []))
        y = np.concatenate([yp] + ([extra_y] if len(extra_y) else []))
        seq = sp + (extra_seq if len(extra_y) else [])
        src = np.array(["pdb"] * len(yp) + ["ppi"] * len(extra_y))
        clu = cluster_by_identity(seq, 0.60)
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups=clu):
            pred[te] = GB().fit(X[tr], y[tr]).predict(X[te])
        m = src == "pdb"
        r, spm, rmse, mae = metrics(y[m], pred[m])
        star = "  ◀ beats 1.40" if mae < 1.40 else ""
        print(f"  {tag:32s} r={r:+.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}{star}")
        return mae

    print("=== HEADLINE 16-STRUCT space, evaluated on PDBbind held-out clusters (leakage-free) ===")
    base = pooled_cv(np.empty((0, len(FEATS))), np.array([]), [], "PDBbind-only (baseline)")
    dual = pooled_cv(Xk, yk, sk, "+ PPIKB STRUCT (dual-train)")
    print(f"\n  Δ MAE (dual − base) = {dual - base:+.3f} kcal/mol  →  {'DUAL WINS' if dual < base else 'no improvement'}")

    # External-43: train on each set, predict Wang holdout
    w = wang43()
    if w:
        Xw, yw = w
        print(f"\n=== Wang external-43 holdout (n={len(yw)}), trained on each set ===")
        for tag, X, y in [("PDBbind-only", Xp, yp), ("PDBbind+PPIKB", np.vstack([Xp, Xk]), np.concatenate([yp, yk]))]:
            p = GB().fit(X, y).predict(Xw)
            r, spm, rmse, mae = metrics(yw, p)
            print(f"  {tag:16s} MAE={mae:.2f}  RMSE={rmse:.2f}  r={r:+.3f}")
    else:
        print("\n(Wang-43 STRUCT feats not found in pdbbind jsonl — external check skipped)")


if __name__ == "__main__":
    main()
