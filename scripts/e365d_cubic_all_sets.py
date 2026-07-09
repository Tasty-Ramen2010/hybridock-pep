"""E365d — apply cubic recalibration to ALL our sets and compare honest vs leaky.

For each set we already have leave-cluster-out (OOF) predictions vs experimental ΔG. We add a cubic map pred->exp
and report it THREE ways so the leakage is visible:
  raw            = no recalibration (current reported number)
  cubic (CV)     = cubic fit out-of-fold (5-fold; the map never sees the point it scores)  <- honest
  cubic (LOO)    = leave-one-out cubic (honest, best for small sets)
  cubic (insamp) = cubic fit on the same points it scores  <- LEAKY, optimistic, do not report

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e365d_cubic_all_sets.py
"""
from __future__ import annotations
import csv, os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
MAE = lambda a, b: float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())
RMSE = lambda a, b: float(np.sqrt(((np.asarray(a, float) - np.asarray(b, float)) ** 2).mean()))


def cubic_insample(pred, exp):
    c = np.polyfit(pred, exp, 3)
    return np.polyval(c, pred)


def cubic_cv(pred, exp, k=5):
    out = np.empty_like(pred, dtype=float)
    for tr, te in KFold(k, shuffle=True, random_state=0).split(pred):
        c = np.polyfit(pred[tr], exp[tr], 3)
        lo, hi = pred[tr].min(), pred[tr].max()
        out[te] = np.polyval(c, np.clip(pred[te], lo, hi))
    return out


def cubic_loo(pred, exp):
    out = np.empty_like(pred, dtype=float)
    for i in range(len(pred)):
        xtr = np.delete(pred, i); ytr = np.delete(exp, i)
        c = np.polyfit(xtr, ytr, 3)
        out[i] = np.polyval(c, np.clip(pred[i], xtr.min(), xtr.max()))
    return out


def load(csv_path, exp_col, pred_col, drop_bungarotoxin=False):
    exp, pred = [], []
    for r in csv.DictReader(open(ROOT / csv_path)):
        if drop_bungarotoxin and "bungarotoxin" in r.get("protein", "").lower():
            continue
        exp.append(float(r[exp_col])); pred.append(float(r[pred_col]))
    return np.array(pred), np.array(exp)


def report(tag, pred, exp):
    r = pearsonr(pred, exp)[0]
    raw = MAE(pred, exp)
    cv = MAE(cubic_cv(pred, exp), exp) if len(pred) >= 15 else float("nan")
    loo = MAE(cubic_loo(pred, exp), exp)
    ins = MAE(cubic_insample(pred, exp), exp)
    print(f"{tag:32s} n={len(pred):4d} r={r:+.2f} | raw {raw:5.2f} | cubic-CV {cv:5.2f} | "
          f"cubic-LOO {loo:5.2f} | cubic-INSAMPLE(leaky) {ins:5.2f}")


def main():
    print("set                               n     r   | raw   | cubic-CV | cubic-LOO | cubic-INSAMPLE(leaky)")
    print("-" * 108)
    # 925 PDBbind (STRUCT model)
    p, e = load("data/hybridock_blind_925.csv", "exp_dG_kcal_mol", "pred_dG_kcal_mol")
    report("PDBbind-925 (blind)", p, e)
    # 155 Wang overlap
    p, e = load("data/hybridock_wang2024_complexes.csv", "exp_dG_kcal_mol", "hybridock_blind_pred_dG_kcal_mol")
    report("Wang overlap-155 (blind)", p, e)
    # external 47 and external minus bungarotoxin (43)
    p, e = load("data/hybridock_wang2024_external_regular.csv", "exp", "pred")
    report("Wang external-47", p, e)
    p, e = load("data/hybridock_wang2024_external_regular.csv", "exp", "pred", drop_bungarotoxin=True)
    report("Wang external-43 (no bungarotoxin)", p, e)
    print("\nHonest columns = cubic-CV / cubic-LOO. cubic-INSAMPLE is fit-on-itself and inflated; not reportable.")


if __name__ == "__main__":
    main()
