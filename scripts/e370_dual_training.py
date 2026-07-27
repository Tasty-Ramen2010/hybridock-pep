"""E370 — does DUAL-TRAINING (PDBbind + PPIKB) help? (Ram's question)

Our headline 16-feature model is PDBbind-only; the PPIKB number is a separate PPIKB-trained model. Here we test
pooling, in the ONE feature space both datasets share: 37-dim ProtDCal-3D (`desc` in e180 = `desc3d` in PPIKB).

Rigor: pool both sets, cluster the WHOLE pool at 60% identity, GroupKFold on pooled clusters, but only SCORE the
PDBbind members of each test fold — so a PPIKB near-duplicate of a PDBbind test peptide is held out in the same
cluster (no cross-dataset leakage). Compare PDBbind-only training vs +all-PPIKB vs +Kd/Ki-only-PPIKB.

Run: OMP_NUM_THREADS=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/e370_dual_training.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e330_ours_pdbbind import cluster_by_identity, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(path, feat_key, want_kdki=False):
    out = []
    for l in (ROOT / path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        d = r.get(feat_key)
        if not d or r.get("y") is None or not r.get("seq"):
            continue
        if want_kdki and r.get("aff_type") not in ("Kd", "KD", "Ki"):
            continue
        out.append((r["seq"], float(r["y"]), np.array(d, float)))
    return out


def main():
    pdb = load("data/e180_protdcal3d.jsonl", "desc")
    ppi_all = load("data/ppikb_features.jsonl", "desc3d")
    ppi_clean = load("data/ppikb_features.jsonl", "desc3d", want_kdki=True)
    print(f"PDBbind={len(pdb)}  PPIKB-all={len(ppi_all)}  PPIKB-Kd/Ki={len(ppi_clean)}\n")

    def run(extra, tag):
        items = [(s, y, x, "pdb") for s, y, x in pdb] + [(s, y, x, "ppi") for s, y, x in extra]
        seqs = [it[0] for it in items]
        y = np.array([it[1] for it in items])
        X = np.nan_to_num(np.vstack([it[2] for it in items]))
        src = np.array([it[3] for it in items])
        clu = cluster_by_identity(seqs, 0.60)
        nc = len(set(clu.tolist()))
        pred = np.full(len(items), np.nan)
        gkf = GroupKFold(5)
        for tr, te in gkf.split(X, y, groups=clu):
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                          subsample=0.8, random_state=0).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        pm = src == "pdb"                      # score PDBbind members only
        r, sp, rmse, mae = metrics(y[pm], pred[pm])
        print(f"  {tag:34s} r={r:+.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}   (train pool {len(items)}, {nc} clusters)")

    print("Evaluated on PDBbind held-out clusters (desc3d 37-dim), leakage-free:")
    run([], "PDBbind-only (baseline)")
    run(ppi_clean, "+ PPIKB Kd/Ki-only (clean)")
    run(ppi_all, "+ PPIKB all (incl. IC50/EC50 noise)")
    print("\nRead: if '+PPIKB' beats baseline → dual-training helps. If it drops (esp. the noisy 'all' set),\n"
          "PPIKB's cross-DB label noise hurts — quality over count (consistent with E301).")


if __name__ == "__main__":
    main()
