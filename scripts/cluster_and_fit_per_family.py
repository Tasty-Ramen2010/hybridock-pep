"""Cluster the 240 held-out receptors by sequence and fit a per-family ridge.

family_hint column is empty across the 284-row CSV, so we cluster from the
raw receptor sequence:

  1. Extract the receptor chain sequence from datasets/raw_pdbs/{PDB}.pdb,
     using the receptor_chain column (falling back to the longest non-peptide
     chain).
  2. Compute pairwise Jaccard similarity over 6-mer shingles.
  3. Agglomerative clustering at threshold 0.3 (1 − Jaccard); clusters with
     ≥ 10 members get their own ridge fit, smaller ones fall through to a
     global fallback ridge.
  4. Report per-cluster LOO-CV Pearson r and RMSE; write
     data/calibration_per_family.json with the dispatch table.

This is the §4 path from docs/calibration_strategies.md. It does not change
production behavior — the new JSON is opt-in via --calibration.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw_pdbs"

AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
          "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
          "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

K = 6  # k-mer size for Jaccard


def receptor_seq(pdb_path: Path, receptor_chain_hint: str | None,
                 peptide_seq: str) -> str | None:
    """Return the receptor chain sequence — explicit hint, else longest chain
    that does not contain the peptide sequence as a substring."""
    if not pdb_path.exists():
        return None
    chains: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            res = int(line[22:26].strip())
        except ValueError:
            continue
        chains[line[21]].append((res, line[26], line[17:20].strip()))

    def to_seq(triples: list[tuple[int, str, str]]) -> str:
        seen, last = [], None
        for r, ic, resn in triples:
            key = (r, ic)
            if key == last:
                continue
            last = key
            seen.append(AA3to1.get(resn, "X"))
        return "".join(seen)

    if receptor_chain_hint:
        rc = receptor_chain_hint.strip()
        if rc in chains:
            return to_seq(chains[rc])
    pep = peptide_seq.upper()
    candidates = sorted(
        ((cid, to_seq(t)) for cid, t in chains.items()),
        key=lambda x: -len(x[1]),
    )
    for cid, s in candidates:
        if pep and pep in s and len(s) <= len(pep) + 5:
            continue
        return s
    return candidates[0][1] if candidates else None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> None:
    eval_rows = json.loads((ROOT / "data" / "eval_holdout_calibrations.json").read_text())
    csv_meta = {r["pdb_id"].lower(): r for r in csv.DictReader(
        (ROOT / "data" / "training_complexes_full.csv").open())}

    # Pull receptor sequence for each eval entry
    seqs: dict[str, str] = {}
    for row in eval_rows:
        pdb = row["pdb"]
        meta = csv_meta.get(pdb, {})
        rc = meta.get("receptor_chain") or None
        seq = receptor_seq(RAW / f"{pdb.upper()}.pdb", rc, row.get("pdb", ""))
        if seq and len(seq) >= K:
            seqs[pdb] = seq

    pdbs = sorted(seqs.keys())
    print(f"Receptor sequences extracted: {len(pdbs)} / {len(eval_rows)}")

    # k-mer Jaccard distance matrix
    kmers = {p: {seqs[p][i:i+K] for i in range(len(seqs[p]) - K + 1)} for p in pdbs}
    n = len(pdbs)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = 1.0 - jaccard(kmers[pdbs[i]], kmers[pdbs[j]])
            dist[i, j] = dist[j, i] = d
    cond = squareform(dist, checks=False)

    # Agglomerative clustering; pick a threshold that produces 4–8 useful clusters
    Z = linkage(cond, method="average")
    for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        labels = fcluster(Z, t=thr, criterion="distance")
        sizes = sorted([(int(c), int((labels == c).sum())) for c in np.unique(labels)],
                       key=lambda x: -x[1])
        big = [s for s in sizes if s[1] >= 10]
        print(f"  thr={thr}: {len(sizes)} clusters; ≥10-member clusters: "
              f"{len(big)}  sizes={[s[1] for s in sizes[:8]]}")

    # Use threshold 0.3 by default
    labels = fcluster(Z, t=0.3, criterion="distance")
    label_for = dict(zip(pdbs, labels.tolist()))
    cluster_sizes = defaultdict(int)
    for lbl in labels:
        cluster_sizes[int(lbl)] += 1
    big_clusters = [c for c, n in cluster_sizes.items() if n >= 10]
    print(f"\nUsing threshold 0.3 → {len(big_clusters)} big clusters (≥10 members)")
    for c in big_clusters:
        members = [p for p, lb in label_for.items() if lb == c]
        print(f"  cluster {c}: {len(members)} members; first 5: {members[:5]}")

    # Build feature matrix
    by_pdb = {r["pdb"]: r for r in eval_rows}
    def fit_ridge(rows: list[dict]) -> dict:
        X = np.array([[r["vina"], r["n_contact"], r["s_ss_weighted"]] for r in rows])
        y = np.array([r["dg_exp"] for r in rows])
        model = Ridge(alpha=0.1, positive=False).fit(X, y)
        # LOO
        loo_preds = []
        for i in range(len(rows)):
            mask = np.arange(len(rows)) != i
            m = Ridge(alpha=0.1, positive=False).fit(X[mask], y[mask])
            loo_preds.append(float(m.predict(X[i:i+1])[0]))
        loo = np.array(loo_preds)
        loo_r = float(np.corrcoef(y, loo)[0, 1]) if len(y) >= 3 else float("nan")
        loo_rmse = float(np.sqrt(((y - loo) ** 2).mean()))
        in_pred = model.predict(X)
        in_r = float(np.corrcoef(y, in_pred)[0, 1]) if len(y) >= 3 else float("nan")
        return {
            "w_vina": float(model.coef_[0]),
            "w_contact": float(model.coef_[1]),
            "w_s_ss_weighted": float(model.coef_[2]),
            "intercept": float(model.intercept_),
            "n_complexes": len(rows),
            "in_sample_r": in_r,
            "loo_pearson_r": loo_r,
            "loo_rmse_kcal_mol": loo_rmse,
            "pdbs": [r["pdb"] for r in rows],
        }

    family_fits = {}
    for c in big_clusters:
        members = [by_pdb[p] for p, lb in label_for.items() if lb == c and p in by_pdb]
        if len(members) < 10:
            continue
        fit = fit_ridge(members)
        family_fits[str(c)] = fit
        print(f"\nFamily {c} (n={fit['n_complexes']}):"
              f"  LOO r={fit['loo_pearson_r']:+.3f}  RMSE={fit['loo_rmse_kcal_mol']:.2f}"
              f"  in-sample r={fit['in_sample_r']:+.3f}")
        print(f"    w_vina={fit['w_vina']:+.3f}  w_contact={fit['w_contact']:+.3f}  "
              f"w_s_ss={fit['w_s_ss_weighted']:+.3f}  b={fit['intercept']:+.3f}")

    # Fallback global ridge on everyone not in a big cluster
    fallback_rows = [by_pdb[p] for p in pdbs if p in by_pdb and label_for[p] not in big_clusters]
    if len(fallback_rows) >= 5:
        fallback = fit_ridge(fallback_rows)
        print(f"\nFallback (n={fallback['n_complexes']}):"
              f"  LOO r={fallback['loo_pearson_r']:+.3f}  RMSE={fallback['loo_rmse_kcal_mol']:.2f}")
    else:
        fallback = None

    # Aggregate per-family LOO over the union for an honest cross-cluster number
    union_y, union_pred = [], []
    for c, fit in family_fits.items():
        members = [by_pdb[p] for p in fit["pdbs"]]
        # In-sample fits — not great; use LOO-style by re-fitting on cluster minus one
        X = np.array([[r["vina"], r["n_contact"], r["s_ss_weighted"]] for r in members])
        y = np.array([r["dg_exp"] for r in members])
        for i in range(len(members)):
            mask = np.arange(len(members)) != i
            m = Ridge(alpha=0.1).fit(X[mask], y[mask])
            union_y.append(y[i])
            union_pred.append(float(m.predict(X[i:i+1])[0]))
    if union_y:
        ur = float(np.corrcoef(union_y, union_pred)[0, 1])
        urmse = float(np.sqrt(((np.array(union_y) - np.array(union_pred)) ** 2).mean()))
        print(f"\n=== Per-family LOO over union (n={len(union_y)}) ===")
        print(f"  Pearson r = {ur:+.3f}")
        print(f"  RMSE      = {urmse:.2f} kcal/mol")

    out = {
        "schema_version": 3,
        "model_type": "per_family_ridge",
        "clustering": {
            "method": "kmer_jaccard_agglomerative",
            "k": K,
            "threshold": 0.3,
        },
        "families": family_fits,
        "fallback": fallback,
        "membership_assignments": label_for,
        "feature_order": ["vina_score", "n_contact_residues", "s_ss_weighted"],
    }
    (ROOT / "data" / "calibration_per_family.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/calibration_per_family.json")


if __name__ == "__main__":
    main()
