"""Build the deployment anchor library: receptor -> known-Kd reference peptides (PDBbind + PPIKB).

Each entry stores the peptide, receptor sequence, experimental ΔG, a peptide feature vector (for
similarity weighting), and the absolute scorer output. Consumed by hybridock_pep.scoring.anchoring at
deploy time. Only receptors with >=2 distinct peptides are useful for anchoring, but singletons are kept
so a user-supplied reference on that exact receptor can match.

Output: data/anchor_library.json  (list of {receptor, peptide, dg_exp, features, score}).
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATS = ["arom_cc", "bsa_hyd", "cys_frac", "hb_count", "length", "mean_burial", "mj_contact",
         "org_density", "poc_eis", "poc_f_arom", "poc_f_hyd", "poc_n", "poc_net", "rg_per_L",
         "sasa_hb", "sasa_sb", "strength_bur"]


def _pf(v):
    if isinstance(v, str):
        v = v.strip()
        return json.loads(v) if v.startswith("[") else float(v)
    return v


def main() -> None:
    # PPIKB structured rows carry receptor seq + 3D descriptors + pocket features.
    ppikb = []
    for r in (json.loads(l) for l in open(os.path.join(ROOT, "data/ppikb_features.jsonl"))):
        if r.get("aff_type") not in ("Kd", "Ki", "KD") or not r.get("desc3d"):
            continue
        try:
            d3 = _pf(r["desc3d"]); pk = _pf(r["pocket_pkf"]); y = _pf(r["y"])
        except Exception:  # noqa: BLE001
            continue
        if not (isinstance(d3, list) and isinstance(pk, list) and np.isfinite(y)):
            continue
        ppikb.append({"receptor": r["protein_seq"], "peptide": r["seq"], "dg_exp": float(y),
                      "feat": d3 + pk + [_pf(r["length"]), _pf(r["net_charge"])]})
    dim = max(len(e["feat"]) for e in ppikb)
    ppikb = [e for e in ppikb if len(e["feat"]) == dim]

    X = np.array([e["feat"] for e in ppikb])
    y = np.array([e["dg_exp"] for e in ppikb])
    grp = np.array([hash(e["receptor"]) % (10**9) for e in ppikb])
    # OOF absolute score so the stored S(ref) is not optimistically in-sample
    S = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(X, y, grp):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr])
        S[te] = m.predict(X[te])

    library = []
    for e, s in zip(ppikb, S):
        if not np.isfinite(s):
            continue
        library.append({"receptor": e["receptor"], "peptide": e["peptide"],
                        "dg_exp": e["dg_exp"], "features": [float(v) for v in e["feat"]],
                        "score": float(s)})
    out = os.path.join(ROOT, "data/anchor_library.json")
    json.dump(library, open(out, "w"))
    from collections import Counter
    c = Counter(e["receptor"] for e in library)
    anchorable = sum(1 for v in c.values() if v >= 2)
    print(f"anchor library: {len(library)} references | {len(c)} receptors | "
          f"{anchorable} anchorable (>=2 peptides) | feature dim {dim}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
