"""E146 — is the charged floor LEARNABLE from data+descriptors (PPI-Affinity gets 0.71 on high-charge
WITHOUT FEP)? Test a charged-specialist trained on the charged subset with DESCRIPTOR features (the kind
PPI-Affinity uses: composition, charge patterns) ON TOP of our 16 physics features.

Our physics-electrostatics features WASH (Coulomb+Born cancel) → we concluded "charged = FEP-only". But
PPI-Affinity's ProtDCal+SVM hits 0.71 on high-charge from DATA, no explicit electrostatics. So the signal
is in sequence/structure DESCRIPTORS, not physics. Test: charged subset of PDBbind-925, add descriptor
features, grouped CV, compare base-16 vs +descriptors. If +descriptors lifts charged → the floor is
FEATURE-driven not FEP-fundamental → Ram is right, train a charged specialist.

Descriptor features (data-driven, no physics electrostatics):
  net_charge, abs_net_charge, frac_pos, frac_neg, frac_charged, charge_run (longest run same sign),
  charge_sep (|mean pos pos − mean neg pos| along seq), n_KR, n_DE, aa composition (20), len
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
AA = "ACDEFGHIKLMNPQRSTVWY"
POS, NEG = set("KR"), set("DE")


def desc(seq):
    L = max(1, len(seq))
    npos = sum(c in POS for c in seq); nneg = sum(c in NEG for c in seq)
    # longest run of same-sign charged residues
    run = best = 0; last = 0
    for c in seq:
        s = 1 if c in POS else (-1 if c in NEG else 0)
        if s != 0 and s == last:
            run += 1
        elif s != 0:
            run = 1
        else:
            run = 0
        last = s; best = max(best, run)
    pos_idx = [i for i, c in enumerate(seq) if c in POS]
    neg_idx = [i for i, c in enumerate(seq) if c in NEG]
    sep = abs((np.mean(pos_idx) if pos_idx else 0) - (np.mean(neg_idx) if neg_idx else 0)) / L
    comp = [seq.count(a) / L for a in AA]
    return [npos - nneg, abs(npos - nneg), npos / L, nneg / L, (npos + nneg) / L, best / L, sep,
            float(npos), float(nneg), float(L)] + comp


DKEYS = ["netq", "absq", "fpos", "fneg", "fchg", "chgrun", "chgsep", "nKR", "nDE", "len"] + list(AA)


def cvr(rows, add_desc, k=5):
    rng = np.random.default_rng(0)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    X = np.array([r["feat"] + (r["desc"] if add_desc else []) for r in rows], float)
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.05,
                                          l2_regularization=2.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred, y


def metrics(p, y):
    return pearsonr(p, y)[0], np.mean(np.abs(p - y)), np.sqrt(np.mean((p - y) ** 2))


def main():
    pdbb = [json.loads(l) for l in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    rows = []
    for r in pdbb:
        q = abs(sum(c in POS for c in r["seq"]) - sum(c in NEG for c in r["seq"]))
        rows.append({"y": r["y"], "absq": q, "length": r["length"],
                     "feat": [r[c] for c in PROD], "desc": desc(r["seq"])})
    allc = rows
    charged = [r for r in rows if r["absq"] >= 2]
    highchg = [r for r in rows if r["absq"] >= 3]
    lowchg = [r for r in rows if r["absq"] <= 1]
    print(f"=== E146 charged-specialist: is the charged floor LEARNABLE from descriptors? ===")
    print(f"  PDBbind-925: charged|q|≥2 n={len(charged)}, high|q|≥3 n={len(highchg)}, low|q|≤1 n={len(lowchg)}\n")
    print(f"  {'subset':<16}{'model':<18}{'r':>8}{'MAE':>7}{'RMSE':>7}")
    for name, sub in [("ALL", allc), ("low |q|≤1", lowchg), ("charged |q|≥2", charged), ("high |q|≥3", highchg)]:
        if len(sub) < 25:
            print(f"  {name:<16}(n={len(sub)} too small)"); continue
        for lbl, add in [("base-16", False), ("+descriptors", True)]:
            r, mae, rmse = metrics(*cvr(sub, add))
            print(f"  {name:<16}{lbl:<18}{r:>+8.3f}{mae:>7.2f}{rmse:>7.2f}")
        print()
    print("  reading: if +descriptors lifts charged/high-charge meaningfully (toward PPI's 0.71) → the")
    print("  charged floor is FEATURE-driven (learnable from data, NOT FEP-fundamental) → train a charged")
    print("  specialist + add descriptor features to production. Flat → it really is FEP-only.")


if __name__ == "__main__":
    main()
