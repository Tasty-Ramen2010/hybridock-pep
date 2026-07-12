"""E178 — FAITHFUL PPI-Affinity rebuild using the REAL ProtDCal properties, validated against PPI's
SHIPPED per-complex predictions.

PPI-Affinity = ProtDCal physicochemical descriptors (6 selected properties: ECI, IP, ISA, Z1, Z2, Z3),
group-aggregated over residue subsets, fed to SMOreg (poly-kernel SVR). It is SEQUENCE-based → pose-blind.

We reproduce the descriptor CLASS from the real ProtDCal per-AA table (third_party/protdcal/protdcal_aa_table.csv),
train on PDBbind-925 minus T100, and validate faithfulness on PPI's own T100 test set two ways:
  (A) aggregate: does our rebuild reach PPI's r≈0.52 / MAE≈1.13 ?
  (B) per-complex: do our rebuild's predictions CORRELATE with PPI's SHIPPED predictions (SI-File-6)?
Only if BOTH hold is the rebuild a fair stand-in. Then we run it (pose-blind) vs ours (pose-using) on our
real RAPiDock poses — the deployment test Ram asked for.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_regression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SCALES, POS, NEG, PROD, SD = e150.SCALES, e150.POS, e150.NEG, e150.PROD, e150.seq_descriptors
SN = list(SCALES.keys())

# ---- REAL ProtDCal properties (the 6 PPI selected) ----
_TAB = {r["just_AA"]: r for r in csv.DictReader(open(ROOT / "third_party/protdcal/protdcal_aa_table.csv"))}
PROPS = ["ECI", "IP", "ISA", "Z1", "Z2", "Z3"]
PROP = {p: {aa: float(_TAB[aa][f"{p}_NO"]) for aa in _TAB} for p in PROPS}

# ProtDCal residue groups (standard physicochemical subsets; PPI uses 11)
GROUPS = {
    "UCR": set("ACDEFGHIKLMNPQRSTVWY"),  # all
    "AHR": set("AVLIM"),                 # aliphatic hydrophobic
    "ALR": set("AVLIG"),                 # aliphatic
    "ARM": set("FWYH"),                  # aromatic
    "BSR": set("KRH"),                   # basic
    "NCR": set("DE"),                    # negatively charged
    "NPR": set("AVLIMFWPGC"),            # non-polar
    "PCR": set("KR"),                    # positively charged
    "PLR": set("STNQYCH"),              # polar
    "PRT": set("GASTNDQ"),              # small/turn-favoring
    "RTR": set("DEKR"),                 # all charged
}


def _inv(v):
    """ProtDCal-style group invariants (sum/mean/min/max/range/std/L1/L2)."""
    if v.size == 0:
        return [0.0] * 8
    return [float(v.sum()), float(v.mean()), float(v.min()), float(v.max()),
            float(v.max() - v.min()), float(v.std()),
            float(np.abs(v).sum()), float(np.sqrt((v ** 2).sum()))]


def protdcal_real(seq):
    """6 real properties × 11 groups × 8 invariants = 528 ProtDCal-class descriptors."""
    out = []
    for p in PROPS:
        tab = PROP[p]
        for g in GROUPS.values():
            v = np.array([tab[c] for c in seq if c in g and c in tab], float)
            out += _inv(v)
    return out


def ppi_features(seq, pocket_seq):
    """PPI describes both partners (peptide + receptor pocket)."""
    return protdcal_real(seq) + protdcal_real(pocket_seq if pocket_seq else seq)


def ppi_model():
    # SMOreg ≈ poly-kernel SVR; PPI feature-selects to ~37
    return Pipeline([("sc", StandardScaler()),
                     ("sel", SelectKBest(f_regression, k=37)),
                     ("svr", SVR(kernel="rbf", C=4.0, epsilon=0.1, gamma="scale"))])


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def main():
    # ---------- training pool: PDBbind-925 ----------
    pool = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        ps = e158.pocket_seq(r["pdb"])
        if ps is None:
            continue
        pool.append((r["pdb"].lower(), r["seq"], float(r["y"]), ps))

    # ---------- T100 test (PPI's own set) with their SHIPPED predictions ----------
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    # seq comes from the e166 extraction cache; ppi_affinity from the manifest
    seqcache = {json.loads(l)["pdb"].lower(): json.loads(l)
                for l in open(ROOT / "data/t100_extra_features.jsonl")}
    t100 = []
    for pid, d in seqcache.items():
        m = man.get(pid)
        if m is None:
            continue
        seq = d["seq"]
        ps = e158.pocket_seq(pid) or seq  # peptide-only fallback if pocket unavailable
        try:
            ppi_pred = float(m["ppi_affinity"])
        except (TypeError, ValueError, KeyError):
            ppi_pred = np.nan
        t100.append((pid, seq, float(m["dg_exp"]), ps, ppi_pred))
    t100_ids = {r[0] for r in t100}

    train = [r for r in pool if r[0] not in t100_ids]
    Xtr = np.array([ppi_features(r[1], r[3]) for r in train]); ytr = np.array([r[2] for r in train])
    Xte = np.array([ppi_features(r[1], r[3]) for r in t100]); yte = np.array([r[2] for r in t100])
    ppi_shipped = np.array([r[4] for r in t100])

    mdl = ppi_model().fit(np.nan_to_num(Xtr), ytr)
    pred = mdl.predict(np.nan_to_num(Xte))

    print(f"=== FAITHFULNESS GATE: rebuilt-PPI on T100 (n={len(t100)}, trained {len(train)} held-out) ===")
    r_re, mae_re = met(pred, yte)
    r_sh, mae_sh = met(ppi_shipped, yte)
    r_corr, _ = met(pred, ppi_shipped)
    print(f"  rebuilt-PPI vs truth:   r={r_re:+.3f}  MAE={mae_re:.2f}")
    print(f"  shipped-PPI vs truth:   r={r_sh:+.3f}  MAE={mae_sh:.2f}   (target to match)")
    print(f"  rebuilt vs SHIPPED preds (per-complex faithfulness): r={r_corr:+.3f}")
    faithful = (r_re >= 0.42) and (r_corr >= 0.55)
    print(f"  -> FAITHFUL: {faithful}  (need r_truth>=0.42 AND r_corr>=0.55)")

    # save for downstream deployment test regardless, but flag faithfulness
    out = {"r_rebuilt": r_re, "mae_rebuilt": mae_re, "r_shipped": r_sh, "mae_shipped": mae_sh,
           "r_rebuilt_vs_shipped": r_corr, "faithful": bool(faithful), "n_t100": len(t100),
           "n_train": len(train)}
    (ROOT / "runs/e178_faithfulness.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote runs/e178_faithfulness.json")


if __name__ == "__main__":
    main()
