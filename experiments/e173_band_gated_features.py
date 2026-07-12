"""E173 — band-gated features + noise/sign-flip removal for short & vlong. Per-band analysis (E172-prep)
showed short and vlong have DIFFERENT, partly SIGN-FLIPPING physics (bulk: short +0.11 / vlong −0.32;
max_burial: short −0.34 / vlong ~0; charge: vlong only). Test whether explicit length-GATED interaction
features (F × 1[length∈band]) + removing the raw sign-flippers improves the bands, vs a base model.

Rigorous: clustered (receptor) CV, per-band r (short ≤8, vlong ≥17), permutation control on every gain.
Crystal-925 (populated bands: short 305, vlong 53) — the signal test; deployment is data-starved separately.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e158_overfit_failure_analysis as e158  # noqa: E402
e150 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py"))
importlib.util.spec_from_file_location("e150", ROOT / "experiments/e150_protdcal_descriptors.py").loader.exec_module(e150)
SCALES, POS, NEG, PROD, SD = e150.SCALES, e150.POS, e150.NEG, e150.PROD, e150.seq_descriptors
SN = list(SCALES.keys())

# physics features and which carry band-specific / flipping signal (from per-band r table)
SHORT_SIGNAL = ["max_burial", "buried_inert", "pro_run", "hb_count", "sasa_hb", "mean_burial"]
VLONG_SIGNAL = ["mean_burial", "bulk", "mj_contact", "net_charge", "abs_charge", "poc_net", "sasa_sb", "aromatic"]
FLIPPERS = ["bulk", "poc_net"]   # raw versions confuse the pooled model


def derived(seq):
    pq = sum(c in POS for c in seq) - sum(c in NEG for c in seq)
    return {"net_charge": float(pq), "abs_charge": float(abs(pq)),
            "hydropathy": float(np.mean([SCALES["kd"].get(c, 0) for c in seq])),
            "aromatic": float(np.mean([SCALES["arom"].get(c, 0) for c in seq])),
            "bulk": float(np.mean([SCALES["bulk"].get(c, 0) for c in seq]))}


def load():
    anchor = {json.loads(l)["pdb"]: json.loads(l) for l in open(ROOT / "data/anchor_925.jsonl")}
    ss = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/ss_features.jsonl")}
    rows = []
    for r in (json.loads(l) for l in open(ROOT / "data/pdbbind_peptides.jsonl")):
        a = anchor.get(r["pdb"])
        if a is None:
            continue
        s = ss.get(r["pdb"].lower())
        f = {k: float(r.get(k, 0)) for k in PROD}
        f.update({k: float(a[k]) for k in ("max_burial", "buried_inert", "pro_run")})
        f.update(derived(r["seq"]))
        f["helix"] = float(s["helix"]) if s else 0.0
        f["sheet"] = float(s["sheet"]) if s else 0.0
        f["length"] = float(len(r["seq"]))
        rows.append((f, r["y"], len(r["seq"]), r["pdb"], r["seq"]))
    return rows


PHYS = list(PROD) + ["max_burial", "buried_inert", "pro_run", "net_charge", "abs_charge",
                     "hydropathy", "aromatic", "bulk", "helix", "sheet", "length"]


def base_vec(f, drop_flippers=False):
    keys = [k for k in PHYS if not (drop_flippers and k in FLIPPERS)]
    return [f[k] for k in keys]


def gated(f, L):
    """band-gated interactions: signal feature × hard band indicator."""
    sg = 1.0 if L <= 8 else 0.0
    vg = 1.0 if L >= 17 else 0.0
    mg = 1.0 if 9 <= L <= 16 else 0.0
    out = []
    for k in SHORT_SIGNAL:
        out.append(f[k] * sg)
    for k in VLONG_SIGNAL:
        out.append(f[k] * vg)
    # the flippers get a med-gated copy too so all three regimes have own sign
    for k in FLIPPERS:
        out.append(f[k] * mg)
    return out


def main():
    rows = load()
    y = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
    rl, _ = e158.greedy_cluster([e158.pocket_seq(r[3]) for r in rows], 0.7)
    short, vlong = L <= 8, L >= 17
    print(f"=== E173 band-gated features (crystal-925, short n={short.sum()}, vlong n={vlong.sum()}) ===\n")

    def cv(build, seed=0):
        X = np.nan_to_num([build(r[0], r[2]) for r in rows]); pred = np.full(len(rows), np.nan)
        yy = y if seed == 0 else np.random.default_rng(seed).permutation(y)
        for tr, te in GroupKFold(5).split(X, yy, rl):
            m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                              l2_regularization=3.0, min_samples_leaf=12, random_state=0).fit(X[tr], yy[tr])
            pred[te] = m.predict(X[te])
        return pred, yy

    def report(name, build):
        p, _ = cv(build)
        R = lambda m=None: float(np.corrcoef(p[m] if m is not None else p, y[m] if m is not None else y)[0, 1])  # noqa
        pp, yp = cv(build, seed=7)
        Rp = float(np.corrcoef(pp, yp)[0, 1])
        print(f"  {name:34s} overall={R():+.3f}  short={R(short):+.3f}  vlong={R(vlong):+.3f}  (perm={Rp:+.3f})")
        return R(), R(short), R(vlong)

    print("  config                              overall    short    vlong    perm-control")
    report("A base (raw physics)", lambda f, L: base_vec(f))
    report("B base + band-gated", lambda f, L: base_vec(f) + gated(f, L))
    report("C drop raw flippers", lambda f, L: base_vec(f, drop_flippers=True))
    report("D drop flippers + band-gated", lambda f, L: base_vec(f, drop_flippers=True) + gated(f, L))
    report("E ONLY band-gated + length", lambda f, L: [f["length"]] + gated(f, L))


if __name__ == "__main__":
    main()
