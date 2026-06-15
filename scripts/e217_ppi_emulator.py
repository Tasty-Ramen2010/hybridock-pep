"""E217 — build a PPI-Affinity EMULATOR by DISTILLING PPI's actual shipped predictions (not fitting to truth).

Prior clones (E178-182) fit descriptors→TRUTH and only correlated 0.33 with PPI's real predictions. This
fits descriptors→PPI's SHIPPED PREDICTIONS = learns PPI's actual input→output function. If the emulator
reproduces PPI's predictions with high fidelity (LOO-CV corr vs PPI preds), we can run it on the FRESH PPIKB
set to get real-PPI-EQUIVALENT predictions → a DIRECT head-to-head (no ratio-scale extrapolation).

Features = the faithful 37 ProtDCal-3D contact descriptors (PPI's .idl class) + pocket + peptide ProtDCal.
Step 1: T100 LOO-CV faithfulness (corr emulator vs PPI shipped). Step 2: best emulator → fresh PPIKB →
ours vs PPI-emulator per band.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hybridock_pep.scoring.affinity_model import _protdcal_descriptors, _SCALES  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402
import e179_protdcal_3d as e179  # noqa: E402
import e202_band_routing_build as e202  # noqa: E402
SN = list(_SCALES.keys())


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else float("nan")


def pkf(ps):
    return [float(np.mean([_SCALES[s].get(c, 0) for c in ps])) for s in SN] if ps else [0.0] * len(SN)


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqc = {json.loads(l)["pdb"].lower(): json.loads(l) for l in open(ROOT / "data/t100_extra_features.jsonl")}
    # T100 features + PPI shipped predictions (the distillation target)
    rows = []
    for pid, m in man.items():
        d = seqc.get(pid)
        if d is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        res = e179.residue_seq_and_coords(pep) if pep else None
        if res is None:
            continue
        try:
            ppi = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            continue
        ps = e158.pocket_seq(pid) or ""
        feat = list(e179.descriptors(res, 6.0, 3)) + pkf(ps) + _protdcal_descriptors(d["seq"])
        rows.append({"f": feat, "ppi": ppi, "y": float(m["dg_exp"]), "seq": d["seq"], "L": len(d["seq"])})
    X = np.nan_to_num([r["f"] for r in rows]); ppi = np.array([r["ppi"] for r in rows]); y = np.array([r["y"] for r in rows])
    print(f"=== STEP 1: distill PPI's predictions (n={len(rows)}), LOO-CV FAITHFULNESS (corr emulator vs PPI) ===")
    models = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
        "SVR-rbf": Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=8.0, gamma="scale"))]),
        "KRR-rbf": Pipeline([("sc", StandardScaler()), ("m", KernelRidge(kernel="rbf", alpha=0.5, gamma=0.01))]),
        "GBT": HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=2.0, random_state=0),
    }
    best = None
    for nm, mdl in models.items():
        pred = np.full(len(rows), np.nan)
        for tr, te in LeaveOneOut().split(X):
            pred[te] = mdl.fit(X[tr], ppi[tr]).predict(X[te])
        fdl = corr(pred, ppi)        # faithfulness: reproduce PPI's predictions
        tru = corr(pred, y)          # the emulator's own accuracy vs truth
        print(f"  {nm:<9} faithfulness(vs PPI preds)={fdl:+.3f}   accuracy(vs truth)={tru:+.3f}")
        if best is None or fdl > best[0]:
            best = (fdl, nm, mdl)
    print(f"  → best emulator: {best[1]} faithfulness={best[0]:+.3f}  (clone-to-truth E181 was 0.33)")

    # STEP 2: final emulator on all T100 → apply to fresh PPIKB → direct head-to-head
    emu = best[2].fit(X, ppi)
    ours_pdbs = {json.loads(l)["pdb"].lower() for l in open(ROOT / "data/pdbbind_peptides.jsonl")}
    t100_seqs = {r["seq"] for r in rows}
    ppikb = [json.loads(l) for l in open(ROOT / "data/ppikb_features.jsonl") if json.loads(l).get("desc3d")]
    seen = set(); fresh = []
    for r in sorted(ppikb, key=lambda x: x["pdb"]):
        if r["pdb"].lower() in ours_pdbs or r["pdb"].lower() in man or r["seq"] in t100_seqs:
            continue
        if r["aff_type"] not in ("Kd", "KD", "pKd") or not (2 <= r["length"] <= 50) or not (-18 < r["y"] < -2):
            continue
        if abs(r.get("npep", r["length"]) - r["length"]) > 2 or r.get("npocket", 0) < 10 or r["seq"] in seen:
            continue
        seen.add(r["seq"]); fresh.append(r)
    Xf = np.nan_to_num([list(r["desc3d"]) + list(r["pocket_pkf"]) + _protdcal_descriptors(r["seq"]) for r in fresh])
    yf = np.array([r["y"] for r in fresh]); Lf = np.array([r["length"] for r in fresh]); qf = np.array([abs(r["net_charge"]) for r in fresh])
    ppi_emu = emu.predict(Xf)

    # OUR production-equivalent (seq+pocket trained on 925)
    base = [json.loads(l) for l in open(ROOT / "data/e180_protdcal3d.jsonl") if json.loads(l).get("desc")]
    Xo, yo = [], []
    for b in base:
        ps = e158.pocket_seq(b["pdb"])
        if ps is None:
            continue
        Xo.append(_protdcal_descriptors(b["seq"]) + pkf(ps) + [0, 0, 0, float(len(b["seq"]))]); yo.append(b["y"])
    om = e202._hgb().fit(np.nan_to_num(Xo), np.array(yo))
    ours = om.predict(np.nan_to_num([_protdcal_descriptors(r["seq"]) + list(r["pocket_pkf"]) + [0, 0, 0, float(r["length"])] for r in fresh]))

    print(f"\n=== STEP 2: DIRECT head-to-head on FRESH PPIKB (n={len(fresh)}) — ours vs PPI-EMULATOR ===")
    print(f"  {'band':<14}{'n':>4}{'OURS':>8}{'PPI-emu':>9}{'winner':>10}")
    for nm, mk in [("OVERALL", np.ones(len(yf), bool)), ("<=12", Lf <= 12), ("long13-16", (Lf >= 13) & (Lf <= 16)),
                   ("vlong>=17", Lf >= 17), ("charged|q|>=2", qf >= 2), ("neutral|q|<=1", qf <= 1)]:
        if mk.sum() < 4:
            continue
        ro = corr(ours[mk], yf[mk]); rp = corr(ppi_emu[mk], yf[mk])
        print(f"  {nm:<14}{int(mk.sum()):>4}{ro:>+8.3f}{rp:>+9.3f}{('OURS' if ro > rp else 'PPI'):>10}")
    print(f"\n  faithfulness caveat: emulator reproduces PPI at corr {best[0]:.2f} on T100; this is PPI's")
    print(f"  learned function applied to fresh data = best non-server estimate of real PPI on fresh.")


if __name__ == "__main__":
    main()
