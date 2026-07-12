"""E29 — WHY do we get 0.228 on the diverse 98? Locate the failure (data/calib/features).

1. Extraction sanity: peptide size, interface contacts — is the data clean or buggy?
2. Per-feature signal: raw corr(feature, ΔG) on the 98 vs crystal-65 — do features transfer?
3. Failure by bin: length / affinity strength — where is error concentrated?
4. DATA hypothesis: pool crystal-65 + 98, LOO — does more diverse training help?
5. SELF-CALIBRATION: does per-length or per-SS calibration help?
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES  # noqa: E402

FK = GEOMETRY_FEATURES
b98 = list(json.loads(Path("/tmp/e28_feats.json").read_text()).values())
geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
mj = json.loads(Path("/tmp/e24_contact.json").read_text())
cr = [dict(geo[p], mj_contact=mj[p]["mj_contact"]) for p in geo if p in mj]

y98 = np.array([r["y"] for r in b98]); X98 = np.array([[r.get(f, 0.0) for f in FK] for r in b98])
L98 = np.array([r["L"] for r in b98])
ycr = np.array([r["y"] for r in cr]); Xcr = np.array([[r.get(f, 0.0) for f in FK] for r in cr])


def loo(X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return p


print("=== 1. EXTRACTION SANITY (is the data clean?) ===")
# proxy for interface size = bsa_hyd+sasa_hb+sasa_sb; n contacts ~ |mj|/avg
iface = X98[:, FK.index("bsa_hyd")] + X98[:, FK.index("sasa_hb")] + X98[:, FK.index("sasa_sb")]
print(f"  peptide length: median {np.median(L98):.0f} range [{L98.min()},{L98.max()}]")
print(f"  interface-burial proxy: median {np.median(iface):.1f}, ZERO-interface complexes: "
      f"{np.sum(iface < 0.5)}/{len(b98)}")
print(f"  hb_count: median {np.median(X98[:,FK.index('hb_count')]):.0f}, "
      f"complexes with 0 H-bonds: {np.sum(X98[:,FK.index('hb_count')]==0)}")
print(f"  >> if many zero-interface/0-hb, extraction grabbed wrong chains or wrong assembly")

print("\n=== 2. PER-FEATURE SIGNAL: raw corr(feature, ΔG) — crystal-65 vs 98 ===")
print(f"  {'feature':<12}{'crystal-65':>12}{'the-98':>10}{'transfers?':>12}")
for i, f in enumerate(FK):
    rc = pearsonr(Xcr[:, i], ycr).statistic
    r9 = pearsonr(X98[:, i], y98).statistic
    tr = "yes" if (rc * r9 > 0 and abs(r9) > 0.12) else ("SIGN-FLIP" if rc * r9 < 0 else "weak")
    print(f"  {f:<12}{rc:>+12.3f}{r9:>+10.3f}{tr:>12}")

print("\n=== 3. FAILURE BY BIN (98, geometry+MJ LOO) ===")
p98 = loo(X98, y98)
err = np.abs(p98 - y98)
for lo, hi, lbl in [(0, 9, "short<9"), (9, 13, "9-12"), (13, 20, "13-19"), (20, 99, ">=20")]:
    m = (L98 >= lo) & (L98 < hi)
    if m.sum() >= 5:
        print(f"  len {lbl:<8} n={m.sum():>2}  LOO-subset r={pearsonr(p98[m],y98[m]).statistic:+.3f}  "
              f"mean|err|={err[m].mean():.2f}")

print("\n=== 4. DATA HYPOTHESIS: pool crystal-65 + 98, LOO (does diverse training help the 98?) ===")
Xp = np.vstack([Xcr, X98]); yp = np.concatenate([ycr, y98])
pp = loo(Xp, yp)
p98_in_pool = pp[len(cr):]
print(f"  98 predicted from POOLED model: r={pearsonr(p98_in_pool,y98).statistic:+.3f} "
      f"RMSE={np.sqrt(((p98_in_pool-y98)**2).mean()):.2f}")
print(f"  98 alone (in-dist LOO):         r={pearsonr(p98,y98).statistic:+.3f}")
print(f"  (if pooled > alone, more diverse DATA helps)")

print("\n=== 5. SELF-CALIBRATION: per-length-bin demean (does removing a length baseline help?) ===")
# within-length-bin: does the feature model rank correctly after removing bin mean?
preds, ys = [], []
for lo, hi in [(0, 11), (11, 15), (15, 99)]:
    m = (L98 >= lo) & (L98 < hi)
    if m.sum() >= 6:
        pm = loo(X98[m], y98[m])
        preds.append(pm - pm.mean()); ys.append(y98[m] - y98[m].mean())
if preds:
    pr = pearsonr(np.concatenate(preds), np.concatenate(ys)).statistic
    print(f"  within-length-bin pooled r = {pr:+.3f}  (vs global 0.228)")
    print(f"  (if higher, a per-length baseline is the missing piece = self-calibration)")
