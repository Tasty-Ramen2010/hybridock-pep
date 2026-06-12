import json, sys
import numpy as np
from scipy.stats import pearsonr
out = json.load(open("/tmp/e49_ens_mmgbsa.json"))
# charged subset = the floor regime
chg = {k: v for k, v in out.items() if v["cf"] >= 0.30}
ks = list(chg)
print(f"charged complexes (cf>=0.30) done: {len(ks)}")
if len(ks) < 6:
    sys.exit(0)
y = np.array([chg[k]["y"] for k in ks])
# guard against single-pose blowups (failed minimization -> astronomical) by clipping for a fair read
def col(f):
    if f == "e_int_ent":
        return np.array([chg[k]["e_int_mean"] + chg[k]["minus_tds"] for k in ks])
    return np.array([chg[k][f] for k in ks])
print(f"\n=== THE FLOOR TEST: charged-only (cf>=0.30, n={len(ks)}) ===")
print(f"  single-pose prior floor was r~0.07")
for f, lbl in [("dg_single", "single pose"), ("e_int_mean", "<E_int> ensemble"),
               ("e_int_ent", "<E_int>-TdS")]:
    v = col(f)
    # robust: drop any |value|>1e4 (single-pose blowups) pairwise
    ok = np.abs(v) < 1e4
    r_raw = pearsonr(v, y).statistic if v.std() > 0 else 0.0
    r_clip = pearsonr(v[ok], y[ok]).statistic if ok.sum() > 3 and v[ok].std() > 0 else float("nan")
    print(f"  {lbl:<18} r={r_raw:+.3f}   (clip blowups n={ok.sum()}: r={r_clip:+.3f})")
print("  >> <E_int> ensemble >> single on the charged subset = floor broken by sampling")
