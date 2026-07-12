"""E19 probe — is crystal-65 pocket->affinity r=0.51 real or overfit? + selectivity relevance."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

cr = json.loads(Path("/tmp/e19_cr.json").read_text())
POCKET = ["poc_n","poc_f_hyd","poc_f_pos","poc_f_neg","poc_net","poc_f_arom","poc_f_pol","poc_eis"]
y = np.array([r["y"] for r in cr])

def loo(feats):
    X = np.array([[r.get(f,0.0) for f in feats] for r in cr], float)
    pred = np.zeros(len(cr))
    for i in range(len(cr)):
        tr = [j for j in range(len(cr)) if j != i]
        mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
        A = np.column_stack([np.ones(len(tr)),(X[tr]-mu)/sd])
        w,*_ = np.linalg.lstsq(A,y[tr],rcond=None)
        pred[i] = np.r_[1,(X[i]-mu)/sd]@w
    return pearsonr(pred,y).statistic

print("LOO pocket->affinity (crystal-65), feature ablation:")
print(f"  all 8 pocket feats:        r={loo(POCKET):+.3f}")
for f in POCKET:
    print(f"  {f:<14} alone:       r={loo([f]):+.3f}")
print(f"  size+hydrophobicity (2):   r={loo(['poc_n','poc_eis']):+.3f}")
print(f"  hyd+arom+net (3):          r={loo(['poc_f_hyd','poc_f_arom','poc_net']):+.3f}")

# permutation null: shuffle y, redo LOO with all 8 — how high does r get by chance?
rng = np.random.default_rng(0)
null = []
Xall = np.array([[r.get(f,0.0) for f in POCKET] for r in cr],float)
for _ in range(200):
    yp = rng.permutation(y)
    pred = np.zeros(len(cr))
    for i in range(len(cr)):
        tr=[j for j in range(len(cr)) if j!=i]
        mu,sd=Xall[tr].mean(0),Xall[tr].std(0)+1e-9
        A=np.column_stack([np.ones(len(tr)),(Xall[tr]-mu)/sd])
        w,*_=np.linalg.lstsq(A,yp[tr],rcond=None)
        pred[i]=np.r_[1,(Xall[i]-mu)/sd]@w
    null.append(pearsonr(pred,yp).statistic)
null=np.array(null)
print(f"\nPermutation null (200x, all 8 feats): mean r={null.mean():+.3f}, "
      f"95th pct={np.percentile(null,95):+.3f}, max={null.max():+.3f}")
print(f"  -> observed +0.51 is {'ABOVE null 95th pct (REAL)' if 0.51>np.percentile(null,95) else 'within null (OVERFIT)'}")
