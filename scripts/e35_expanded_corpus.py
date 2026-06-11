"""E35 — expanded corpus: 406 harvested peptides + crystal-65 + the-98.
Compute the universal INTENSIVE features (bsa_hyd, mj_per_contact, f_hyd_iface,
frac_pol_satisfied) on the harvested peptides; pool everything; test whether MORE diverse
data lifts the universal-intensive model's generalization toward PPI-Affinity 0.554."""
import json,sys,warnings,numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from pathlib import Path
from scipy.stats import pearsonr
sys.path.insert(0,"scripts")
from e31_intensive_features import intensive_features
UNI=["bsa_hyd","mj_per_contact","f_hyd_iface","frac_pol_satisfied"]
# harvested peptides -> intensive features
harv=json.load(open("/tmp/e30_harvest.json")); work=Path("/tmp/ppb_work")
out=json.loads(Path("/tmp/e35_harv.json").read_text()) if Path("/tmp/e35_harv.json").exists() else {}
todo=[(k,v) for k,v in harv.items() if v and k not in out]
print(f"computing intensive features on {len(todo)} harvested peptides...",flush=True)
for k,v in todo:
    pepf=work/f"{k}_pep.pdb"; recf=work/f"{k}_rec.pdb"
    if not pepf.exists() or not recf.exists(): out[k]=None; continue
    try:
        f=intensive_features(pepf,recf)
        out[k]=dict(f,y=v["y"]) if f else None
    except Exception: out[k]=None
    if len([x for x in out.values() if x])%25==0: Path("/tmp/e35_harv.json").write_text(json.dumps(out))
Path("/tmp/e35_harv.json").write_text(json.dumps(out))
harv_f=[v for v in out.values() if v]
# crystal-65 + 98 intensive
inten=json.load(open("/tmp/e31_intensive.json")); cr=inten["cr"]; b98=inten["b98"]
print(f"\ncorpus: crystal-65={len(cr)} + the-98={len(b98)} + harvested={len(harv_f)} = {len(cr)+len(b98)+len(harv_f)}")
def loo(rows,feats):
    y=np.array([r["y"] for r in rows]); X=np.array([[r.get(f,0.) for f in feats] for r in rows]); p=np.zeros(len(y))
    for i in range(len(y)):
        tr=[j for j in range(len(y)) if j!=i];mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
        A=np.column_stack([np.ones(len(tr)),(X[tr]-mu)/sd]);w,*_=np.linalg.lstsq(A,y[tr],rcond=None);p[i]=np.r_[1,(X[i]-mu)/sd]@w
    return pearsonr(p,y).statistic,np.sqrt(((p-y)**2).mean())
print("\n=== generalization trajectory (universal intensive features, LOO) ===")
for nm,rows in [("crystal-65 only (65)",cr),("+the-98 (163)",cr+b98),
                ("FULL corpus (~570)",cr+b98+harv_f)]:
    r,e=loo(rows,UNI); print(f"  {nm:<24} n={len(rows):>3}  r={r:+.3f} RMSE={e:.2f}")
# held-out: train on (cr+harv), predict the-98 (true generalization to independent set)
def transfer(train,test,feats):
    Xtr=np.array([[r.get(f,0.) for f in feats] for r in train]);ytr=np.array([r["y"] for r in train])
    Xte=np.array([[r.get(f,0.) for f in feats] for r in test]);yte=np.array([r["y"] for r in test])
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-9;A=np.column_stack([np.ones(len(Xtr)),(Xtr-mu)/sd]);w,*_=np.linalg.lstsq(A,ytr,rcond=None)
    return pearsonr(np.column_stack([np.ones(len(Xte)),(Xte-mu)/sd])@w,yte).statistic
print(f"\n  TRUE generalization (train crystal65+harvested -> predict independent 98):")
print(f"    {transfer(cr+harv_f,b98,UNI):+.3f}  (was -0.14 with old extensive features; PPI-Affinity 0.554)")
