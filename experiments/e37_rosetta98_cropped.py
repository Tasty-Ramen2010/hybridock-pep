"""E37 — Rosetta ref2015+relax on the 98 with POCKET-CROPPED receptors (fast).
Tests if FlexPepDock physics (ref2015: vdW + Lazaridis-Karplus SOLVATION + orientation H-bonds
+ Dunbrack rotamer + reference energies) carries cross-target signal our geometry misses, AND
whether it stays sign-consistent (ref2015 has reference-energy entropy proxy that single-pose
MM-GBSA lacked)."""
import json,sys,warnings,time,numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0,"scripts")
from pathlib import Path
from Bio.PDB import PDBParser,PDBIO,Select
from scipy.stats import pearsonr
from rosetta_ref2015_eval import init_rosetta, score_complex
P=PDBParser(QUIET=True)
def crop(pep,rec,out,rad=10.0):
    pm=P.get_structure("p",str(pep))[0]; rm=P.get_structure("r",str(rec))[0]
    pxyz=np.array([a.coord for r in pm.get_residues() if r.id[0]==" " for a in r if a.element!="H"])
    keep=set()
    for ch in rm:
        for res in ch:
            if res.id[0]!=" ": continue
            for a in res:
                if a.element!="H" and np.min(((pxyz-a.coord)**2).sum(1))<=rad*rad: keep.add((ch.id,res.id)); break
    class S(Select):
        def accept_residue(self,r): return (r.get_parent().id,r.id) in keep
    io=PDBIO(); io.set_structure(P.get_structure("r2",str(rec))); io.save(str(out),S())
    return out if out.exists() and out.stat().st_size>200 else None
pr=init_rosetta()
b98=json.load(open("/tmp/e28_feats.json")); work=Path("/tmp/ppep_work"); cropd=Path("/tmp/ppep_crop"); cropd.mkdir(exist_ok=True)
out_path=Path("/tmp/e37.json"); out=json.loads(out_path.read_text()) if out_path.exists() else {}
t0=time.time()
for key,r in b98.items():
    if key in out: continue
    pep=work/f"{key}_pep.pdb"; rec=work/f"{key}_rec.pdb"
    if not pep.exists() or not rec.exists(): continue
    cr=crop(pep,rec,cropd/f"{key}_crop.pdb")
    if not cr: continue
    try: s=score_complex(pr,str(pep),str(cr),relax=True)
    except Exception as e: print(f"  {key} FAIL {type(e).__name__}",flush=True); continue
    out[key]=dict(y=r["y"],**s); out_path.write_text(json.dumps(out))
    if len(out)%10==0: print(f"  {len(out)}/98 ({(time.time()-t0)/len(out):.0f}s/cplx)",flush=True)
y=np.array([v["y"] for v in out.values()])
def loo1(x,y):
    p=np.zeros(len(y))
    for i in range(len(y)):
        tr=[j for j in range(len(y)) if j!=i];a,b=np.polyfit(x[tr],y[tr],1);p[i]=a*x[i]+b
    return pearsonr(p,y).statistic,np.sqrt(((p-y)**2).mean())
print(f"\n=== Rosetta ref2015+relax on the 98 (n={len(out)}) ===")
for f in ["ros_total","ros_ifdG"]:
    v=np.array([vv[f] for vv in out.values()])
    if v.std()>0:
        r,e=loo1(v,y); print(f"  {f} LOO-fit r={r:+.3f} RMSE={e:.2f} | raw {pearsonr(v,y).statistic:+.3f}")
print("  [same data] PPI-Affinity 0.554 | ours geometry+MJ 0.228 | crystal-65 ref2015 was 0.423")
# does ref2015 ADD to our geometry on the 98? join
geo=json.load(open("/tmp/e28_feats.json"))
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES as EXT
rows=[dict(geo[k],ros=out[k]["ros_ifdG"]) for k in out if k in geo]
yy=np.array([r["y"] for r in rows])
def loo(feats):
    X=np.array([[r.get(f,0.) for f in feats] for r in rows]);p=np.zeros(len(yy))
    for i in range(len(yy)):
        tr=[j for j in range(len(yy)) if j!=i];mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
        A=np.column_stack([np.ones(len(tr)),(X[tr]-mu)/sd]);w,*_=np.linalg.lstsq(A,yy[tr],rcond=None);p[i]=np.r_[1,(X[i]-mu)/sd]@w
    return pearsonr(p,yy).statistic,np.sqrt(((p-yy)**2).mean())
print(f"\n  geometry+MJ:        r={loo(EXT)[0]:+.3f}")
print(f"  + ref2015 ifdG:     r={loo(EXT+['ros'])[0]:+.3f}  (does FlexPepDock physics bridge our gap?)")
