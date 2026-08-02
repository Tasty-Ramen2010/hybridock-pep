#!/usr/bin/env python3
"""build_clean_eval.py — robust multi-complex eval set for RAPiDock.
Avoids the 3 harness bugs that produced silent "NO POSES":
  (1) ABSOLUTE paths only (nsweep runs inference with cwd=infer-dir -> relative paths break);
  (2) sequence extracted with MDAnalysis 'name CA' so len(seq)==crystal CA count (RMSD ref matches);
  (3) sources from RefPepDB-RecentSet + propedia_formatted (NOT benchmark_real, which has a stale
      Extended-conformation cache). Writes data/verdict_clean.csv (6 complexes x 4 length buckets).
"""
import os, glob, csv, collections, warnings
warnings.filterwarnings("ignore")
import MDAnalysis as mda
ROOT="/home/igem/unknown_software"
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def mda_seq(pep):
    try:
        ca=mda.Universe(pep).select_atoms("name CA")
        names=[a.resname for a in ca]
        return "".join(AA.get(n,"") for n in names), len(names)
    except Exception: return None,0
cands=[]
for base in ["datasets/RefPepDB-RecentSet","datasets/propedia_formatted"]:
    for d in sorted(glob.glob(f"{ROOT}/{base}/*")):
        n=os.path.basename(d); rec=f"{d}/{n}_protein_pocket.pdb"; pep=f"{d}/{n}_peptide.pdb"
        if not(os.path.exists(rec) and os.path.exists(pep)): continue
        s,L=mda_seq(pep)
        if s and len(s)==L and 4<=L<=30:
            b="short" if L<=8 else "medium" if L<=12 else "long" if L<=19 else "very_long"
            cands.append((n,rec,pep,s,L,b))
by=collections.defaultdict(list)
for c in cands: by[c[5]].append(c)
sel=[]
for b in ["short","medium","long","very_long"]: sel+=by.get(b,[])[:6]
with open(f"{ROOT}/data/verdict_clean.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["name","receptor","peptide_pdb","seq","pep_len","ss_class","length_bucket"])
    for n,rec,pep,s,L,b in sel: w.writerow([n,rec,pep,s,L,"NA",b])
print(f"wrote {len(sel)} complexes:", collections.Counter(x[5] for x in sel))
