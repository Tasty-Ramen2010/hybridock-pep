"""E356 — the untested physics term: does RISM desolvation-on-binding have residual shape?

Desolvation ΔG on binding = exchem(complex) − exchem(receptor) − exchem(peptide), each from 3D-RISM (the same
machinery as E349). Correlate with the scorer residual on a subset of peptide-Kd complexes. If it correlates,
desolvation is a missing term; if ~0, it's not (as the sequence proxies suggested), and we've ruled out the last
untested physics lever honestly.

Run: OMP_NUM_THREADS=4 python experiments/e356_rism_desolv.py --n 15
"""
from __future__ import annotations
import sys, json, argparse, subprocess, tempfile, shutil, os, time
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch

AMBER = Path("/home/igem/miniconda3/envs/ambertools")
RISM = AMBER / "bin" / "rism3d.snglpnt"; TLEAP = AMBER / "bin" / "tleap"; PDB4AMBER = AMBER / "bin" / "pdb4amber"
XVV = AMBER / "dat" / "rism1d" / "cSPCE" / "cSPCE_kh.xvv"
ENV = {**os.environ, "AMBERHOME": str(AMBER), "PATH": f"{AMBER/'bin'}:{os.environ.get('PATH','')}"}
WORK = Path("/home/igem/unknown_software/runs/e356_rism"); WORK.mkdir(parents=True, exist_ok=True)
_P = PDBParser(QUIET=True)


class _Sel(Select):
    def __init__(self, ch): self.ch = set(ch)
    def accept_chain(self, c): return c.id in self.ch
    def accept_residue(self, r): return r.id[0] == " "


def _isnum(t):
    try:
        float(t); return True
    except ValueError:
        return False


def parse_exchem(stdout):
    val = np.nan
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("rism_excessChemicalPotential"):
            toks = [t for t in s.split()[1:] if _isnum(t)]
            if toks:
                val = float(toks[0])
    return val


def find_chains(pdb, seq):
    st = _P.get_structure(pdb, fetch(pdb))[0]
    seq = seq.upper(); pep = None
    for ch in st:
        res = [r for r in ch if r.id[0] == " "]
        try:
            cs = "".join(seq1(r.get_resname()) for r in res)
        except Exception:
            continue
        n = min(len(cs), len(seq))
        if cs and (seq in cs or cs in seq or (n and sum(cs[i] == seq[i] for i in range(n)) / max(len(cs), len(seq)) > 0.7)):
            pep = ch.id
    if pep is None:
        return None
    rec = "".join(sorted(ch.id for ch in st if ch.id != pep and any(r.id[0] == " " for r in ch)))
    return (pep, rec) if rec else None


def exchem(pdb, chains, tag):
    wd = WORK / f"{pdb}_{tag}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    st = _P.get_structure(pdb, fetch(pdb))
    raw = wd / "raw.pdb"; io = PDBIO(); io.set_structure(st); io.save(str(raw), _Sel(chains))
    clean = wd / "clean.pdb"
    subprocess.run([str(PDB4AMBER), "-i", str(raw), "-o", str(clean), "--nohyd", "--dry"],
                   env=ENV, capture_output=True, timeout=300, cwd=wd)
    if not clean.exists():
        raise RuntimeError("pdb4amber failed")
    (wd / "leap.in").write_text(
        "source leaprc.protein.ff14SB\nmol = loadpdb clean.pdb\nsaveamberparm mol mol.prmtop mol.rst7\nquit\n")
    subprocess.run([str(TLEAP), "-f", "leap.in"], env=ENV, capture_output=True, timeout=600, cwd=wd)
    prm, rst = wd / "mol.prmtop", wd / "mol.rst7"
    if not (prm.exists() and rst.exists()):
        raise RuntimeError("tleap failed")
    p = subprocess.run([str(RISM), "--pdb", str(clean), "--prmtop", str(prm), "--rst", str(rst),
                        "--xvv", str(XVV), "--closure", "kh", "--buffer", "10", "--grdspc", "0.5,0.5,0.5",
                        "--tolerance", "1e-4"], env=ENV, capture_output=True, timeout=4000, cwd=wd, text=True)
    v = parse_exchem(p.stdout)
    if v is None or np.isnan(v):
        raise RuntimeError(f"RISM failed rc={p.returncode}")
    return v


def desolv(pdb, seq):
    ch = find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    g_cplx = exchem(pdb, pep + rec, "cplx")
    g_rec = exchem(pdb, rec, "rec")
    g_pep = exchem(pdb, pep, "pep")
    return g_cplx - g_rec - g_pep    # solvation change on binding (desolvation penalty if positive)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=15); a = ap.parse_args()
    rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    charged = [r for r in rows if 6 <= len(r["seq"]) <= 18]
    import random; random.seed(1); random.shuffle(charged)
    out = []
    print(f"=== E356 RISM desolvation-on-binding, n={a.n} ===", flush=True)
    for i, r in enumerate(charged[:a.n]):
        t = time.time()
        try:
            d = desolv(r["pdb"], r["seq"])
            out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "desolv": d})
            print(f"[{i+1}/{a.n}] {r['pdb']} Δsolv={d:+.1f} kcal ({(time.time()-t)/60:.0f}m)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{a.n}] {r['pdb']} FAIL {str(e)[:50]}", flush=True)
        json.dump(out, open("data/e356_desolv.json", "w"))
    _gate(out)


def _gate(out):
    from scipy.stats import pearsonr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    if len(out) < 5:
        print("too few"); return
    allrows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
            "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
    X = np.array([[float(r[f]) for f in FULL] for r in allrows]); y = np.array([float(r["y"]) for r in allrows])
    g = np.array([hash(r["seq"][:4]) % 100000 for r in allrows]); oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(X, y, g):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
    resid = {allrows[i]["pdb"]: float(y[i] - oof[i]) for i in range(len(allrows))}
    ds = np.array([o["desolv"] for o in out]); rs = np.array([resid[o["pdb"]] for o in out]); yy = np.array([o["y"] for o in out])
    print(f"\n=== GATE n={len(out)} ===")
    print(f"  corr(Δsolv, y)               = {pearsonr(ds,yy)[0]:+.3f}")
    print(f"  corr(Δsolv, scorer_residual) = {pearsonr(ds,rs)[0]:+.3f}   <- if >|0.25|, desolvation is a missing term")


if __name__ == "__main__":
    main()
