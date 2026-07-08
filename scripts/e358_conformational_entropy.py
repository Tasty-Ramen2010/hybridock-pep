"""E358 — PRISM-S v2: conformational entropy done PROPERLY (the term E354 missed).

Fixes the 4 audited errors (docs/entropy_implementation_audit):
  E1: compute the DOMINANT term — conformational (between-basin) entropy from DIHEDRAL occupancy, not within-basin
      RMSD vibration.
  E2: torsion coordinates (φ, ψ, χ1), not Cartesian.
  E4: longer sampling than 4 ps.
And crucially, done with NO catastrophic cancellation, via the Mutual Information Expansion (Ram's "derivatives, no
huge subtraction"): the total entropy is decomposed into LOCAL terms
        S ≈ Σ_i S_i(marginal per dihedral)  −  Σ_(i<j adjacent) I_ij(mutual information)
and the BINDING change is taken LOCALLY, per dihedral:
        TΔS_conf = Σ_i [S_i(free) − S_i(bound)]  −  Σ_adj [I_ij(free) − I_ij(bound)]
Each ΔS_i is bounded in [0, ln B] (a few nats) — small, well-conditioned. We NEVER subtract two large total
entropies; the huge common part cancels term-by-term, exactly like the FEP derivative trick. Miller–Madow bias
correction handles finite sampling.

Free state = peptide alone; bound = peptide with receptor Cα pinned. Same solvent model both sides so its bias
largely cancels in the difference.

Run: OMP_NUM_THREADS=2 python scripts/e358_conformational_entropy.py --smoke
     OMP_NUM_THREADS=2 python scripts/e358_conformational_entropy.py --gate --n 40
"""
from __future__ import annotations
import sys, json, argparse, time, tempfile
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch
import openmm as mm
from openmm import app, unit

PLAT = mm.Platform.getPlatformByName("CUDA")
KCAL_PER_NAT = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(unit.kilocalorie_per_mole)  # RT in kcal
NBIN = 24                       # 15° bins
GAMMA = {"CB": None}            # χ1 third atom picked per residue below
CHI1_G = {"SER": "OG", "THR": "OG1", "CYS": "SG", "VAL": "CG1", "ILE": "CG1", "THY": "CG",
          "ASP": "CG", "ASN": "CG", "GLU": "CG", "GLN": "CG", "HIS": "CG", "LEU": "CG", "MET": "CG",
          "PHE": "CG", "TYR": "CG", "TRP": "CG", "ARG": "CG", "LYS": "CG", "PRO": "CG"}
_P = PDBParser(QUIET=True)


class _Sel(Select):
    def __init__(self, ch): self.ch = set(ch)
    def accept_chain(self, c): return c.id in self.ch
    def accept_residue(self, r): return r.id[0] == " "


def _compo_sim(a, b):
    """Composition cosine similarity of two sequences (order-independent — robust to cyclic/scrambled numbering)."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    va = np.array([ca[k] for k in keys]); vb = np.array([cb[k] for k in keys])
    d = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / d) if d else 0.0


def find_chains(pdb, seq):
    """Peptide chain = best sequence match (positional OR composition, robust to cyclic/scrambled); receptor =
    all OTHER chains with >=4 residues (excludes lone-residue crystallization artifacts like a free PRO)."""
    st = _P.get_structure(pdb, fetch(pdb))[0]
    seq = seq.upper()
    chains = []
    for ch in st:
        res = [r for r in ch if r.id[0] == " "]
        try:
            cs = "".join(seq1(r.get_resname()) for r in res)
        except Exception:
            cs = ""
        chains.append((ch.id, cs, len(res)))
    # score each chain as a peptide candidate
    best, best_score = None, 0.0
    for cid, cs, nres in chains:
        if not cs:
            continue
        n = min(len(cs), len(seq))
        pos = (sum(cs[i] == seq[i] for i in range(n)) / max(len(cs), len(seq))) if n else 0.0
        score = max(1.0 if (seq in cs or cs in seq) else 0.0, pos, _compo_sim(cs, seq) if abs(len(cs) - len(seq)) <= 4 else 0.0)
        if score > best_score:
            best, best_score = cid, score
    if best is None or best_score < 0.6:
        return None
    rec = "".join(sorted(cid for cid, cs, nres in chains if cid != best and nres >= 4))   # drop tiny artifact chains
    return (best, rec) if rec else None


def _dihedral(p):
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / (np.linalg.norm(b1) + 1e-9)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def _build(pdb, chains, pep_chain, bound):
    from pdbfixer import PDBFixer
    st = _P.get_structure(pdb, fetch(pdb))
    tmp = tempfile.mktemp(suffix=".pdb"); io = PDBIO(); io.set_structure(st); io.save(tmp, _Sel(chains))
    fx = PDBFixer(filename=tmp)
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    p0 = np.array(fx.positions.value_in_unit(unit.nanometer))
    if bound:                     # pin receptor Cα
        wall = mm.CustomExternalForce("0.5*kw*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        wall.addGlobalParameter("kw", 50000.0)
        for q in ("x0", "y0", "z0"):
            wall.addPerParticleParameter(q)
        for a in fx.topology.atoms():
            if a.residue.chain.id != pep_chain and a.name == "CA":
                wall.addParticle(a.index, [p0[a.index][0], p0[a.index][1], p0[a.index][2]])
        system.addForce(wall)
    # index the peptide backbone (φ/ψ) and χ1 atoms, keyed by peptide-LOCAL ordinal (consistent free vs bound)
    pep_res_order = []
    for r in fx.topology.residues():
        if r.chain.id == pep_chain:
            pep_res_order.append(r.index)
    ord_of = {ridx: k for k, ridx in enumerate(pep_res_order)}
    res_atoms = {}
    for a in fx.topology.atoms():
        if a.residue.chain.id == pep_chain:
            res_atoms.setdefault((ord_of[a.residue.index], a.residue.name), {})[a.name] = a.index
    return ff, fx.topology, fx.positions, system, res_atoms


def _dihedral_defs(res_atoms):
    """Return list of (label, (i0,i1,i2,i3)) atom-index quadruples for φ, ψ, χ1 across the peptide."""
    keys = sorted(res_atoms.keys())
    defs = []
    for k, (ridx, rname) in enumerate(keys):
        a = res_atoms[(ridx, rname)]
        prev = res_atoms.get(keys[k - 1]) if k > 0 else None
        nxt = res_atoms.get(keys[k + 1]) if k < len(keys) - 1 else None
        if prev and all(x in a for x in ("N", "CA", "C")) and "C" in prev:
            defs.append((f"phi{ridx}", (prev["C"], a["N"], a["CA"], a["C"])))
        if nxt and all(x in a for x in ("N", "CA", "C")) and "N" in nxt:
            defs.append((f"psi{ridx}", (a["N"], a["CA"], a["C"], nxt["N"])))
        g = CHI1_G.get(rname)
        if g and all(x in a for x in ("N", "CA", "CB")) and g in a:
            defs.append((f"chi{ridx}", (a["N"], a["CA"], a["CB"], a[g])))
    return defs


def sample_dihedrals(pdb, chains, pep_chain, bound, n_equil=30000, n_frames=300, n_stride=200):
    ff, top, pos, system, res_atoms = _build(pdb, chains, pep_chain, bound)
    defs = _dihedral_defs(res_atoms)
    if not defs:
        raise RuntimeError("no dihedrals")
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, PLAT); ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(n_equil)   # 40 ps equil
    quads = [q for _, q in defs]
    series = np.zeros((n_frames, len(defs)))
    for f in range(n_frames):
        integ.step(n_stride)
        x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        for j, q in enumerate(quads):
            series[f, j] = _dihedral(x[list(q)])
    labels = [lab for lab, _ in defs]
    return labels, series      # angles in radians, shape (frames, n_dihedral)


def _marg_entropy(theta):
    """Miller–Madow-corrected marginal entropy (nats) of a circular variable from samples."""
    h, _ = np.histogram(theta, bins=NBIN, range=(-np.pi, np.pi))
    N = h.sum(); p = h[h > 0] / N
    S = -np.sum(p * np.log(p))
    return S + (np.count_nonzero(h) - 1) / (2 * N)      # Miller–Madow bias correction


def _mutual_info(a, b):
    ha, _, _ = np.histogram2d(a, b, bins=NBIN, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    N = ha.sum(); pij = ha / N
    pi = pij.sum(1, keepdims=True); pj = pij.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = pij * (np.log(pij) - np.log(pi) - np.log(pj))
    return np.nansum(term[pij > 0])


def dS_conf(pdb, seq):
    """TΔS_conf(bind) = Σ_i[S_i(free)-S_i(bound)] - Σ_adj[I_ij(free)-I_ij(bound)]  (kcal/mol, LOCAL, no cancellation)."""
    ch = find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    lf, sf = sample_dihedrals(pdb, pep, pep, bound=False)
    lb, sb = sample_dihedrals(pdb, pep + rec, pep, bound=True)
    common = [l for l in lf if l in lb]
    if not common:
        raise RuntimeError("no common dihedrals")
    fi = {l: i for i, l in enumerate(lf)}; bi = {l: i for i, l in enumerate(lb)}
    # 1st-order: per-dihedral local ΔS (each bounded, small — no giant subtraction)
    dS1 = sum(_marg_entropy(sf[:, fi[l]]) - _marg_entropy(sb[:, bi[l]]) for l in common)
    # 2nd-order MIE: adjacent φ_i-ψ_i and backbone-χ1 correlations only (local, robust on short MD)
    def resid(l): return int("".join(c for c in l if c.isdigit()))
    dI = 0.0
    for a in common:
        for b in common:
            if a >= b:
                continue
            if resid(a) == resid(b):     # same-residue dihedral pair (φ-ψ, backbone-χ1) = the local coupling
                dI += _mutual_info(sf[:, fi[a]], sf[:, fi[b]]) - _mutual_info(sb[:, bi[a]], sb[:, bi[b]])
    dS_nats = dS1 - dI
    return dS_nats * KCAL_PER_NAT, len(common)   # TΔS in kcal/mol (>0 => entropy lost on binding = penalty)


def _gate(out):
    from scipy.stats import pearsonr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    if len(out) < 6:
        print(f"only {len(out)}"); return
    allrows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
            "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
    X = np.array([[float(r[f]) for f in FULL] for r in allrows]); y = np.array([float(r["y"]) for r in allrows])
    g = np.array([hash(r["seq"][:4]) % 100000 for r in allrows]); oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(8).split(X, y, g):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
    resid = {allrows[i]["pdb"]: float(y[i] - oof[i]) for i in range(len(allrows))}
    ts = np.array([o["tds"] for o in out]); rs = np.array([resid[o["pdb"]] for o in out]); yy = np.array([o["y"] for o in out])
    print(f"\n=== GATE n={len(out)}  (PRISM-S v2 conformational entropy) ===")
    print(f"  TΔS: mean={ts.mean():+.2f} std={ts.std():.2f} range [{ts.min():+.1f},{ts.max():+.1f}]")
    print(f"  corr(TΔS, y)               = {pearsonr(ts,yy)[0]:+.3f}")
    print(f"  corr(TΔS, scorer_residual) = {pearsonr(ts,rs)[0]:+.3f}   <- vs E354 confinement −0.06..−0.22, crude proxy +0.09")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
    rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    pool = [r for r in rows if 6 <= len(r["seq"]) <= 16]
    if a.smoke:
        for r in pool[:2]:
            t = time.time()
            try:
                ts, n = dS_conf(r["pdb"], r["seq"])
                print(f"{r['pdb']} {r['seq'][:14]:14s} TΔS_conf={ts:+.2f} kcal ({n} dih, {(time.time()-t)/60:.1f}m)", flush=True)
            except Exception as e:
                print(f"{r['pdb']} FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        return
    if a.gate:
        import random; random.seed(3); random.shuffle(pool)
        out = []
        for i, r in enumerate(pool[:a.n]):
            t = time.time()
            try:
                ts, n = dS_conf(r["pdb"], r["seq"])
                out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "tds": ts, "n": n})
                print(f"[{i+1}/{a.n}] {r['pdb']} TΔS={ts:+.2f} ({(time.time()-t)/60:.1f}m)", flush=True)
            except Exception as e:
                print(f"[{i+1}/{a.n}] {r['pdb']} FAIL {str(e)[:50]}", flush=True)
            json.dump(out, open("data/e358_conf_entropy.json", "w"))
        _gate(out)


if __name__ == "__main__":
    main()
