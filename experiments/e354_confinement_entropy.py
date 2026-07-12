"""E354 — PRISM-S prototype + decisive gate: confinement configurational entropy, and does it have RESIDUAL SHAPE?

Ram's idea: compute binding config-entropy along a CONFINEMENT path (integrate the work of progressively
restraining the peptide toward its mean), not as S_bound − S_free. This gives TΔS_config without the cancellation.

Confinement free energy (Simplified Confinement Method spirit): ramp a harmonic restraint k(λ) toward the mean
structure; ΔG_conf = ∫ ⟨∂U_r/∂lnk⟩ d(lnk) over a geometric ladder of k. Referenced to the same analytic pinned
state in both bound and free, so the huge reference entropy cancels ANALYTICALLY:
  TΔS_config(bind) = ΔG_conf(free peptide) − ΔG_conf(bound peptide)      (>0 => peptide rigidifies on binding = penalty)

Free state  = peptide alone (its own fluctuations). Bound state = peptide with receptor held fixed (harmonic wall
on receptor Cα). Both short Langevin MD in implicit solvent (gbn2) for speed. This is the CHEAP prototype.

GATE: compute TΔS_config on a subset of peptide-Kd complexes and correlate with the scorer RESIDUAL. Crude entropy
proxies had ZERO residual shape (E353c); if the REAL confinement entropy does too, entropy is not the missing piece
and we stop. If it correlates, build PRISM-S.

Run: OMP_NUM_THREADS=2 python experiments/e354_confinement_entropy.py --smoke        # 1-2 peptides, validate pipeline
     OMP_NUM_THREADS=2 python experiments/e354_confinement_entropy.py --gate --n 30   # the decisive correlation test
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
KJ = unit.kilojoule_per_mole
KT = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(KJ)
KCAL = 0.239006
# geometric ladder of restraint force constants (kJ/mol/nm^2): free -> strongly pinned
KLADDER = [10.0, 50.0, 200.0, 800.0, 3200.0, 12800.0]
_P = PDBParser(QUIET=True)


class _Sel(Select):
    def __init__(self, chains): self.ch = set(chains)
    def accept_chain(self, c): return c.id in self.ch
    def accept_residue(self, r): return r.id[0] == " "


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
    return pep, rec


def _build(pdb, chains, restrain_receptor_chain=None):
    """Solvate-free implicit-solvent system for the given chains; return (system, positions, pep_ca_idx)."""
    from pdbfixer import PDBFixer
    st = _P.get_structure(pdb, fetch(pdb))[0]
    tmp = tempfile.mktemp(suffix=".pdb"); io = PDBIO(); io.set_structure(st); io.save(tmp, _Sel(chains))
    fx = PDBFixer(filename=tmp)
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findNonstandardResidues(); fx.replaceNonstandardResidues(); fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    return ff, fx.topology, fx.positions, system


def confinement_dG(pdb, pep_chain, rec_chains, kind, n_equil=2000, n_samp=40, n_stride=100):
    """ΔG of confining the PEPTIDE heavy atoms toward their running-mean, integrated over the k-ladder.
    kind='free' (peptide alone) or 'bound' (peptide + receptor pinned)."""
    chains = pep_chain if kind == "free" else pep_chain + rec_chains
    ff, top, pos, system = _build(pdb, chains)
    atoms = list(top.atoms())
    pep_heavy = [a.index for a in atoms if a.residue.chain.id == pep_chain and a.element is not None
                 and a.element.symbol != "H"]
    rec_ca = [a.index for a in atoms if a.residue.chain.id != pep_chain and a.name == "CA"]
    if not pep_heavy:
        raise RuntimeError("no peptide heavy atoms")
    # RMSD restraint (best-fit → removes rigid-body translation/rotation, so we confine INTERNAL config only).
    # U_r = 0.5*kconf*rmsd^2 ; integrand ⟨U_r⟩ over the k-ladder → internal configurational confinement free energy.
    p0 = np.array(pos.value_in_unit(unit.nanometer))
    rmsd = mm.RMSDForce(pos, pep_heavy)
    force = mm.CustomCVForce("0.5*kconf*rmsd^2")
    force.addCollectiveVariable("rmsd", rmsd)
    force.addGlobalParameter("kconf", 0.0)
    system.addForce(force)
    # pin receptor rigidly (strong wall) so 'bound' = peptide fluctuating in a fixed pocket
    if kind == "bound" and rec_ca:
        wall = mm.CustomExternalForce("0.5*kwall*((x-wx)^2+(y-wy)^2+(z-wz)^2)")
        wall.addGlobalParameter("kwall", 50000.0)
        for p in ("wx", "wy", "wz"):
            wall.addPerParticleParameter(p)
        for i in rec_ca:
            wall.addParticle(i, [p0[i][0], p0[i][1], p0[i][2]])
        system.addForce(wall)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, PLAT); ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    # TI over ln(k): for a harmonic restraint U_r ∝ k, dU_r/d(ln k) = U_r, so ΔG_conf = ∫⟨U_r⟩ d(ln k).
    # RMSD auto-aligns → ⟨U_r⟩ reflects INTERNAL fluctuation only (no drift blow-up). Free peptide fluctuates
    # more internally than the pocket-confined bound peptide → larger ΔG_conf(free) → TΔS_config > 0.
    ctx.setParameter("kconf", KLADDER[0]); integ.step(n_equil)
    lnk, meanU = [], []
    for k in KLADDER:
        ctx.setParameter("kconf", k); integ.step(n_equil // 2)
        r2 = []
        for _ in range(n_samp):
            integ.step(n_stride)
            rv = force.getCollectiveVariableValues(ctx)[0]     # rmsd in nm (rigid-body-aligned = internal only)
            r2.append(rv * rv)
        lnk.append(np.log(k)); meanU.append(0.5 * k * float(np.mean(r2)))   # ⟨U_r⟩ = 0.5 k ⟨rmsd²⟩ (kJ/mol)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    dG = float(_trap(np.array(meanU), np.array(lnk)))        # ∫⟨U_r⟩ dlnk (kJ/mol), the confinement free energy
    return dG * KCAL, len(pep_heavy)


def one(pdb, seq):
    ch = find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no peptide chain")
    pep, rec = ch
    if not rec:
        raise RuntimeError("no receptor")
    gf, nf = confinement_dG(pdb, pep, rec, "free")
    gb, nb = confinement_dG(pdb, pep, rec, "bound")
    return gf - gb, nf     # TΔS_config(bind) proxy = ΔG_conf(free) − ΔG_conf(bound)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    charged = [r for r in rows if 6 <= len(r["seq"]) <= 18 and any(c in "DEKR" for c in r["seq"])]
    if a.smoke:
        for r in charged[:2]:
            t = time.time()
            try:
                ts, n = one(r["pdb"], r["seq"])
                print(f"{r['pdb']} {r['seq'][:16]:16s} TΔS_config={ts:+.2f} kcal ({n} atoms, {(time.time()-t)/60:.1f}min)", flush=True)
            except Exception as e:
                print(f"{r['pdb']} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        return
    if a.gate:
        import random
        random.seed(0); random.shuffle(charged)
        sub = charged[:a.n]
        out = []
        for i, r in enumerate(sub):
            t = time.time()
            try:
                ts, n = one(r["pdb"], r["seq"])
                out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "tds": ts, "n": n})
                print(f"[{i+1}/{len(sub)}] {r['pdb']} TΔS={ts:+.2f} ({(time.time()-t)/60:.1f}m)", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(sub)}] {r['pdb']} FAIL {str(e)[:60]}", flush=True)
            json.dump(out, open("data/e354_confinement.json", "w"))
        # gate: correlate TΔS with scorer residual
        _gate_stats(out)


def _gate_stats(out):
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
            "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
    # residual from a scorer trained on the FULL peptide set (so residual is well-defined), evaluated on subset
    allrows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    Xall = np.array([[float(r[f]) for f in FULL] for r in allrows]); yall = np.array([float(r["y"]) for r in allrows])
    gall = np.array([hash(r["seq"][:4]) % 100000 for r in allrows])
    oof = np.full(len(yall), np.nan)
    for tr, te in GroupKFold(8).split(Xall, yall, gall):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0)
        m.fit(Xall[tr], yall[tr]); oof[te] = m.predict(Xall[te])
    resid = {allrows[i]["pdb"]: yall[i] - oof[i] for i in range(len(allrows))}
    ts = np.array([o["tds"] for o in out]); rs = np.array([resid[o["pdb"]] for o in out]); y = np.array([o["y"] for o in out])
    print(f"\n=== GATE (n={len(out)}) ===")
    print(f"  corr(TΔS_confinement, y)              = {pearsonr(ts,y)[0]:+.3f}")
    print(f"  corr(TΔS_confinement, scorer_residual)= {pearsonr(ts,rs)[0]:+.3f}   <- THE GATE")
    print("VERDICT: " + ("entropy HAS residual shape → build PRISM-S." if abs(pearsonr(ts,rs)[0]) > 0.25
                         else "even real confinement entropy has ~no residual shape → the wall is elsewhere; stop honestly."))


if __name__ == "__main__":
    main()
