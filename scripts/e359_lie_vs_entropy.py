"""E359 — --ultra-level (500ps) head-to-head: LIE vs PRISM-S conformational entropy vs both, on peptide Kd.

Ram: --ultra is the compute-heavy tier — take it to 500ps and watch. And: does PRISM-S beat LIE (which nails
kcal/mol in a few GPU-hours)? We run ONE long MD per complex (free + bound, 500ps) and extract BOTH:
  • LIE enthalpy: ⟨V_elec⟩, ⟨V_vdw⟩ of the peptide–receptor interaction over the bound trajectory (Åqvist LIE
    features; coefficients fit by leave-one-out — LIE is always a fitted method).
  • PRISM-S entropy: dihedral MIE conformational TΔS (free−bound), local/no-cancellation (E358).
Then compare, calibrated leave-one-out, vs experimental ΔG:
  LIE-only  |  entropy-only  |  LIE+entropy  |  scorer  |  scorer+LIE+entropy
This tells us where we lack vs LIE and whether entropy adds to the enthalpic picture.

Run: OMP_NUM_THREADS=2 python scripts/e359_lie_vs_entropy.py --gate --n 15
"""
from __future__ import annotations
import sys, json, argparse, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
import e358_conformational_entropy as e358
import openmm as mm
from openmm import unit

COUL = 138.935458      # kJ·nm/(mol·e²)
NBIN = e358.NBIN
KCAL_PER_NAT = e358.KCAL_PER_NAT
KJ2KCAL = 0.239006


def _nb_params(system):
    nb = next(system.getForce(i) for i in range(system.getNumForces())
              if system.getForce(i).__class__.__name__ == "NonbondedForce")
    q = np.zeros(nb.getNumParticles()); sig = np.zeros_like(q); eps = np.zeros_like(q)
    for i in range(nb.getNumParticles()):
        c, s, e = nb.getParticleParameters(i)
        q[i] = c.value_in_unit(unit.elementary_charge); sig[i] = s.value_in_unit(unit.nanometer)
        eps[i] = e.value_in_unit(unit.kilojoule_per_mole)
    return q, sig, eps


def _lie_energy(x, pep_idx, rec_idx, q, sig, eps, cutoff=1.2):
    """Peptide–receptor MM interaction: (V_elec, V_vdw) in kcal/mol for one frame (receptor atoms within cutoff)."""
    xp = x[pep_idx]; xr = x[rec_idx]
    # prune receptor atoms near the peptide
    from scipy.spatial import cKDTree
    tree = cKDTree(xr)
    near = np.unique(np.concatenate(tree.query_ball_point(xp, cutoff) + [[]]).astype(int)) if len(xp) else np.array([], int)
    if len(near) == 0:
        return 0.0, 0.0
    rj = xr[near]; qj = q[rec_idx][near]; sj = sig[rec_idx][near]; ej = eps[rec_idx][near]
    ve = vv = 0.0
    for a in range(len(pep_idx)):
        d = np.linalg.norm(rj - xp[a], axis=1); m = d < cutoff
        if not m.any():
            continue
        dm = d[m]
        ve += COUL * q[pep_idx][a] * np.sum(qj[m] / dm)
        sij = 0.5 * (sj[m] + sig[pep_idx][a]); eij = np.sqrt(ej[m] * eps[pep_idx][a])
        sr6 = (sij / dm) ** 6
        vv += np.sum(4 * eij * (sr6 ** 2 - sr6))
    return ve * KJ2KCAL, vv * KJ2KCAL


def run_complex(pdb, seq, n_equil=100000, n_frames=250, n_stride=1000):
    """500ps free+bound. Returns (Velec, Vvdw over bound traj, TΔS_conf)."""
    ch = e358.find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    # --- bound: dihedrals + LIE interaction energies ---
    ff, top, pos, system, res_atoms = e358._build(pdb, pep + rec, pep, bound=True)
    q, sig, eps = _nb_params(system)
    pep_idx = np.array([a.index for a in top.atoms() if a.residue.chain.id == pep], int)
    rec_idx = np.array([a.index for a in top.atoms() if a.residue.chain.id != pep], int)
    defs = e358._dihedral_defs(res_atoms); quads = [Q for _, Q in defs]; blab = [l for l, _ in defs]
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    ctx = mm.Context(system, integ, e358.PLAT); ctx.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin); integ.step(n_equil)
    bser = np.zeros((n_frames, len(defs))); ves = np.zeros(n_frames); vvs = np.zeros(n_frames)
    for f in range(n_frames):
        integ.step(n_stride)
        x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        for j, Q in enumerate(quads):
            bser[f, j] = e358._dihedral(x[list(Q)])
        ves[f], vvs[f] = _lie_energy(x, pep_idx, rec_idx, q, sig, eps)
    # --- free: dihedrals only ---
    flab, fser = e358.sample_dihedrals(pdb, pep, pep, bound=False, n_equil=n_equil, n_frames=n_frames, n_stride=n_stride)
    common = [l for l in flab if l in blab]
    fi = {l: i for i, l in enumerate(flab)}; bi = {l: i for i, l in enumerate(blab)}
    dS1 = sum(e358._marg_entropy(fser[:, fi[l]]) - e358._marg_entropy(bser[:, bi[l]]) for l in common)
    def rid(l): return int("".join(c for c in l if c.isdigit()))
    dI = 0.0
    for a in common:
        for b in common:
            if a < b and rid(a) == rid(b):
                dI += e358._mutual_info(fser[:, fi[a]], fser[:, fi[b]]) - e358._mutual_info(bser[:, bi[a]], bser[:, bi[b]])
    tds = (dS1 - dI) * KCAL_PER_NAT
    return float(ves.mean()), float(vvs.mean()), float(tds), len(common)


def _gate(out):
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    if len(out) < 6:
        print(f"only {len(out)}"); return
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    y = np.array([o["y"] for o in out])
    Ve = np.array([o["ve"] for o in out]); Vv = np.array([o["vv"] for o in out]); Ts = np.array([o["tds"] for o in out])
    sc = np.array([_scorer_pred(pep[o["pdb"]]) for o in out])   # deployed scorer prediction
    def loo(F):
        F = np.atleast_2d(F).T if F.ndim == 1 else F
        p = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(F):
            m = LinearRegression().fit(F[tr], y[tr]); p[te] = m.predict(F[te])
        return p
    def rep(tag, F):
        p = loo(F); print(f"  {tag:24s} r={pearsonr(p,y)[0]:+.3f}  MAE={np.mean(np.abs(p-y)):.3f}  RMSE={np.sqrt(np.mean((p-y)**2)):.3f}")
    print(f"\n=== E359 LIE vs ENTROPY  (n={len(out)}, 500ps, calibrated LOO) ===")
    print(f"  Velec mean={Ve.mean():+.1f}  Vvdw mean={Vv.mean():+.1f}  TΔS mean={Ts.mean():+.2f}")
    rep("LIE (Velec,Vvdw)", np.column_stack([Ve, Vv]))
    rep("entropy (TΔS)", Ts)
    rep("LIE + entropy", np.column_stack([Ve, Vv, Ts]))
    rep("scorer alone", sc)
    rep("scorer+LIE+entropy", np.column_stack([sc, Ve, Vv, Ts]))
    print("  corr(TΔS, y)=%+.3f  corr(Velec,y)=%+.3f  corr(Vvdw,y)=%+.3f" % (
        pearsonr(Ts, y)[0], pearsonr(Ve, y)[0], pearsonr(Vv, y)[0]))


_FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
         "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
_SCORER = None


def _scorer_pred(row):
    global _SCORER
    if _SCORER is None:
        import numpy as np, json
        from sklearn.ensemble import HistGradientBoostingRegressor
        rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
        X = np.array([[float(r[f]) for f in _FULL] for r in rows]); Y = np.array([float(r["y"]) for r in rows])
        _SCORER = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0).fit(X, Y)
    return float(_SCORER.predict(np.array([[float(row[f]) for f in _FULL]]))[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=15); a = ap.parse_args()
    rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
    pool = [r for r in rows if 6 <= len(r["seq"]) <= 15]
    if a.smoke:
        pool = pool[:1]
    import random; random.seed(5); random.shuffle(pool)
    out = []
    for i, r in enumerate(pool[:(1 if a.smoke else a.n)]):
        t = time.time()
        try:
            ve, vv, tds, n = run_complex(r["pdb"], r["seq"])
            out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "ve": ve, "vv": vv, "tds": tds, "n": n})
            print(f"[{i+1}] {r['pdb']} Velec={ve:+.1f} Vvdw={vv:+.1f} TΔS={tds:+.2f} ({(time.time()-t)/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"[{i+1}] {r['pdb']} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
        json.dump(out, open("data/e359_lie_entropy.json", "w"))
    if a.gate:
        _gate(out)


if __name__ == "__main__":
    main()
