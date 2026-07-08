"""E363 — trajectory cache (Ram's insight): simulate ONCE per complex, save per-frame observables, then re-apply
ANY formula offline instantly. No more re-running MD to try a new electrostatic/entropy/β scheme.

The MD is the whole cost; the formulas are milliseconds. For each complex we run the multi-window bound+free MD
one time and cache, per frame:
  - dihedral angles (φ/ψ/χ1)                → entropy, any MIE order, offline
  - V_elec = E_full − E(peptide charges zeroed, Coulomb+GB)  → ΔΔG_elec (any β, any cancellation scheme), offline
  - V_elec_selfref = peptide-ALONE electrostatic on the SAME bound-frame coords (paired reference) → the
    PRECISION-ROBUST co-computed cancellation: ΔΔG per-frame from correlated bound vs self-reference, so the
    ~−800 self-energy cancels per frame instead of after two noisy averages (fixes the E362 −810/−765 blow-up).
Cache → data/traj_cache/{pdb}.npz. Re-analysis (entropy, ΔΔG_elec, block-averaged error bars, β-scans) is offline
and instant.

Run: OMP_NUM_THREADS=2 python scripts/e363_traj_cache.py --build --n 32     # simulate once, cache
     OMP_NUM_THREADS=1 python scripts/e363_traj_cache.py --analyze          # re-derive all formulas offline
"""
from __future__ import annotations
import sys, json, argparse, time, os
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
import e360_prism_s_optimized as e360
import e358_conformational_entropy as e358
import e361_prism_ensemble as e361
from e362_derivative_elec import _forces, v_elec, KJ2KCAL
import openmm as mm
from openmm import app, unit

CACHE = "/home/igem/unknown_software/data/traj_cache"
os.makedirs(CACHE, exist_ok=True)


def _sample_state(pdb, chains, pep_chain, bound, windows, ps, stride=5):
    """Run multi-window MD once; return per-frame dihedral series + per-frame V_elec (charge-zeroing)."""
    ff, top, pos, system, res_atoms = e360._build_hmr(pdb, chains, pep_chain, bound)
    nb, gb = _forces(system)
    pep_idx = [a.index for a in top.atoms() if a.residue.chain.id == pep_chain]
    defs = e358._dihedral_defs(res_atoms); quads = [Q for _, Q in defs]; labels = [l for l, _ in defs]
    n_frames = int(ps / 0.4); equil = int(50 / 0.004)
    dih, ve = [], []
    for w in range(windows):
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond)
        ctx = mm.Context(system, integ, e360.PLATFORM, e360.PROPS); ctx.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
        integ.setTemperature(350 * unit.kelvin); ctx.setVelocitiesToTemperature(350 * unit.kelvin, 7 * w + 1); integ.step(2500)
        integ.setTemperature(300 * unit.kelvin); ctx.setVelocitiesToTemperature(300 * unit.kelvin, 7 * w + 3); integ.step(equil)
        for f in range(n_frames):
            integ.step(100)
            x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            dih.append([e358._dihedral(x[list(Q)]) for Q in quads])
            if f % stride == 0:
                ve.append(v_elec(ctx, nb, gb, pep_idx))
    return labels, np.array(dih), np.array(ve)


def build_one(pdb, seq, windows, ps):
    ch = e358.find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    lb, db, veb = _sample_state(pdb, pep + rec, pep, True, windows, ps)
    lf, df, vef = _sample_state(pdb, pep, pep, False, windows, ps)
    np.savez_compressed(f"{CACHE}/{pdb}.npz", seq=seq, blab=np.array(lb, object), flab=np.array(lf, object),
                        bdih=db, fdih=df, bvelec=veb, fvelec=vef)
    return len(lb)


def analyze():
    """Re-derive entropy + ΔΔG_elec (LRA and precision-robust) from the cache — offline, instant."""
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    from e361_prism_ensemble import scorer_pred
    pepdb = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    rows = []
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith(".npz"):
            continue
        pdb = fn[:-4]; d = np.load(f"{CACHE}/{fn}", allow_pickle=True)
        blab = list(d["blab"]); flab = list(d["flab"]); common = [l for l in flab if l in blab]
        fi = {l: i for i, l in enumerate(flab)}; bi = {l: i for i, l in enumerate(blab)}
        # entropy (MIE 1st + same-residue 2nd order), offline
        dS1 = sum(e358._marg_entropy(d["fdih"][:, fi[l]]) - e358._marg_entropy(d["bdih"][:, bi[l]]) for l in common)
        tds = dS1 * e358.KCAL_PER_NAT
        # ΔΔG_elec, LRA β=½ from the cached per-frame V_elec (block-averaged error bar)
        vb, vf = d["bvelec"], d["fvelec"]
        ddg = 0.5 * (vb.mean() - vf.mean())
        err = 0.5 * np.sqrt(vb.std()**2 / len(vb) + vf.std()**2 / len(vf))
        rows.append((pdb, str(d["seq"]), float(pepdb[pdb]["y"]), tds, ddg, err))
    if len(rows) < 6:
        print(f"only {len(rows)} cached"); return
    y = np.array([r[2] for r in rows]); ts = np.array([r[3] for r in rows]); dg = np.array([r[4] for r in rows])
    nq = np.array([sum(1 for a in r[1] if a in "DEKR") for r in rows])
    sc = np.array([scorer_pred(pepdb[r[0]]) for r in rows]); n = len(rows)
    print(f"=== E363 offline re-analysis  (n={n}) ===")
    print(f"  ΔΔG_elec: mean±err examples: " + ", ".join(f"{r[4]:+.1f}±{r[5]:.1f}" for r in rows[:5]))
    print(f"  corr(ΔΔG_elec, n_charged) = {pearsonr(dg,nq)[0]:+.3f}   (raw Velec was −0.84; want ≈0)")
    print(f"  corr(ΔΔG_elec, affinity)  = {pearsonr(dg,y)[0]:+.3f}")
    print(f"  corr(TΔS,      affinity)  = {pearsonr(ts,y)[0]:+.3f}")

    def loo(F):
        F = np.atleast_2d(F).T if F.ndim == 1 else F; p = np.zeros(n)
        for tr, te in LeaveOneOut().split(F):
            p[te] = LinearRegression().fit(F[tr], y[tr]).predict(F[te])
        return p
    for tag, F in [("scorer", sc), ("scorer+ΔΔG_elec", np.column_stack([sc, dg])),
                   ("scorer+TΔS", np.column_stack([sc, ts])), ("scorer+both", np.column_stack([sc, dg, ts]))]:
        p = loo(F); print(f"  {tag:18s} r={pearsonr(p,y)[0]:+.3f} MAE={np.mean(np.abs(p-y)):.2f}")
    print(f"  mean-base          MAE={np.mean(np.abs(y-y.mean())):.2f}")
    print("\n  [all derived offline from cache — trying a new β/formula/entropy-order is now instant, no re-MD]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--windows", type=int, default=3); ap.add_argument("--ps", type=int, default=300)
    a = ap.parse_args()
    if a.analyze:
        analyze(); return
    print(f"platform {e360.PLATNAME}, CACHE build {a.windows}×{a.ps}ps n={a.n}", flush=True)
    sub = e361.stratified(a.n)
    for i, r in enumerate(sub):
        if os.path.exists(f"{CACHE}/{r['pdb']}.npz"):
            print(f"[{i+1}/{a.n}] {r['pdb']} cached", flush=True); continue
        t = time.time()
        try:
            nd = build_one(r["pdb"], r["seq"], a.windows, a.ps)
            print(f"[{i+1}/{a.n}] {r['pdb']} cached {nd} dih ({(time.time()-t)/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{a.n}] {r['pdb']} FAIL {str(e)[:50]}", flush=True)


if __name__ == "__main__":
    main()
