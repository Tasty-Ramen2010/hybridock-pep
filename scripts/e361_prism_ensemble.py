"""E361 — PRISM ensemble scorer (OUR own, not LIE): ⟨Velec⟩ + multi-window conformational entropy, wide-range test.

E359 taught us: the only ensemble term that carried real signal was ⟨Velec⟩ (raw corr +0.66); Vvdw and the full
fitted LIE were noise, and single-window entropy was noise. So we KEEP ⟨Velec⟩ (ensemble peptide–receptor
electrostatic, a genuine physics term) and pair it with the CONVERGED multi-window conformational entropy (E360),
dropping LIE's Vvdw/coefficient-fitting. And we test on a WIDE-RANGE, stratified set (E359's random subset was
too narrow, std 1.1 — nothing beats the mean there).

Per complex: 3×300ps multi-window bound + free MD (HMR 4fs, auto platform). Bound windows also collect ⟨Velec⟩.
Test, leave-one-out on the wide-range set:
  scorer | scorer+Velec | scorer+entropy | scorer+Velec+entropy
against experimental ΔG — does adding OUR ensemble terms beat the single-pose scorer on a proper range?

Run: OMP_NUM_THREADS=2 python scripts/e361_prism_ensemble.py --gate --n 32 --windows 3 --ps 300
"""
from __future__ import annotations
import sys, json, argparse, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
import e360_prism_s_optimized as e360
import e358_conformational_entropy as e358
from e359_lie_vs_entropy import _nb_params, _lie_energy
import openmm as mm
from openmm import unit


def sample_bound_with_velec(pdb, chains, pep_chain, windows, ps):
    """Multi-window bound MD; returns (labels, pooled dihedrals, mean Velec over all frames)."""
    ff, top, pos, system, res_atoms = e360._build_hmr(pdb, chains, pep_chain, bound=True)
    q, sig, eps = _nb_params(system)
    pep_idx = np.array([a.index for a in top.atoms() if a.residue.chain.id == pep_chain], int)
    rec_idx = np.array([a.index for a in top.atoms() if a.residue.chain.id != pep_chain], int)
    defs = e358._dihedral_defs(res_atoms)
    if not defs:
        raise RuntimeError("no dihedrals")
    quads = [Q for _, Q in defs]; labels = [l for l, _ in defs]
    n_frames = int(ps / 0.4); equil = int(50 / 0.004)
    pooled, ve_all = [], []
    for w in range(windows):
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond)
        ctx = mm.Context(system, integ, e360.PLATFORM, e360.PROPS); ctx.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=500)
        integ.setTemperature(400 * unit.kelvin); ctx.setVelocitiesToTemperature(400 * unit.kelvin, w + 1); integ.step(2500)
        integ.setTemperature(300 * unit.kelvin); ctx.setVelocitiesToTemperature(300 * unit.kelvin, w + 7); integ.step(equil)
        ser = np.zeros((n_frames, len(defs)))
        for f in range(n_frames):
            integ.step(100)
            x = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            for j, Q in enumerate(quads):
                ser[f, j] = e358._dihedral(x[list(Q)])
            ve, _ = _lie_energy(x, pep_idx, rec_idx, q, sig, eps)   # Velec only (the good term)
            ve_all.append(ve)
        pooled.append(ser)
    return labels, np.concatenate(pooled, 0), float(np.mean(ve_all))


def run_complex(pdb, seq, windows, ps):
    ch = e358.find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no chains")
    pep, rec = ch
    lb, sb, velec = sample_bound_with_velec(pdb, pep + rec, pep, windows, ps)
    lf, sf = e360.sample_multiwindow(pdb, pep, pep, False, windows, ps)
    common = [l for l in lf if l in lb]
    if not common:
        raise RuntimeError("no common dih")
    fi = {l: i for i, l in enumerate(lf)}; bi = {l: i for i, l in enumerate(lb)}
    dS1 = sum(e358._marg_entropy(sf[:, fi[l]]) - e358._marg_entropy(sb[:, bi[l]]) for l in common)
    def rid(l): return int("".join(c for c in l if c.isdigit()))
    dI = sum(e358._mutual_info(sf[:, fi[a]], sf[:, fi[b]]) - e358._mutual_info(sb[:, bi[a]], sb[:, bi[b]])
             for a in common for b in common if a < b and rid(a) == rid(b))
    tds = (dS1 - dI) * e358.KCAL_PER_NAT
    return velec, tds, len(common)


_FULL = ["poc_n","poc_f_hyd","poc_f_arom","poc_net","poc_eis","bsa_hyd","sasa_hb","sasa_sb","arom_cc",
         "hb_count","mj_contact","strength_bur","rg_per_L","org_density","cys_frac","mean_burial"]
_SC = None


def scorer_pred(row):
    global _SC
    if _SC is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        rows = [json.loads(x) for x in open("data/pdbbind_peptides.jsonl")]
        X = np.array([[float(r[f]) for f in _FULL] for r in rows]); Y = np.array([float(r["y"]) for r in rows])
        _SC = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0).fit(X, Y)
    return float(_SC.predict(np.array([[float(row[f]) for f in _FULL]]))[0])


def stratified(n):
    """Pick n complexes spread evenly across the affinity range (wide range, unlike E359's random subset)."""
    rows = [r for r in (json.loads(x) for x in open("data/pdbbind_peptides.jsonl")) if 6 <= len(r["seq"]) <= 15]
    rows.sort(key=lambda r: float(r["y"]))
    idx = np.linspace(0, len(rows) - 1, n).astype(int)
    return [rows[i] for i in idx]


def _gate(out):
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open("data/pdbbind_peptides.jsonl")}
    y = np.array([o["y"] for o in out]); ve = np.array([o["velec"] for o in out]); ts = np.array([o["tds"] for o in out])
    sc = np.array([scorer_pred(pep[o["pdb"]]) for o in out])
    n = len(y)
    def loo(F):
        F = np.atleast_2d(F).T if F.ndim == 1 else F; p = np.zeros(n)
        for tr, te in LeaveOneOut().split(F):
            p[te] = LinearRegression().fit(F[tr], y[tr]).predict(F[te])
        return p
    print(f"\n=== E361 PRISM ensemble  (n={n}, wide-range y std={y.std():.2f} [{y.min():.1f},{y.max():.1f}]) ===")
    print(f"  raw: corr(Velec,y)={pearsonr(ve,y)[0]:+.3f}  corr(TΔS,y)={pearsonr(ts,y)[0]:+.3f}")
    for tag, F in [("scorer", sc), ("scorer+Velec", np.column_stack([sc, ve])),
                   ("scorer+entropy", np.column_stack([sc, ts])),
                   ("scorer+Velec+entropy", np.column_stack([sc, ve, ts]))]:
        p = loo(F); print(f"  {tag:22s} r={pearsonr(p,y)[0]:+.3f}  MAE={np.mean(np.abs(p-y)):.2f}  RMSE={np.sqrt(np.mean((p-y)**2)):.2f}")
    print(f"  mean-base              MAE={np.mean(np.abs(y-y.mean())):.2f}  RMSE={y.std():.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--windows", type=int, default=3); ap.add_argument("--ps", type=int, default=300)
    a = ap.parse_args()
    print(f"platform {e360.PLATNAME}, {a.windows}×{a.ps}ps HMR-4fs, STRATIFIED wide-range n={a.n}", flush=True)
    sub = stratified(a.n)
    out = []
    for i, r in enumerate(sub):
        t = time.time()
        try:
            ve, tds, nn = run_complex(r["pdb"], r["seq"], a.windows, a.ps)
            out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "velec": ve, "tds": tds, "n": nn})
            print(f"[{i+1}/{a.n}] {r['pdb']} y={float(r['y']):.1f} Velec={ve:+.1f} TΔS={tds:+.2f} ({(time.time()-t)/60:.1f}m)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{a.n}] {r['pdb']} FAIL {str(e)[:50]}", flush=True)
        json.dump(out, open("data/e361_prism_ensemble.json", "w"))
    if a.gate and len(out) >= 8:
        _gate(out)


if __name__ == "__main__":
    main()
