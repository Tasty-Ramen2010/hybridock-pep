"""Per-complex diagnosis of the ensemble-MM-GBSA test. Every win/loss is training data:
characterize each complex by the physical variables that will become ML features, fit
<E_int> -> experimental dG, and read each residual physically."""
import json
import numpy as np
from scipy.stats import pearsonr

out = json.load(open("/tmp/e49_ens_mmgbsa.json"))
bench = {r["pdb"].upper(): r for r in json.load(open("data/benchmark_crystal.json"))}


def netchg(s):
    return s.count("K") + s.count("R") - s.count("D") - s.count("E")


rows = []
for k, v in out.items():
    seq = bench[k].get("peptide_seq", "")
    blow = abs(v["dg_single"]) > 1e4
    rows.append(dict(pdb=k, seq=seq, L=len(seq), cf=v["cf"], nq=netchg(seq),
                     y=v["y"], single=v["dg_single"], eint=v["e_int_mean"],
                     std=v["e_int_std"], mts=v["minus_tds"], blow=blow))

# Fit <E_int> -> y on NON-blowup complexes (single-pose blowups excluded from the fit;
# they are automatically 'ensemble rescued' wins).
good = [r for r in rows if not r["blow"]]
ev = np.array([r["eint"] for r in good]); yv = np.array([r["y"] for r in good])
a, b = np.polyfit(ev, yv, 1)                      # y ~ a*<E_int> + b
for r in rows:
    r["pred"] = a * r["eint"] + b
    r["resid"] = r["pred"] - r["y"]               # >0 predicted too weak; <0 too strong

print(f"=== n={len(rows)} done | charged(cf>=0.3)={sum(r['cf']>=0.3 for r in rows)} ===")
rE = pearsonr([r["eint"] for r in good], yv).statistic
rS = pearsonr([min(r['single'],1e4) for r in good], yv).statistic
print(f"<E_int> r={rE:+.3f} vs single(clip) r={rS:+.3f}  | fit y={a:+.3f}*<E_int>{b:+.1f}\n")

print(f"{'pdb':<6}{'seq':<16}{'L':>3}{'cf':>5}{'nq':>4}{'y':>7}{'single':>9}{'<Eint>':>8}"
      f"{'std':>6}{'-TdS':>7}{'pred':>7}{'res':>6}  diagnosis")
for r in sorted(rows, key=lambda r: -r["cf"]):
    if r["blow"]:
        diag = "single-pose CLASH BLOWUP -> ensemble RESCUED (huge win)"
    elif abs(r["resid"]) < 1.0:
        diag = "tight fit"
        if r["cf"] >= 0.3:
            diag += " | CHARGED predicted well (salt-bridge averaging worked)"
        elif r["std"] > 4:
            diag += " | floppy but ranked"
    elif abs(r["resid"]) < 2.5:
        diag = "ok"
    else:
        if r["resid"] > 0:
            diag = "OUTLIER: predicted too WEAK"
            if r["std"] > 4.5:
                diag += " (very floppy -> <E_int> under-counts; entropy/sampling)"
            elif r["cf"] >= 0.3:
                diag += " (charged -> GB electrostatics still off; polarization?)"
        else:
            diag = "OUTLIER: predicted too STRONG"
            if r["mts"] > 8:
                diag += " (big entropy penalty not subtracted -> add -TdS)"
            elif r["cf"] >= 0.3:
                diag += " (charged, ensemble over-stabilized salt bridge)"
    sg = f"{r['single']:>9.1f}" if not r["blow"] else f"{'1e9+':>9}"
    print(f"{r['pdb']:<6}{r['seq'][:15]:<16}{r['L']:>3}{r['cf']:>5.2f}{r['nq']:>4}{r['y']:>7.1f}"
          f"{sg}{r['eint']:>8.1f}{r['std']:>6.1f}{r['mts']:>7.2f}{r['pred']:>7.1f}{r['resid']:>+6.1f}  {diag}")

# Pattern summary: which physical features predict the residual? (the ML feature scout)
print("\n=== residual drivers (what our ML model must encode) ===")
gg = [r for r in rows if not r["blow"]]
for feat in ["cf", "nq", "L", "std", "mts"]:
    v = np.array([r[feat] for r in gg]); res = np.array([r["resid"] for r in gg])
    if v.std() > 0:
        print(f"  corr(residual, {feat:<4}) = {pearsonr(res, v).statistic:+.3f}")
print("  (+ = that feature makes <E_int> predict too weak; - = too strong)")
