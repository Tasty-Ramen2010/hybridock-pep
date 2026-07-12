"""E38 — Ram's hypothesis: length-MODULATED weights + conformational entropy penalty.

DIAGNOSIS: length corr with ΔG = +0.43 (crystal-65) vs -0.40 (the-98) — FLIPS. Length isn't
a clean predictor; it modulates everything and flips across peptide populations. The MISSING
PHYSICS is the conformational entropy penalty: freezing a flexible peptide on binding costs
~T·ΔS per ordered residue. In a narrow well-folded set (crystal-65) this term is ~constant so
longer=more contacts=stronger; in a diverse set spanning ordered->disordered (the-98) the
entropy penalty varies and long disordered peptides bind WEAKER. We never modeled it.

Test three physically-grounded fixes:
  A) explicit entropy term: ΔG += +s_ent · n_ordered  (n_ordered = residues with >threshold burial)
  B) length-MODULATED features: x_i, x_i·log(L), x_i/L  (effective weight varies with L)
  C) per-residue intensive favorable - per-residue entropy (the proper decomposition)
Evaluate GENERALIZATION (cross-dataset transfer) — does length-aware physics stop the flip?
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scipy.stats import pearsonr  # noqa: E402

# Build feature rows: intensive features + length + n_ordered (buried-residue count proxy)
inten = json.loads(Path("/tmp/e31_intensive.json").read_text())
e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
b98raw = json.loads(Path("/tmp/e28_feats.json").read_text())


def attach(rows_inten, lengths):
    out = []
    for r, L in zip(rows_inten, lengths):
        d = dict(r)
        d["L"] = float(L)
        d["logL"] = float(np.log(L))
        # n_ordered proxy: total buried hydrophobic SASA already ~ ordered interface;
        # use bsa_hyd (extensive) as the "amount frozen" and L for the entropy count
        d["entropy_pen"] = float(L)  # linear conformational entropy (per-residue freeze)
        out.append(d)
    return out


Lcr = [geo[p.upper()].get("L") or len(geo[p.upper()]["seq"]) if "seq" in geo[p.upper()] else 12
       for p in [g["pdb"] for g in json.loads(Path("/tmp/e19_cr.json").read_text())]]
# crystal lengths from benchmark
bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
cr_rows = json.loads(Path("/tmp/e19_cr.json").read_text())
Lcr = [bench[r["pdb"].upper()]["peptide_len"] for r in cr_rows]
L98 = [v["L"] for v in b98raw.values()]
cr = attach(inten["cr"], Lcr)
b98 = attach(inten["b98"], L98)
ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])

UNI = ["bsa_hyd", "mj_per_contact", "f_hyd_iface", "frac_pol_satisfied"]


def mat(rows, feats):
    return np.array([[r.get(f, 0.0) for f in feats] for r in rows])


def loo(rows, feats):
    y = np.array([r["y"] for r in rows]); X = mat(rows, feats); p = np.zeros(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[i] = np.r_[1, (X[i] - mu) / sd] @ w
    return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())


def transfer(tr, te, feats):
    Xtr, ytr = mat(tr, feats), np.array([r["y"] for r in tr])
    Xte, yte = mat(te, feats), np.array([r["y"] for r in te])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pr = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
    return pearsonr(pr, yte).statistic


# build interaction (length-modulated) features
def add_interactions(rows):
    for r in rows:
        for f in UNI:
            r[f + "_xlogL"] = r.get(f, 0.0) * r["logL"]
            r[f + "_perL"] = r.get(f, 0.0) / r["L"]
    return rows


cr = add_interactions(cr); b98 = add_interactions(b98)
INTER = [f + "_xlogL" for f in UNI]
PERL = [f + "_perL" for f in UNI]

print("=== generalization (cr<->98 transfer) — does length-aware physics stop the flip? ===")
print(f"  {'feature set':<40}{'cr->98':>9}{'98->cr':>9}{'pool LOO':>10}")
sets = {
    "universal intensive (4) [baseline]": UNI,
    "+ entropy penalty (L)": UNI + ["entropy_pen"],
    "+ logL": UNI + ["logL"],
    "length-modulated (x·logL)": UNI + INTER,
    "per-residue (x/L)": PERL,
    "intensive + per-residue + L": UNI + PERL + ["L"],
    "FULL length-aware": UNI + INTER + ["entropy_pen", "logL"],
}
pool = cr + b98
for nm, fs in sets.items():
    a = transfer(cr, b98, fs); b = transfer(b98, cr, fs); pr, _ = loo(pool, fs)
    print(f"  {nm:<40}{a:>+9.3f}{b:>+9.3f}{pr:>+10.3f}")
print(f"\n  baseline universal pool LOO was 0.421; old extensive transfer was -0.14")
print("  >> if a length-modulated/entropy set lifts BOTH transfer directions, Ram's hypothesis is the fix")
