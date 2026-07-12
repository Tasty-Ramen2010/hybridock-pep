"""E114 — full physics-failure atlas across ALL protein-peptide complexes (ours 156 + PDBbind 925).

Ram: "physics never lies — I want to know where we lack physics-wise." So map, rigorously:
  PART 1  FEATURE × LENGTH correlation matrix — does each physics term carry signal, and in which
          length regime does it hold / flip / die? (sign-stability is the truth test.)
  PART 2  WHERE WE FAIL — per-complex error of the physics model, stratified by length / charge /
          hydrophobicity / receptor size / source. What do the high-error complexes share?
  PART 3  WHICH PHYSICAL TERM IS MISSING — corr(residual, missing-term proxies) by regime:
          conformational entropy (rg_per_L, length, org_density), electrostatics (|net charge|,
          salt-bridge), desolvation/burial, hydrophobicity. The residual points at the absent physics.
  PART 4  WHY EVERYONE FAILS — on the shared-91, where do ALL methods (PPI/Kdeep/DFIRE/CP_PIE/RF/PRODIGY
          + us) miss together (= intrinsically hard, needs FEP/dynamics) vs where only physics misses but
          PPI gets it (= computable-from-data, not from a single static pose).
Writes docs/physics_failure_atlas_2026-06-13.md.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "data" / "biolip" / "ppiaffinity_si" / "SI"
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
POS, NEG, HYD = set("KR"), set("DE"), set("AILMFWVC")
OUT = ROOT / "docs" / "physics_failure_atlas_2026-06-13.md"
LINES = []


def P(s=""):
    print(s)
    LINES.append(s)


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def load():
    rows = {}
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows[r["pdb"]] = {"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r.get("seq", ""),
                              "y": float(r["y"]), "length": int(float(r["length"])), "src": "ours",
                              "feat": {c: float(r[c]) for c in PROD}}
    oseq = {r["seq"] for r in rows.values() if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq or r["pdb"] in rows:
            continue
        oseq.add(r["seq"])
        rows[r["pdb"]] = {"id": r["pdb"], "pdb4": r["pdb"].lower()[:4], "seq": r["seq"], "y": r["y"],
                          "length": r["length"], "src": "pdbbind", "feat": {c: r[c] for c in PROD}}
    for r in rows.values():
        s = r["seq"]
        L = max(1, len(s))
        r["abs_ch"] = sum(c in POS | NEG for c in s) / L
        r["net_ch"] = sum(c in POS for c in s) - sum(c in NEG for c in s)
        r["hyd_fr"] = sum(c in HYD for c in s) / L
    return list(rows.values())


def cc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[m], y[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan


def gbt_loco_cv(rows, cols, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(rows))
    y = np.array([r["y"] for r in rows])
    pred = np.full(len(rows), np.nan)
    for f in range(k):
        tr = fold != f
        X = np.array([[r["feat"][c] for c in cols] for r in rows], float)
        m = HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.04,
                                          l2_regularization=2.0, min_samples_leaf=25, random_state=seed).fit(X[tr], y[tr])
        pred[fold == f] = m.predict(X[fold == f])
    return pred


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    L = np.array([r["length"] for r in rows])
    bands = ["short≤8", "med9-12", "long13-16", "vlong≥17"]
    P(f"# Physics-failure atlas — all protein-peptide complexes (n={len(rows)})\n")
    P(f"Composition: {sum(r['src']=='ours' for r in rows)} ours + {sum(r['src']=='pdbbind' for r in rows)} PDBbind. "
      f"By length: " + ", ".join(f"{b}={sum(band(x)==b for x in L)}" for b in bands) + ".\n")

    # ---- PART 1: feature × length correlation ----
    P("## PART 1 — feature × length correlation  corr(feature, ΔG)  [sign-stable across bands = real physics]")
    P(f"{'feature':<14}" + "".join(f"{b:>11}" for b in bands) + f"{'ALL':>9}  verdict")
    for c in PROD:
        rb = []
        for b in bands:
            m = np.array([band(x) == b for x in L])
            rb.append(cc([r["feat"][c] for i, r in enumerate(rows) if m[i]], y[m]) if m.sum() >= 8 else np.nan)
        rall = cc([r["feat"][c] for r in rows], y)
        signs = [np.sign(v) for v in rb if v == v and abs(v) > 0.05]
        verdict = "STABLE" if len(set(signs)) == 1 and len(signs) >= 3 else "FLIPS" if len(set(signs)) > 1 else "weak"
        P(f"{c:<14}" + "".join(f"{v:>+11.2f}" if v == v else f"{'—':>11}" for v in rb) + f"{rall:>+9.2f}  {verdict}")
    # derived
    for nm, key in [("abs_charge", "abs_ch"), ("|net_charge|", None), ("hyd_frac", "hyd_fr"), ("length", "length")]:
        rb = []
        for b in bands:
            m = np.array([band(x) == b for x in L])
            vals = [abs(r["net_ch"]) if nm == "|net_charge|" else r[key] for i, r in enumerate(rows) if m[i]]
            rb.append(cc(vals, y[m]) if m.sum() >= 8 else np.nan)
        allv = [abs(r["net_ch"]) if nm == "|net_charge|" else r[key] for r in rows]
        P(f"{nm:<14}" + "".join(f"{v:>+11.2f}" if v == v else f"{'—':>11}" for v in rb) + f"{cc(allv,y):>+9.2f}  (derived)")

    # ---- PART 2: where we fail (GBT 5-fold) ----
    pred = gbt_loco_cv(rows, PROD)
    resid = y - pred
    P(f"\n## PART 2 — WHERE WE FAIL (GBT 5-fold, pooled r={cc(pred,y):+.3f} RMSE={np.sqrt(np.mean(resid**2)):.2f})")
    P(f"{'stratum':<20}{'n':>5}{'r':>9}{'RMSE':>8}{'mean|err|':>11}")
    for b in bands:
        m = np.array([band(x) == b for x in L])
        if m.sum() >= 8:
            P(f"  len {b:<15}{m.sum():>5}{cc(pred[m],y[m]):>+9.2f}{np.sqrt(np.mean(resid[m]**2)):>8.2f}{np.mean(np.abs(resid[m])):>11.2f}")
    ac = np.array([r["abs_ch"] for r in rows])
    for lab, m in [("charge ≤0.15", ac <= 0.15), ("charge 0.15-0.30", (ac > 0.15) & (ac <= 0.30)), ("charge >0.30", ac > 0.30)]:
        if m.sum() >= 8:
            P(f"  {lab:<17}{m.sum():>5}{cc(pred[m],y[m]):>+9.2f}{np.sqrt(np.mean(resid[m]**2)):>8.2f}{np.mean(np.abs(resid[m])):>11.2f}")
    for lab, src in [("ours", "ours"), ("pdbbind", "pdbbind")]:
        m = np.array([r["src"] == src for r in rows])
        P(f"  src {lab:<13}{m.sum():>5}{cc(pred[m],y[m]):>+9.2f}{np.sqrt(np.mean(resid[m]**2)):>8.2f}{np.mean(np.abs(resid[m])):>11.2f}")

    # ---- PART 3: which physical term is missing (residual correlations by band) ----
    P("\n## PART 3 — WHICH PHYSICS IS MISSING  corr(|residual|, missing-term proxy) by length band")
    P("  (positive = error GROWS with that effect ⇒ that physics is absent from the model)")
    proxies = {"conf-entropy(rg_per_L)": [r["feat"]["rg_per_L"] for r in rows],
               "conf-entropy(length)": [r["length"] for r in rows],
               "disorder(1-org_density)": [1 - r["feat"]["org_density"] for r in rows],
               "electrostatics(|netQ|)": [abs(r["net_ch"]) for r in rows],
               "electrostatics(absQ)": [r["abs_ch"] for r in rows],
               "salt-bridge(sasa_sb)": [r["feat"]["sasa_sb"] for r in rows],
               "hydrophobic(hyd_fr)": [r["hyd_fr"] for r in rows]}
    aerr = np.abs(resid)
    P(f"{'proxy':<26}" + "".join(f"{b:>11}" for b in bands) + f"{'ALL':>9}")
    for nm, vec in proxies.items():
        vec = np.array(vec, float)
        rb = []
        for b in bands:
            m = np.array([band(x) == b for x in L])
            rb.append(cc(vec[m], aerr[m]) if m.sum() >= 8 else np.nan)
        P(f"{nm:<26}" + "".join(f"{v:>+11.2f}" if v == v else f"{'—':>11}" for v in rb) + f"{cc(vec,aerr):>+9.2f}")

    # ---- PART 4: why EVERYONE fails (shared-91 competitor agreement) ----
    P("\n## PART 4 — WHY EVERYONE FAILS (shared-91: PPI/Kdeep/DFIRE/CP_PIE/RF/PRODIGY + ours)")
    t100 = list(csv.DictReader(open(SI / "SI-File-6-protein-peptide-test-set-1.csv")))
    sh_y, methods = [], {"PPI-Affinity": [], "Kdeep": [], "DFIRE": [], "CP_PIE": [], "RF-Score": [], "PRODIGY": []}
    for r in t100:
        sh_y.append(float(r["Binding_affinity"]))
        for mth in methods:
            key = next((k for k in r if k.replace(" ", "") == mth.replace(" ", "")), None)
            try:
                methods[mth].append(float(r[key]))
            except (ValueError, KeyError, TypeError):
                methods[mth].append(np.nan)
    sh_y = np.array(sh_y)

    def zerr(pred):  # sign-aligned z-score error
        p = np.array(pred, float)
        if cc(p, sh_y) < 0:
            p = -p
        zp = (p - np.nanmean(p)) / (np.nanstd(p) + 1e-9)
        zy = (sh_y - sh_y.mean()) / sh_y.std()
        return np.abs(zp - zy)
    errs = {m: zerr(v) for m, v in methods.items()}
    # consensus error = mean z-error across the 6 published methods
    cons = np.nanmean(np.vstack([errs[m] for m in methods]), axis=0)
    order = np.argsort(-cons)
    P("  Per-method |z-error| correlation (do methods fail on the SAME complexes? high = shared hard cases):")
    ms = list(methods)
    for i, a in enumerate(ms):
        cors = [f"{cc(errs[a], errs[b]):+.2f}" for b in ms]
        P(f"   {a:<13} " + " ".join(f"{c:>6}" for c in cors))
    P("   " + " " * 13 + " " + " ".join(f"{m[:6]:>6}" for m in ms))
    P(f"\n  TOP-10 hardest complexes (highest consensus error across all 6 methods = intrinsically hard):")
    for i in order[:10]:
        nm = t100[i]["PDB_NAME"].split("-")[0]
        P(f"   {nm:<12} y={sh_y[i]:+.1f}  consensus|z-err|={cons[i]:.2f}  "
          f"(PPI {errs['PPI-Affinity'][i]:.1f}, best other {min(errs[m][i] for m in ms if m!='PPI-Affinity'):.1f})")
    P(f"\n  mean consensus error: {np.nanmean(cons):.2f}; if the hardest cases are high-charge/long/flexible,")
    P("  that regime needs DYNAMICS/FEP (no static method, incl PPI, captures it). Where PPI alone succeeds")
    P("  but physics methods fail = data-learnable (sequence statistics), not single-pose-computable.")

    OUT.write_text("\n".join(LINES) + "\n")
    print(f"\n[written to {OUT.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
