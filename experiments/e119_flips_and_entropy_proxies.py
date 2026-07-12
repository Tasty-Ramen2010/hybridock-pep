"""E119 — (A) WHY do features flip across length/dataset?  (B) non-GPU conformational-entropy proxies.

A. FLIP MECHANISM: hypothesis = EXTENSIVE features (counts/sums/areas that grow with peptide size) are
   size-confounded. Within a length band the size variance shrinks, and the band's length↔ΔG relation
   differs by subset → the raw feature's sign flips (Simpson). Test by PARTIAL correlation controlling
   for length, and by comparing raw (extensive) vs per-residue (intensive) versions of each feature.
B. NON-GPU ENTROPY PROXIES: cheap surrogates for the conformational entropy the atlas says is missing
   (no MD): #rotatable side-chain bonds, Gly+Pro fraction, flexibility scale, 1/org_density, length,
   side-chain configurational entropy (Abagyan-style per-residue). Correlate each with ΔG (and, when
   e115 s_free is ready, with the real MD s_free) to find a deployable no-GPU entropy feature.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
# Abagyan/Pickett side-chain configurational entropy (cal/mol/K, relative); Gly/Ala/Pro low, Lys/Arg/Met high
SC_ENTROPY = {"A": 0.0, "G": 0.0, "P": 0.0, "S": 3.5, "C": 3.5, "T": 3.5, "V": 1.7, "D": 5.0, "N": 5.0,
              "I": 5.0, "L": 5.2, "E": 7.1, "Q": 7.1, "M": 8.0, "F": 5.5, "Y": 5.9, "W": 5.9, "H": 6.2,
              "K": 9.0, "R": 9.3}
ROT = {"A": 0, "G": 0, "P": 0, "S": 1, "C": 1, "T": 1, "V": 1, "D": 2, "N": 2, "I": 2, "L": 2, "F": 2,
       "Y": 2, "H": 2, "E": 3, "Q": 3, "M": 3, "W": 2, "K": 4, "R": 4}
FLEX = {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.50, "G": 0.54, "H": 0.32,
        "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.30, "F": 0.31, "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31,
        "Y": 0.42, "V": 0.39}


def seqhash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def load():
    rows = []
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            rows.append({"seq": r.get("seq", ""), "y": float(r["y"]), "length": int(float(r["length"])),
                         "feat": {c: float(r[c]) for c in PROD}})
    oseq = {r["seq"] for r in rows if r["seq"]}
    for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["seq"] in oseq:
            continue
        oseq.add(r["seq"])
        rows.append({"seq": r["seq"], "y": r["y"], "length": r["length"], "feat": {c: r[c] for c in PROD}})
    return rows


def pcorr(x, y, z):
    """partial corr(x,y | z) — removes linear dependence on z (length)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearsonr(rx, ry)[0] if np.std(rx) > 0 and np.std(ry) > 0 else np.nan


def cc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[m], y[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    L = np.array([r["length"] for r in rows], float)
    print(f"=== E119 flips + non-GPU entropy proxies (n={len(rows)}) ===\n")

    # extensive vs intensive: per-residue normalize the raw features
    EXT = ["poc_n", "bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count", "mj_contact", "mean_burial"]
    print("A. FLIP MECHANISM — raw corr vs partial corr (controlling length) vs per-residue (intensive):")
    print(f"   corr(length, ΔG) = {cc(L, y):+.3f}  (the confounder)\n")
    print(f"   {'feature':<14}{'raw r':>9}{'partial|L':>11}{'/L (intensive) r':>18}")
    for c in EXT:
        raw = cc([r["feat"][c] for r in rows], y)
        par = pcorr([r["feat"][c] for r in rows], y, L)
        perres = cc([r["feat"][c] / max(1, len(r["seq"])) for r in rows], y)
        print(f"   {c:<14}{raw:>+9.2f}{par:>+11.2f}{perres:>+18.2f}")
    print("   → if raw≈0/flips but partial|L and /L are sign-consistent & stronger, the flip IS size-confounding.")

    # show a flip resolving: pick the feature that flips most across bands, show /L stabilizes it
    print("\n   per-band check for bsa_hyd (raw) vs bsa_hyd/L (intensive):")
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        m = np.array([band(int(x)) == b for x in L])
        if m.sum() >= 8:
            raw = cc([rows[i]["feat"]["bsa_hyd"] for i in range(len(rows)) if m[i]], y[m])
            inten = cc([rows[i]["feat"]["bsa_hyd"] / max(1, len(rows[i]["seq"])) for i in range(len(rows)) if m[i]], y[m])
            print(f"     {b:<11} raw={raw:+.2f}  /L={inten:+.2f}")

    # B. non-GPU entropy proxies
    print("\nB. NON-GPU CONFORMATIONAL-ENTROPY PROXIES  corr(proxy, ΔG)  [>0 expected: floppier→weaker]")
    proxies = {}
    for r in rows:
        s = r["seq"].upper()
        Ls = max(1, len(s))
        r["p_scent"] = sum(SC_ENTROPY.get(c, 5) for c in s) / Ls          # mean side-chain config entropy
        r["p_scent_tot"] = sum(SC_ENTROPY.get(c, 5) for c in s)           # total (extensive)
        r["p_rot"] = sum(ROT.get(c, 2) for c in s) / Ls                   # mean rotatable side-chain bonds
        r["p_flex"] = np.mean([FLEX.get(c, 0.45) for c in s])             # mean B-factor flexibility
        r["p_gp"] = (s.count("G") + s.count("P")) / Ls                    # Gly+Pro (rigidifiers → lower entropy)
        r["p_disorder"] = 1 - r["feat"]["org_density"]                    # structural disorder
        r["p_len"] = float(Ls)
    for nm in ["p_scent", "p_scent_tot", "p_rot", "p_flex", "p_gp", "p_disorder", "p_len"]:
        v = [r[nm] for r in rows]
        line = f"   {nm:<14} ALL={cc(v,y):+.2f}"
        for b in ["med9-12", "long13-16", "vlong≥17"]:
            m = np.array([band(int(x)) == b for x in L])
            line += f"  {b}={cc([rows[i][nm] for i in range(len(rows)) if m[i]], y[m]):+.2f}"
        print(line)

    # if s_free computed, correlate proxies with the real MD entropy (find best cheap surrogate)
    sf_path = ROOT / "data/sfree_results.jsonl"
    if sf_path.exists():
        sfree = {json.loads(l)["hash"]: json.loads(l)["s_free"] for l in sf_path.read_text().splitlines()}
        pairs = [(r, sfree[seqhash(r["seq"].upper())]) for r in rows if seqhash(r["seq"].upper()) in sfree]
        if len(pairs) > 20:
            sfv = [p[1] for p in pairs]
            print(f"\n   vs REAL MD s_free (n={len(pairs)}) — which proxy best surrogates it (no GPU)?")
            for nm in ["p_scent", "p_rot", "p_flex", "p_gp", "p_disorder", "p_len"]:
                print(f"     corr({nm}, s_free) = {cc([p[0][nm] for p in pairs], sfv):+.3f}")


if __name__ == "__main__":
    main()
