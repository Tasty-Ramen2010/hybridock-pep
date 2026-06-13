"""E102 — in-depth QC of the peptide-affinity data we can actually obtain.

Findings during the pull:
  * Current BioLiP_nr (86,458 rows) has **0 peptide+affinity entries** — Jan-2025 PDBbind-CN removal
    (licensing) stripped exactly the peptide Kd column PPI-Affinity relied on. BioLiP route = DEAD for
    peptides now. (Confirmed: col16 PDBbind empty; ligand 'peptide' rows = 4217 but none with affinity.)
  * PPI-Affinity figshare SI ships the T100 protein-PEPTIDE TEST set (100 complexes) WITH ΔG truth AND
    six competitor predictions (PRODIGY/DFIRE/CP_PIE/Kdeep/RF-Score/PPI-Affinity) — an independent
    head-to-head benchmark we can grade ourselves on. Plus EPIX4 (57 SAR variants, IC50, one family).

This script QCs all of it and computes the COMPETITOR LEADERBOARD on T100 (the bar to beat), with the
hard-won degeneracy check (unique-Kd count) so we never re-enter the cr65-vlong flat-label trap.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "data" / "biolip"
SI = BIO / "ppiaffinity_si" / "SI"
RT = 0.5922  # kcal/mol at 298K


def fitted(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5 or np.std(x[m]) == 0:
        return np.nan, np.nan, np.nan, int(m.sum())
    r = pearsonr(x[m], y[m])[0]
    rho = spearmanr(x[m], y[m]).statistic
    # sign-free linear fit for RMSE (fair to arbitrary-unit predictors)
    a, b = np.polyfit(x[m], y[m], 1)
    rmse = float(np.sqrt(np.mean((a * x[m] + b - y[m]) ** 2)))
    return abs(r), rho, rmse, int(m.sum())


def degeneracy(y):
    y = np.round(np.asarray(y, float), 2)
    u = len(set(y))
    from collections import Counter
    dup = {k: v for k, v in Counter(y).items() if v > 1}
    return u, len(y), np.std(y), max(y) - min(y), dup


def biolip_raw():
    F = BIO / "BioLiP_nr.txt"
    print("=== A. BioLiP_nr RAW assessment ===")
    if not F.exists():
        print("   (BioLiP_nr.txt not present)\n"); return
    n = pep = pepaff = aff = 0
    src = {"manual": 0, "MOAD": 0, "PDBbind": 0, "BindingDB": 0}
    with open(F) as fh:
        for ln in fh:
            c = ln.rstrip("\n").split("\t")
            if len(c) < 17:
                continue
            n += 1
            isp = c[4] == "peptide"
            pep += isp
            a14, a15, a16, a17 = c[13], c[14], c[15], c[16]
            hasa = any([a14, a15, a16, a17])
            aff += hasa
            src["manual"] += bool(a14); src["MOAD"] += bool(a15)
            src["PDBbind"] += bool(a16); src["BindingDB"] += bool(a17)
            if isp and hasa:
                pepaff += 1
    print(f"   rows={n}  peptide-ligand rows={pep}  rows-with-any-affinity={aff}")
    print(f"   affinity source counts: {src}")
    print(f"   *** PEPTIDE + AFFINITY = {pepaff} ***  → BioLiP peptide-Kd is GONE (PDBbind removed Jan-2025)\n")


def parse_t100():
    f = SI / "SI-File-6-protein-peptide-test-set-1.csv"
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        m = re.match(r"([0-9a-zA-Z]{4})([A-Za-z0-9]+)\.pdb-([0-9a-zA-Z]{4})_([A-Za-z0-9]+)_([A-Za-z0-9]+)_", r["PDB_NAME"])
        r["pdb"] = m.group(1).lower() if m else None
        r["rec_chain"] = m.group(2) if m else None
        r["pep_chain"] = m.group(5) if m else None
        r["y"] = float(r["Binding_affinity"])
    return rows


def t100_qc(rows):
    print("=== B. PPI-Affinity T100 protein-peptide TEST set QC ===")
    y = np.array([r["y"] for r in rows])
    u, ntot, sd, rng, dup = degeneracy(y)
    print(f"   n={ntot}  ΔG range=[{y.min():.1f},{y.max():.1f}] kcal  std={sd:.2f}  UNIQUE values={u} ({u/ntot:.0%})")
    print(f"   degeneracy (worst dup ΔG values): {dict(sorted(dup.items(), key=lambda kv:-kv[1])[:4])}")
    okpdb = sum(1 for r in rows if r["pdb"])
    print(f"   parseable PDB+chains: {okpdb}/{ntot}  e.g. {rows[0]['pdb']} rec={rows[0]['rec_chain']} pep={rows[0]['pep_chain']}")
    # overlap with our 156
    ours = set()
    for fn in ["data/pooled_benchmark_train.csv", "data/pooled_benchmark_test.csv"]:
        for r in csv.DictReader(open(ROOT / fn)):
            ours.add(r["pdb"].lower()[:4])
    ov = sum(1 for r in rows if r["pdb"] and r["pdb"][:4] in ours)
    print(f"   overlap with our existing 156 (by PDB id): {ov}  → {'mostly INDEPENDENT' if ov < 15 else 'CONTAMINATED'}\n")

    print("   --- COMPETITOR LEADERBOARD on T100 (fitted |r| vs truth ΔG; THE BAR) ---")
    preds = [c for c in rows[0].keys() if c not in ("PDB_NAME", "Binding_affinity", "pdb", "rec_chain", "pep_chain", "y")]
    board = []
    for c in preds:
        try:
            x = [float(r[c]) for r in rows]
        except (ValueError, KeyError):
            continue
        r_, rho, rmse, nn = fitted(x, y)
        board.append((c.strip(), r_, rho, rmse, nn))
    for nm, r_, rho, rmse, nn in sorted(board, key=lambda t: -(t[1] if t[1] == t[1] else 0)):
        print(f"      {nm:<14} |r|={r_:.3f}  ρ={rho:+.3f}  RMSE(fit)={rmse:.2f}  (n={nn})")
    print("      → THIS is the head-to-head target. We fetch these 100 structures + score with ours next.\n")
    return rows


def epix4_qc():
    f = SI / "SI-File-7-protein-peptide-test-EPIX4.csv"
    rows = list(csv.DictReader(open(f)))
    print("=== C. EPIX4 SAR set QC ===")
    seqs = [r.get("Sequence", "") for r in rows]
    ic50 = []
    for r in rows:
        try:
            ic50.append(float(r["IC50(nM)"]))
        except (ValueError, KeyError):
            ic50.append(np.nan)
    ic50 = np.array(ic50)
    dg = RT * np.log(ic50 * 1e-9)  # IC50→pseudo-ΔG (approx, IC50≠Kd)
    lens = [len(s) for s in seqs if s]
    print(f"   n={len(rows)}  unique seqs={len(set(seqs))}  length range={min(lens)}-{max(lens)}")
    print(f"   single-family SAR (EPI-X4 variants); IC50 not Kd → noisier; pseudo-ΔG range "
          f"[{np.nanmin(dg):.1f},{np.nanmax(dg):.1f}]")
    print(f"   USE: within-family selectivity/SAR ranking test, NOT cross-family affinity training.\n")


def main():
    print(f"\n######## E102 PEPTIDE-AFFINITY DATA QC ########\n")
    biolip_raw()
    rows = t100_qc(parse_t100())
    epix4_qc()
    # persist a clean usable manifest of T100 for the structure-fetch + scoring stage
    out = [{"pdb": r["pdb"], "rec_chain": r["rec_chain"], "pep_chain": r["pep_chain"],
            "dg_exp": r["y"], "ppi_affinity": r.get("PPI-Affinity"), "prodigy": r.get("PRODIGY")}
           for r in rows if r["pdb"]]
    (BIO / "t100_peptide_manifest.json").write_text(json.dumps(out, indent=1))
    print(f"=== D. VERDICT ===")
    print("   • Current BioLiP: DEAD for peptide-Kd (PDBbind affinities removed Jan-2025).")
    print("   • Usable NOW: T100 (100 indep peptide complexes, ΔG + 6 competitor scores) → wrote")
    print(f"     {(BIO/'t100_peptide_manifest.json').relative_to(ROOT)} for the fetch+score head-to-head.")
    print("   • Full 1149/1901 training set: NOT on figshare → needs PDBbind v2020 registration")
    print("     (free account; protein≥40res + peptide 3-40res subset) — a human-login step for Ram.")
    print("   • Next compute step: fetch the 100 RCSB structures, RAPiDock-dock (or score crystal),")
    print("     run our 16-feat + ML-best-5, and drop OUR column into the leaderboard above.\n")


if __name__ == "__main__":
    main()
