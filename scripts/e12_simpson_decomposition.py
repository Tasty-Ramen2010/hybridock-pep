"""E12 — Simpson's decomposition: is the universal physics WITHIN-protein?

E11 showed n_contact flips sign across datasets — impossible as physics. Hypothesis:
the confounder is PER-PROTEIN BASELINE affinity. crystal-65 compares 65 different
proteins (between-protein); PEPBI compares mutants within ~31 groups (within-protein).
Within a protein, more/better contacts -> stronger (clean). Pooled across proteins,
each protein's baseline swamps/reverses the marginal (Simpson).

Test: split each feature's correlation into WITHIN-group (demeaned per binding
group/protein) vs BETWEEN-group (group means). If WITHIN is sign-stable & physical
across datasets but BETWEEN is the part that flips, the confounder is the baseline,
and the universal sign-stable model is a mixed-effects (per-protein intercept) form.

Features tested: n_contact, bsa, hb_density, nis_p_frac, e_int (where available).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import openpyxl

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402

P = PDBParser(QUIET=True)
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}


def feats_from_pair(pep, poc, cut=5.5, hb=3.5):
    pepr = [r for r in P.get_structure("p", pep)[0].get_residues() if r.id[0] == " "]
    pocat = [a for r in P.get_structure("q", poc)[0].get_residues()
             if r.id[0] == " " for a in r if a.element != "H"]
    return _feats(pepr, pocat, cut, hb)


def feats_from_chainB(pdb, cut=5.5, hb=3.5):
    s = P.get_structure("x", pdb)[0]
    if "B" not in [c.id for c in s]:
        return None
    pepr = [r for r in s["B"] if r.id[0] == " "]
    pocat = [a for c in s if c.id != "B" for r in c if r.id[0] == " "
             for a in r if a.element != "H"]
    return _feats(pepr, pocat, cut, hb)


def _feats(pepr, pocat, cut, hb):
    if not pepr or not pocat:
        return None
    ns = NeighborSearch(pocat)
    nc = nhb = npolar = nnis = 0
    for rp in pepr:
        contacted = False
        for a in rp:
            if a.element == "H":
                continue
            if ns.search(a.coord, cut):
                contacted = True
            if a.element in ("N", "O") and any(
                    b.element in ("N", "O") and np.linalg.norm(a.coord - b.coord) <= hb
                    for b in ns.search(a.coord, hb)):
                nhb += 1
        if contacted:
            nc += 1
        else:
            nnis += 1
            if rp.resname.upper() in POLAR:
                npolar += 1
    L = len(pepr)
    return dict(L=L, nc=nc, hb_density=nhb / max(1, nc),
                nis_p=npolar / max(1, nnis))


def within_between(rows, group_key):
    """Return (within-group demeaned corr, between-group-mean corr) for each feature."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    feats = ["nc", "hb_density", "nis_p", "L"]
    res = {}
    for f in feats:
        wy, wv = [], []   # within (demeaned)
        by, bv = [], []   # between (group means)
        for g, members in groups.items():
            if len(members) < 2:
                continue
            yv = np.array([m["y"] for m in members])
            fv = np.array([m[f] for m in members])
            wy.append(yv - yv.mean())
            wv.append(fv - fv.mean())
            by.append(yv.mean())
            bv.append(fv.mean())
        # also include all group means (even singletons) for between
        bym = [np.mean([m["y"] for m in mm]) for mm in groups.values()]
        bfm = [np.mean([m[f] for m in mm]) for mm in groups.values()]
        within = pearsonr(np.concatenate(wv), np.concatenate(wy)).statistic if wv and np.std(np.concatenate(wv)) > 0 else float("nan")
        between = pearsonr(bfm, bym).statistic if np.std(bfm) > 0 else float("nan")
        res[f] = (within, between)
    return res


def kmer_groups(seqs, th=0.3, k=3):
    ks = [{s[i:i+k] for i in range(max(0, len(s)-k+1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - th).fit_predict(D)


def main():
    # crystal-65 with kmer-family grouping
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    cr = []
    for r in e0:
        if not r.get("pep_pdb"):
            continue
        f = feats_from_pair(r["pep_pdb"], r["poc_pdb"])
        if f:
            f.update(y=r["y"], seq=sm.get(r["pdb"].upper(), "X"))
            cr.append(f)
    g = kmer_groups([r["seq"] for r in cr], 0.3)
    for r, gi in zip(cr, g):
        r["grp"] = int(gi)

    # PEPBI with binding-group grouping
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    wb = openpyxl.load_workbook("/tmp/pepbi/PEPBI.xlsx", read_only=True)
    rows = list(wb["PEPBI Data"].iter_rows(values_only=True))
    hdr = rows[1]
    ci = lambda n: hdr.index(n)
    num = lambda x: (float(x) if str(x).replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit() else None) if x is not None else None
    c_nm, c_dg, c_kd, c_bg = ci("PEPBI Complex Name"), ci("ΔG (kcal/mol)"), ci("KD (M)"), ci("Binding Group")
    pb = []
    for r in rows[2:]:
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        f = feats_from_chainB(files[nm])
        if f:
            f.update(y=dg, grp=r[c_bg])
            pb.append(f)

    print(f"crystal-65 n={len(cr)} groups={len(set(r['grp'] for r in cr))}")
    print(f"PEPBI      n={len(pb)} groups={len(set(r['grp'] for r in pb))}")

    print("\n" + "=" * 70)
    print("WITHIN-group (demeaned) vs BETWEEN-group corr(feature, ΔG)")
    print("If WITHIN is sign-stable across datasets but BETWEEN flips => Simpson,")
    print("confounder = per-protein baseline. (ΔG<0=strong; physical nc<0, nis_p<0)")
    print("=" * 70)
    cw = within_between(cr, "grp")
    pw = within_between(pb, "grp")
    print(f"{'feature':<12}{'cryst WITHIN':>14}{'pepbi WITHIN':>14}{'cryst BETWEEN':>15}{'pepbi BETWEEN':>15}")
    for f in ["nc", "hb_density", "nis_p", "L"]:
        cwi, cbe = cw.get(f, (float('nan'), float('nan')))
        pwi, pbe = pw.get(f, (float('nan'), float('nan')))
        print(f"{f:<12}{cwi:>14.3f}{pwi:>14.3f}{cbe:>15.3f}{pbe:>15.3f}")

    Path("/tmp/e12_cr.json").write_text(json.dumps(cr))
    Path("/tmp/e12_pb.json").write_text(json.dumps(pb))


if __name__ == "__main__":
    main()
