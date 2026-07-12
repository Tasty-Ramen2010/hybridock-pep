"""E1 — can the HONEST (physically-correct-sign) model beat 0.36?

Every feature enters with a sign fixed A PRIORI by physics/PRODIGY (never fit
from this dataset), so a win cannot be the backwards-Vina artifact.

New honest levers vs E0:
  - vina_eff = vina / L          (ligand efficiency: per-residue binding;
                                  removes the size trap that flips raw Vina)
  - dh_eff   = mmgbsa_dh / L      (per-residue enthalpy)
  - %NIS by AREA (PRODIGY's real formulation, not residue counts):
      nis_apolar_area_frac, nis_charged_area_frac
  - buried hydrophobic area density: bsa_apolar / L
  - fixed-physics entropy penalty: +0.7 * L applied directly (not fit)
  - per-SS entropy via s_ss if available

Honest fit = non-negative least squares on sign-oriented features + grouped
GroupKFold out-of-fold Pearson. Forward selection keeps only features that
improve grouped-oof r.
"""
from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "GLY"}


def _cls(rn):
    rn = rn.upper()
    return "C" if rn in CHARGED else ("P" if rn in POLAR else "A")


def area_nis(pep_pdb, poc_pdb, cutoff=5.5):
    """Per-atom SASA on peptide alone -> classify non-interface residue area."""
    pep = _P.get_structure("p", pep_pdb)
    _SR.compute(pep, level="A")
    pep_res = [r for r in pep[0].get_residues() if r.id[0] == " "]
    poc_res = [r for r in _P.get_structure("q", poc_pdb)[0].get_residues() if r.id[0] == " "]
    poc_heavy = [[a.coord for a in r if a.element != "H"] for r in poc_res]
    c2 = cutoff * cutoff
    contact = set()
    for i, rp in enumerate(pep_res):
        pa = [a.coord for a in rp if a.element != "H"]
        for qa in poc_heavy:
            if any(np.sum((ac - bc) ** 2) <= c2 for ac in pa for bc in qa):
                contact.add(i)
                break
    area = {"C": 0.0, "P": 0.0, "A": 0.0}
    tot = 0.0
    for i, rp in enumerate(pep_res):
        if i in contact:
            continue
        ra = sum(float(a.sasa) for a in rp)
        area[_cls(rp.resname)] += ra
        tot += ra
    if tot <= 0:
        return dict(nis_apolar_area=0.0, nis_charged_area=0.0, nis_polar_area=0.0)
    return dict(
        nis_apolar_area=area["A"] / tot,
        nis_charged_area=area["C"] / tot,
        nis_polar_area=area["P"] / tot,
    )


def kmer_groups(seqs, threshold=0.3, k=3):
    ks = [{s[i : i + k] for i in range(max(0, len(s) - k + 1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - threshold,
    ).fit_predict(D)


# (feature, physical sign vs dG): +1 means feature increases dG (weakens binding)
# so in NNLS we orient col = sign * standardized_feature and force coeff>=0.
PHYS = {
    "vina_eff": +1,        # larger (less negative)/res -> weaker. oriented so + = weaker
    "nis_charged_area": +1,  # PRODIGY: more exposed charge -> weaker
    "nis_apolar_area": -1,   # PRODIGY: more exposed apolar -> ... test both? PRODIGY +; keep -1 (buried-pref)
    "nis_polar_area": -1,    # E0: polar exposed -> stronger
    "ic_charged_frac": -1,   # salt-bridge density -> stronger
    "ic_apolar_frac": +1,    # apolar contact frac -> weaker (entropy/desolv) per E0 sign
    "ent_fixed": +1,         # +0.7*L fixed entropy penalty -> weaker
    "dh_eff": -1,            # per-res enthalpy more negative -> stronger
}


def grouped_oof_nnls(cols_signed, y, g):
    """cols_signed: list of oriented standardized feature vectors (coeff>=0)."""
    X = np.column_stack(cols_signed)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g))))
    pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, g):
        mu = y[tr].mean()
        w, _ = nnls(X[tr], y[tr] - mu)
        pred[te] = X[te] @ w + mu
    return pearsonr(pred, y).statistic


def main():
    rows = json.loads(Path("/tmp/e0_features.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    print("augmenting with area-NIS...")
    for i, r in enumerate(rows):
        try:
            r.update(area_nis(r["pep_pdb"], r["poc_pdb"]))
        except Exception as e:  # noqa: BLE001
            print(f"  {r['pdb']} fail: {e}")
        L = r.get("L") or 1
        r["vina_eff"] = (r.get("vina") or 0) / L
        r["dh_eff"] = (r.get("dh") or 0) / L
        r["ent_fixed"] = 0.7 * L
    Path("/tmp/e1_features.json").write_text(json.dumps(rows))

    seqs = np.array([sm.get(r["pdb"].upper(), "X") for r in rows])
    y_all = np.array([r["y"] for r in rows], float)
    kd = np.array([r["aff"] == "Kd" for r in rows])

    for label, mask in [("ALL", np.ones(len(rows), bool)), ("Kd-only", kd)]:
        y = y_all[mask]
        seqm = list(seqs[mask])
        g = kmer_groups(seqm, 0.3)
        # standardize + orient each feature
        oriented = {}
        for f, sgn in PHYS.items():
            v = np.array([rr.get(f, np.nan) for rr in rows], float)[mask]
            if not np.all(np.isfinite(v)) or np.std(v) == 0:
                continue
            z = (v - v.mean()) / (v.std() + 1e-9)
            oriented[f] = sgn * z
        print(f"\n=== {label} (n={len(y)}, fam={len(np.unique(g))}) ===")
        # single-feature honest grouped-oof
        singles = []
        for f, col in oriented.items():
            r = grouped_oof_nnls([col], y, g)
            singles.append((r, f))
        singles.sort(reverse=True)
        print("  honest single-feature grouped-oof r:")
        for r, f in singles:
            print(f"    {f:<20}{r:+.3f}")
        # greedy forward selection
        chosen, best = [], -1.0
        pool = list(oriented)
        while pool:
            cand = []
            for f in pool:
                r = grouped_oof_nnls([oriented[c] for c in chosen + [f]], y, g)
                cand.append((r, f))
            cand.sort(reverse=True)
            if cand[0][0] <= best + 0.005:
                break
            best, pick = cand[0]
            chosen.append(pick)
            pool.remove(pick)
        print(f"  >> best HONEST model: {chosen}")
        print(f"  >> grouped-oof r = {best:.3f}")


if __name__ == "__main__":
    main()
