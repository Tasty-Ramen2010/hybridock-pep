"""E68 — intra-peptide ORGANIZATION score (pre-organization / rigidity) from the 3D structure.

Ram: a strong binder doesn't magically organize — internal BONDS hold it rigid, and rigidity = low
free-state entropy = binds strong despite small interface (the E67 misses). Read the peptide's 3D
structure, detect intramolecular interactions by inter-atomic distance + a bond-type dictionary, score
them, normalize per residue (intensive), and test whether it (a) is sign-stable vs ΔG across datasets,
(b) flags the under-predicted strong binders, (c) lifts the crystal-65 LOO recalibration alongside
hyd_frac + cysteine.

Bond dictionary (heavy-atom distance cutoffs, |i−j|>=2 to skip trivial neighbours):
  disulfide   Cys SG–SG          < 2.5 Å   weight 3.0 (covalent, true pre-organization)
  salt_bridge (D/E carboxyl)–(K/R/H cation) < 4.0 Å  weight 1.5
  ss_hbond    backbone N···O     < 3.5 Å, |i−j|>=3   weight 1.0 (helix/sheet)
  aromatic    F/Y/W/H ring centroids < 6.0 Å         weight 1.0 (pi-stacking)
Weights are the feature CONSTRUCTION; the calibration fits the overall kcal/mol coefficient.

CAVEAT: peptide PDB is the BOUND conformation, so salt-bridge/Hbond may be binding-induced; disulfides
are genuinely free-state. Honest proxy for pre-organization, strongest on the covalent term.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser  # noqa: E402

P = PDBParser(QUIET=True)
CACHE = Path("/tmp/e68_org.json")
W = dict(disulfide=3.0, salt_bridge=1.5, ss_hbond=1.0, aromatic=1.0)
ANION = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
CATION = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"], "HIS": ["ND1", "NE2"]}
AROM = {"PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]}


def org_features(pep_pdb):
    res = [r for r in P.get_structure("p", str(pep_pdb))[0].get_residues() if r.id[0] == " "]
    L = len(res)
    atoms = {}  # idx -> {name: coord}
    for i, r in enumerate(res):
        atoms[i] = {a.name: a.coord for a in r}
    rn = [r.resname.upper() for r in res]

    def dist(c1, c2):
        return float(np.linalg.norm(c1 - c2))

    n_ss = n_sb = n_hb = n_ar = 0
    # disulfide
    cys = [i for i, n in enumerate(rn) if n == "CYS" and "SG" in atoms[i]]
    for a in range(len(cys)):
        for b in range(a + 1, len(cys)):
            if abs(cys[a] - cys[b]) >= 2 and dist(atoms[cys[a]]["SG"], atoms[cys[b]]["SG"]) < 2.5:
                n_ss += 1
    # salt bridges
    for i in range(L):
        for j in range(i + 2, L):
            pair = None
            if rn[i] in ANION and rn[j] in CATION:
                pair = (i, ANION[rn[i]], j, CATION[rn[j]])
            elif rn[j] in ANION and rn[i] in CATION:
                pair = (j, ANION[rn[j]], i, CATION[rn[i]])
            if pair:
                ai, an, ci, cn = pair
                ds = [dist(atoms[ai][x], atoms[ci][y]) for x in an if x in atoms[ai]
                      for y in cn if y in atoms[ci]]
                if ds and min(ds) < 4.0:
                    n_sb += 1
    # backbone ss hbonds
    for i in range(L):
        if "N" not in atoms[i]:
            continue
        for j in range(L):
            if abs(i - j) >= 3 and "O" in atoms[j] and dist(atoms[i]["N"], atoms[j]["O"]) < 3.5:
                n_hb += 1
                break
    # aromatic stacking (ring centroids)
    cents = {i: np.mean([atoms[i][a] for a in AROM[rn[i]] if a in atoms[i]], axis=0)
             for i in range(L) if rn[i] in AROM and any(a in atoms[i] for a in AROM[rn[i]])}
    ci = list(cents)
    for a in range(len(ci)):
        for b in range(a + 1, len(ci)):
            if abs(ci[a] - ci[b]) >= 2 and dist(cents[ci[a]], cents[ci[b]]) < 6.0:
                n_ar += 1
    score = W["disulfide"] * n_ss + W["salt_bridge"] * n_sb + W["ss_hbond"] * n_hb + W["aromatic"] * n_ar
    return dict(n_ss=n_ss, n_sb=n_sb, n_hb=n_hb, n_ar=n_ar, org_score=score,
                org_density=score / max(1, L),
                hyd_frac=sum(c in "AILMFVWC" for c in "".join(
                    {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                     "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                     "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}.get(n, "X")
                    for n in rn)) / max(1, L),
                cys_frac=rn.count("CYS") / max(1, L), L=L)


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            out[f"cr_{r['pdb']}"] = {**org_features(r["peptide_pdb"]), "y": r["dg_exp"], "ds": "cr65"}
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:30]}")
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep = work / f"{k}_pep.pdb"
        if pep.exists():
            try:
                out[f"98_{k}"] = {**org_features(pep), "y": v["y"], "ds": "the98"}
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:30]}")
    CACHE.write_text(json.dumps(out))
    return out


def main():
    d = build()
    rows = list(d.values())
    cr = [r for r in rows if r["ds"] == "cr65"]
    t98 = [r for r in rows if r["ds"] == "the98"]
    print(f"=== E68 intra-peptide organization.  cr65={len(cr)} the98={len(t98)} ===")
    tot = dict(n_ss=sum(r["n_ss"] for r in rows), n_sb=sum(r["n_sb"] for r in rows),
               n_hb=sum(r["n_hb"] for r in rows), n_ar=sum(r["n_ar"] for r in rows))
    print(f"  bonds detected (all): {tot}")

    print("\n=== (1) corr(feature, ΔG) per dataset — sign-stable? ===")
    for f in ["org_density", "org_score", "n_ss", "n_sb", "n_hb", "n_ar", "cys_frac", "hyd_frac"]:
        c = spearmanr([r[f] for r in cr], [r["y"] for r in cr]).statistic
        t = spearmanr([r[f] for r in t98], [r["y"] for r in t98]).statistic
        st = "YES" if c * t > 0 else "flip"
        print(f"  {f:<13} cr65={c:+.3f}  the98={t:+.3f}  {st}")

    print("\n=== (2) does org flag the UNDER-predicted strong binders? (the98 residual) ===")
    cat = json.loads(Path("/tmp/e63_catalog.json").read_text())
    m98 = {k[3:]: v for k, v in cat.items() if v["ds"] == "the98"}
    j = [(d[f"98_{k}"], m98[k]) for k in m98 if f"98_{k}" in d]
    X = np.array([[b["mmgbsa"], b["rg_per_L"]] for _, b in j]); y = np.array([b["y"] for _, b in j])
    A = np.column_stack([np.ones(len(X)), X]); w = np.linalg.lstsq(A, y, rcond=None)[0]; resid = y - A @ w
    for f in ["org_density", "cys_frac", "n_ss"]:
        print(f"  corr({f:<12}, residual) = {spearmanr([a[f] for a, _ in j], resid).statistic:+.3f}  "
              f"(neg => flags under-rated strong)")

    print("\n=== (3) crystal-65 LOO recalibration: base geom +rg_per_L  vs  +hyd_frac+cys+org ===")
    geom = {r["pdb"]: r for r in json.loads(Path("/tmp/e66_cr65_geom.json").read_text())}
    recs = []
    for r in cr:
        pdb = None
        # cr key is 'cr_<pdb>'; find original
        for k, v in d.items():
            if v is r:
                pdb = k[3:]; break
        if pdb in geom:
            recs.append({**geom[pdb], "hyd_frac": r["hyd_frac"], "cys_frac": r["cys_frac"],
                         "org_density": r["org_density"]})
    from hybridock_pep.scoring.geometry_features import GEOMETRY_FEATURE_KEYS

    def loo(keys):
        y = np.array([r["y"] for r in recs]); pr = np.zeros(len(recs))
        for i in range(len(recs)):
            tr = [r for jx, r in enumerate(recs) if jx != i]
            X = np.array([[r[k] for k in keys] for r in tr], float); yt = np.array([r["y"] for r in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); w = np.linalg.lstsq(A, yt, rcond=None)[0]
            xi = (np.array([recs[i][k] for k in keys], float) - mu) / sd
            pr[i] = w[0] + np.dot(w[1:], xi)
        return pearsonr(pr, y)[0], spearmanr(pr, y).statistic, float(np.sqrt(np.mean((pr - y) ** 2)))
    base = list(GEOMETRY_FEATURE_KEYS)
    for nm, keys in [("base (geom+rg_per_L)", base),
                     ("+ hyd_frac", base + ["hyd_frac"]),
                     ("+ hyd_frac + cys_frac", base + ["hyd_frac", "cys_frac"]),
                     ("+ hyd + cys + org_density", base + ["hyd_frac", "cys_frac", "org_density"])]:
        p, s, rmse = loo(keys)
        print(f"  {nm:<28} LOO Pearson={p:+.3f}  Spearman={s:+.3f}  RMSE={rmse:.2f}  (n={len(recs)})")


if __name__ == "__main__":
    main()
