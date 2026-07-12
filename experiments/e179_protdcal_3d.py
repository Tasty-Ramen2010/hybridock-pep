"""E179 — FAITHFUL ProtDCal 3D contact descriptors for the exact 37 PPI-Affinity peptide features.

PPI's descriptors (SI-File-2 .idl) are w{Nc,FLC,NLC}(prop)_NO_GROUP_INV — 3D weighted-contact operators
(SM-3), NOT sequence composition. We compute them on the peptide chain's bound conformation.

wNc_i  = 0.5 * Σ_{j: |i-j|>t, dist<d}  P_i*P_j        (non-local weighted contacts)
wNLC_i = 0.5 * Σ_{j: |i-j|<=t_loc, dist<d} P_i*P_j     (local weighted contacts, t=1 family)
wFLC_i = (Σ weighted local contacts) / (Σ local contacts)   (weighted fraction, local)
then restrict residues to SM-11 group G, aggregate with invariant INV (SM-6..9).

Faithfulness probe (this script): compute the 37 on T100 peptide structures, GroupKFold-CV with SVR on
JUST these 37 → does it reach PPI's ~0.5 AND correlate with PPI's SHIPPED predictions? Sweep contact (d,t)
and pick the config that best matches shipped (= recovering their effective parameters). If even the best
config gives corr~0, 3D contacts still don't reproduce PPI and we report that honestly.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/protdcal"))
sys.path.insert(0, str(ROOT / "scripts"))
from protdcal_spec import GROUPS, PROPS  # noqa: E402
import e158_overfit_failure_analysis as e158  # noqa: E402

_TAB = {r["just_AA"]: r for r in csv.DictReader(open(ROOT / "third_party/protdcal/protdcal_aa_table.csv"))}
PROP = {p: {aa: float(_TAB[aa][f"{p}_NO"]) for aa in _TAB} for p in PROPS}
T2O = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
_parser = PDBParser(QUIET=True)

# parse the 37 .idl descriptors
IDL = [l.strip() for l in open(ROOT / "data/biolip/ppiaffinity_si/SI/SI-File-2-ppep_project.idl") if l.strip()]


def parse_desc(d):
    # wNc(ECI)_NO_AHR_N1
    w = d[1:d.index("(")]                 # Nc / FLC / NLC
    prop = d[d.index("(") + 1:d.index(")")]
    rest = d[d.index(")") + 1:].split("_")  # ['', 'NO', 'AHR', 'N1']
    grp, inv = rest[2], rest[3]
    return w, prop, grp, inv


DESCS = [parse_desc(d) for d in IDL]


def residue_seq_and_coords(pep_pdb):
    """returns list of (one-letter, Cbeta-or-Calpha coord) in chain order."""
    try:
        st = _parser.get_structure("p", str(pep_pdb))
    except Exception:  # noqa: BLE001
        return None
    res = []
    for ch in st[0]:
        for r in ch:
            if r.id[0] != " ":
                continue
            aa = T2O.get(r.resname)
            if aa is None:
                continue
            atom = r["CB"] if "CB" in r else (r["CA"] if "CA" in r else None)
            if atom is None:
                continue
            res.append((aa, atom.coord))
    return res if len(res) >= 2 else None


def per_residue_w(res, prop, d_cut, t_cut):
    """returns dict windex-> per-residue array for this property."""
    n = len(res)
    P = np.array([PROP[prop].get(aa, 0.0) for aa, _ in res])
    XYZ = np.array([c for _, c in res])
    D = np.sqrt(((XYZ[:, None, :] - XYZ[None, :, :]) ** 2).sum(-1))
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    contact = (D < d_cut) & (sep > 0)
    nonlocal_c = contact & (sep > t_cut)
    local_c = contact & (sep <= t_cut)
    W = P[:, None] * P[None, :]
    nc = 0.5 * (nonlocal_c * W).sum(1)
    nlc = 0.5 * (local_c * W).sum(1)
    nloc = local_c.sum(1)
    flc = np.where(nloc > 0, (local_c * W).sum(1) / np.maximum(nloc, 1), 0.0)
    return {"Nc": nc, "NLC": nlc, "FLC": flc}


def invariant(v, inv):
    if v.size == 0:
        return 0.0
    if inv == "N1":
        return float(np.abs(v).sum())
    if inv == "N2":
        return float(np.sqrt((v ** 2).sum()))
    if inv == "Ar":
        return float(v.mean())
    if inv == "P2":
        return float(np.sqrt((v ** 2).mean()))
    if inv == "V":
        return float(v.var())
    if inv == "DE":
        return float(v.std())
    if inv == "RA":
        return float(v.max() - v.min())
    if inv == "S":  # skewness
        s = v.std()
        return float(((v - v.mean()) ** 3).mean() / s ** 3) if s > 1e-9 else 0.0
    if inv == "K":  # kurtosis
        s = v.std()
        return float(((v - v.mean()) ** 4).mean() / s ** 4) if s > 1e-9 else 0.0
    if inv == "I50":  # inter-quartile range
        return float(np.percentile(v, 75) - np.percentile(v, 25))
    if inv in ("MI30", "TI30", "SI30"):  # information content, 30 bins
        N = v.size
        if N < 2 or (v.max() - v.min()) < 1e-9:
            return 0.0
        hist, _ = np.histogram(v, bins=30)
        nz = hist[hist > 0]
        TI = N * np.log2(N) - (nz * np.log2(nz)).sum() if N > 1 else 0.0
        if inv == "TI30":
            return float(TI)
        if inv == "MI30":
            return float(TI / N) if N else 0.0
        if inv == "SI30":
            return float(TI / (N * np.log2(N))) if N > 1 else 0.0
    return 0.0


def descriptors(res, d_cut, t_cut):
    cache = {p: per_residue_w(res, p, d_cut, t_cut) for p in PROPS}
    aas = np.array([aa for aa, _ in res])
    out = []
    for w, prop, grp, inv in DESCS:
        vals = cache[prop][w]
        mask = np.array([aa in GROUPS[grp] for aa in aas])
        out.append(invariant(vals[mask], inv))
    return out


def met(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return float(np.corrcoef(p[ok], y[ok])[0, 1]), float(np.mean(np.abs(p[ok] - y[ok])))


def main():
    man = {m["pdb"].lower(): m for m in json.loads((ROOT / "data/biolip/t100_peptide_manifest.json").read_text())}
    seqcache = {json.loads(l)["pdb"].lower(): json.loads(l)
                for l in open(ROOT / "data/t100_extra_features.jsonl")}
    rows = []
    for pid, d in seqcache.items():
        m = man.get(pid)
        if m is None:
            continue
        pep = next(iter((ROOT / "runs/t100_extract").glob(f"{pid}_*_pep.pdb")), None)
        if pep is None:
            continue
        res = residue_seq_and_coords(pep)
        if res is None:
            continue
        try:
            ship = float(m["ppi_affinity"])
        except (TypeError, ValueError):
            ship = np.nan
        rows.append((pid, res, float(m["dg_exp"]), ship, d["seq"]))
    print(f"T100 with peptide structures: n={len(rows)}", flush=True)
    y = np.array([r[2] for r in rows]); ship = np.array([r[3] for r in rows])
    grp, _ = e158.greedy_cluster([r[4] for r in rows], 0.6)

    print(f"\n=== sweep contact (d,t): faithful 37-descriptor CV vs truth & vs SHIPPED preds ===")
    print(f"  (shipped-PPI vs truth on this subset: r={met(ship,y)[0]:+.3f})")
    best = None
    for d_cut in (6.0, 8.0, 10.0, 12.0):
        for t_cut in (1, 2, 3):
            X = np.nan_to_num([descriptors(r[1], d_cut, t_cut) for r in rows])
            pred = np.full(len(rows), np.nan)
            for tr, te in GroupKFold(5).split(X, y, grp):
                m = Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", C=4.0, gamma="scale"))])
                m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
            r_truth, _ = met(pred, y); r_ship, _ = met(pred, ship)
            print(f"  d={d_cut:>4} t={t_cut}: CV r_truth={r_truth:+.3f}  corr_vs_SHIPPED={r_ship:+.3f}")
            if best is None or r_ship > best[0]:
                best = (r_ship, r_truth, d_cut, t_cut)
    print(f"\n  BEST match-to-shipped: d={best[2]} t={best[3]}  corr_vs_shipped={best[0]:+.3f}  r_truth={best[1]:+.3f}")
    print("  -> if corr_vs_shipped jumps high (>0.5) the 3D contact descriptors ARE faithful (vs ~0 for seq proxy)")


if __name__ == "__main__":
    main()
