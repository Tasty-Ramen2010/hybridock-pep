"""E74 — what UNIQUELY separates strong vs weak CHARGED binders? (shape/hydrophobicity, not raw charge)

E72/E73: net electrostatics is a wash on charged (coul −177 ≈ desolv +209); packing (vdw/burial) is the
separator. Ram: there must be something unique to strong vs weak CHARGED binders we can capture. The
physics intuition — a salt bridge's strength depends on its ENVIRONMENT, not just its existence:

  H1 charged-residue BURIAL: a buried charged residue forming a protected salt bridge binds strong;
     a surface-exposed charge (still solvated) does little. -> mean ΔSASA of D/E/K/R residues only.
  H2 HYDROPHOBIC SHIELDING: a salt bridge in a low-dielectric (hydrophobic) micro-environment is much
     stronger (desolvation already paid by the hydrophobic burial). -> hydrophobic burial NEAR charged res.
  H3 CHARGE COMPLEMENTARITY: peptide net charge opposite to pocket net charge = real attraction.
  H4 charged-residue PACKING: vdw/burial restricted to the charged residues.
  H5 buried-charge FRACTION: how many of the peptide's charges are actually buried vs dangling.
  H6 hydrophobic/charged BALANCE: strong charged binders may anchor with hydrophobics, charge is bonus.

Per-residue ΔSASA from structures (the98 + cr65), charged-residue-resolved. Test each on charged subset,
sign-stability across datasets. Goal: a sign-stable charged-strength separator to wire.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e74_charged.json")
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
CHG3 = POS3 | NEG3
HPHO3 = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "CYS", "PRO", "TYR"}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
      "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
      "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def per_res(pep_pdb, rec_pdb):
    pep = P.get_structure("p", str(pep_pdb))[0]
    SR.compute(pep, level="R")
    free = [(r.resname.upper(), r.id[1], r.sasa, np.mean([a.coord for a in r], axis=0))
            for r in pep.get_residues() if r.id[0] == " "]
    rec = P.get_structure("r", str(rec_pdb))[0]
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    cx = Structure("c"); m = Model(0); cx.add(m); used = set()
    for ch in list(pep.get_chains()) + list(rec.get_chains()):
        cid = ch.id
        while cid in used:
            cid = chr((ord(cid) + 1) % 90 + 33)
        used.add(cid); c2 = ch.copy(); c2.id = cid; m.add(c2)
    SR.compute(cx, level="R")
    comp = {}
    for ch in cx.get_chains():
        for r in ch.get_residues():
            if r.id[0] == " ":
                comp.setdefault((r.resname.upper(), r.id[1]), r.sasa)
    # receptor pocket charge near peptide
    pep_xyz = np.array([fr[3] for fr in free])
    poc_pos = poc_neg = 0
    for ch in rec.get_chains():
        for r in ch.get_residues():
            if r.id[0] != " " or r.resname.upper() not in CHG3:
                continue
            cen = np.mean([a.coord for a in r], axis=0)
            if np.min(((pep_xyz - cen) ** 2).sum(1)) < 100:  # within 10 Å of peptide
                if r.resname.upper() in POS3:
                    poc_pos += 1
                else:
                    poc_neg += 1
    res = []
    for rn, seq, fs, cen in free:
        cs = comp.get((rn, seq), fs)
        res.append(dict(rn=rn, dsasa=max(0.0, fs - cs), fs=fs, cen=cen))
    return res, poc_pos, poc_neg


def featurize(res, poc_pos, poc_neg, y, ds):
    chg = [r for r in res if r["rn"] in CHG3]
    L = len(res)
    seq = "".join(A3.get(r["rn"], "X") for r in res)
    nc = sum(c in "KR" for c in seq) - sum(c in "DE" for c in seq)
    f = dict(ds=ds, y=y, L=L, net_charge=nc, seq=seq)
    # H1 charged-residue burial (mean ΔSASA of charged res)
    f["chg_burial"] = float(np.mean([r["dsasa"] for r in chg])) if chg else 0.0
    f["chg_buried_frac"] = float(np.mean([r["dsasa"] > 40 for r in chg])) if chg else 0.0
    # H2 hydrophobic shielding: hydrophobic ΔSASA within 8 Å of any buried charged residue
    shield = 0.0
    buried_chg = [r for r in chg if r["dsasa"] > 20]
    for r in res:
        if r["rn"] in HPHO3 and r["dsasa"] > 10:
            if buried_chg and min(np.linalg.norm(r["cen"] - c["cen"]) for c in buried_chg) < 8.0:
                shield += r["dsasa"]
    f["hyd_shield"] = shield / max(1, L)
    # H3 charge complementarity (peptide charge vs pocket charge sign)
    poc_net = poc_pos - poc_neg
    f["chg_compl"] = -float(np.sign(nc) * np.sign(poc_net)) if (nc and poc_net) else 0.0
    f["chg_compl_mag"] = -(nc * poc_net) / (abs(nc) + abs(poc_net) + 1e-6)
    # H5 buried-charge fraction over the whole peptide
    f["n_buried_chg"] = float(sum(r["dsasa"] > 40 for r in chg))
    # H6 hyd/chg balance: hydrophobic buried area vs charged buried area
    hyd_bur = sum(r["dsasa"] for r in res if r["rn"] in HPHO3)
    chg_bur = sum(r["dsasa"] for r in chg)
    f["hyd_chg_balance"] = hyd_bur / (chg_bur + hyd_bur + 1e-6)
    f["mean_burial"] = float(np.mean([r["dsasa"] for r in res]))
    return f


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            try:
                res, pp, pn = per_res(pep, rec)
                out[f"98_{k}"] = featurize(res, pp, pn, v["y"], "the98")
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:30]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            res, pp, pn = per_res(r["peptide_pdb"], r["pocket_pdb"])
            out[f"cr_{r['pdb']}"] = featurize(res, pp, pn, r["dg_exp"], "cr65")
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:30]}")
    CACHE.write_text(json.dumps(out))
    return out


def sp(rows, f):
    x = np.array([r[f] for r in rows], float); y = np.array([r["y"] for r in rows], float)
    m = ~(np.isnan(x) | np.isnan(y))
    return spearmanr(x[m], y[m]).statistic if m.sum() > 5 else np.nan


def main():
    rows = list(build().values())
    ch = [r for r in rows if abs(r["net_charge"]) >= 2]
    chc = [r for r in ch if r["ds"] == "cr65"]; ch9 = [r for r in ch if r["ds"] == "the98"]
    print(f"=== E74 charged-strength hunt. charged total={len(ch)} (cr65={len(chc)}, the98={len(ch9)}) ===")
    print("\nSpearman vs ΔG on CHARGED (neg = feature↑ → stronger). Sign-stable across datasets = real:")
    print(f"{'hypothesis feature':<22}{'all chg':>9}{'cr65 chg':>10}{'the98 chg':>11}  stable?")
    feats = [("chg_burial", "H1 charged-res burial"), ("chg_buried_frac", "H1 frac charged buried"),
             ("hyd_shield", "H2 hydrophobic shielding"), ("chg_compl", "H3 charge complementarity"),
             ("chg_compl_mag", "H3 compl magnitude"), ("n_buried_chg", "H5 n buried charges"),
             ("hyd_chg_balance", "H6 hyd/chg balance"), ("mean_burial", "(ref) mean burial")]
    for f, desc in feats:
        a, c, d = sp(ch, f), sp(chc, f), sp(ch9, f)
        st = "YES" if (not np.isnan(c) and not np.isnan(d) and c * d > 0) else "flip/na"
        mark = "  <== sign-stable" if st == "YES" and min(abs(c), abs(d)) > 0.2 else ""
        print(f"  {desc:<22}{a:>+9.3f}{c:>+10.3f}{d:>+11.3f}  {st}{mark}")

    print("\n=== combined: best sign-stable feats, fitted Spearman on charged ===")
    for nm, cols in [("mean_burial alone", ["mean_burial"]),
                     ("+ hyd_shield", ["mean_burial", "hyd_shield"]),
                     ("+ chg_burial", ["mean_burial", "chg_burial"]),
                     ("+ hyd_chg_balance", ["mean_burial", "hyd_chg_balance"]),
                     ("all sign-stable", ["mean_burial", "hyd_shield", "hyd_chg_balance", "chg_burial"])]:
        sub = [r for r in ch if not any(np.isnan(r[c]) for c in cols)]
        if len(sub) < 10:
            continue
        X = np.array([[r[c] for c in cols] for r in sub]); yy = np.array([r["y"] for r in sub])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        w = np.linalg.lstsq(A, yy, rcond=None)[0]
        print(f"  {nm:<22} charged Spearman={spearmanr(A @ w, yy).statistic:+.3f} (n={len(sub)})")


if __name__ == "__main__":
    main()
