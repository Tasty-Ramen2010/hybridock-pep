"""E0 — Length-residualization gate for absolute peptide ΔG.

Decision experiment (see docs/kcalmol_research_synthesis.md):
does ANY cheap feature carry ΔG signal that survives removing peptide size?

For each feature we report:
  raw_r         Pearson(feature, ΔG_exp)
  spear         Spearman(feature, ΔG_exp)
  partial|L     Pearson of residuals after regressing BOTH on peptide_len
  partial|BSA   Pearson of residuals after regressing BOTH on buried SASA

A feature only matters if |partial r| stays >= 0.3 AND keeps its raw sign.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Structure, Model
from Bio.PDB.SASA import ShrakeRupley
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()

CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}


def _cls(resn: str) -> str:
    resn = resn.upper()
    if resn in CHARGED:
        return "C"
    if resn in POLAR:
        return "P"
    return "A"


def _sasa(struct: Structure.Structure) -> float:
    _SR.compute(struct, level="S")
    return float(struct.sasa)


def _residues(path: str):
    return [r for r in _P.get_structure("x", path)[0].get_residues() if r.id[0] == " "]


def _merged_structure(pep_pdb: str, poc_pdb: str) -> Structure.Structure:
    s = Structure.Structure("c")
    m = Model.Model(0)
    s.add(m)
    used = set()
    for src in (pep_pdb, poc_pdb):
        for ch in _P.get_structure("a", src)[0]:
            c = ch.copy()
            cid = c.id
            while cid in used:
                cid = chr(((ord(cid) - 64) % 26) + 65)  # next letter
            c.id = cid
            used.add(cid)
            m.add(c)
    return s


def interface_features(pep_pdb: str, poc_pdb: str, cutoff: float = 5.5) -> dict:
    pep_res = _residues(pep_pdb)
    poc_res = _residues(poc_pdb)
    poc_heavy = [[(a.coord, a.element) for a in r if a.element != "H"] for r in poc_res]

    ic = {"CC": 0, "PP": 0, "AA": 0, "MIX": 0}
    contact_pep = set()
    c2 = cutoff * cutoff
    for i, rp in enumerate(pep_res):
        pa = [a.coord for a in rp if a.element != "H"]
        hit_res = None
        for j, rq in enumerate(poc_res):
            qa = poc_heavy[j]
            found = False
            for ac in pa:
                for bc, _ in qa:
                    if np.sum((ac - bc) ** 2) <= c2:
                        found = True
                        break
                if found:
                    break
            if found:
                contact_pep.add(i)
                cp, cq = _cls(rp.resname), _cls(rq.resname)
                key = cp + cq
                if key == "CC":
                    ic["CC"] += 1
                elif key == "PP":
                    ic["PP"] += 1
                elif key == "AA":
                    ic["AA"] += 1
                else:
                    ic["MIX"] += 1
    n_contact = len(contact_pep)

    sasa_pep = _sasa(_P.get_structure("p", pep_pdb))
    sasa_poc = _sasa(_P.get_structure("q", poc_pdb))
    sasa_cpx = _sasa(_merged_structure(pep_pdb, poc_pdb))
    bsa = sasa_pep + sasa_poc - sasa_cpx

    nis_c = nis_p = nis_tot = 0
    for i, rp in enumerate(pep_res):
        if i in contact_pep:
            continue
        nis_tot += 1
        cl = _cls(rp.resname)
        nis_c += cl == "C"
        nis_p += cl == "P"

    return dict(
        n_contact=n_contact,
        ic_cc=ic["CC"], ic_pp=ic["PP"], ic_aa=ic["AA"], ic_mix=ic["MIX"],
        ic_charged_frac=ic["CC"] / max(1, n_contact),
        ic_apolar_frac=ic["AA"] / max(1, n_contact),
        bsa=bsa,
        nis_c_frac=nis_c / nis_tot if nis_tot else 0.0,
        nis_p_frac=nis_p / nis_tot if nis_tot else 0.0,
    )


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


FEATS = [
    "vina", "dh", "n_contact", "bsa", "ic_cc", "ic_pp", "ic_aa", "ic_mix",
    "ic_charged_frac", "ic_apolar_frac", "nis_c_frac", "nis_p_frac", "L",
]


def build_rows() -> list[dict]:
    """Merge the crystal benchmark with the MM-GBSA scored baseline.

    Reproducible source of /tmp/e0_rows.json consumed by the E0/E1/E2 chain.
    """
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    scored = json.loads((ROOT / "data/benchmark_crystal_scored_baseline.json").read_text())
    sc = {r["pdb"].upper(): r for r in scored}
    rows = []
    for r in base:
        s = sc.get(r["pdb"].upper(), {})
        rows.append(dict(
            pdb=r["pdb"], aff=r.get("affinity_type"), y=r["dg_exp"], L=r["peptide_len"],
            vina=r.get("vina_docked"), dh=s.get("mmgbsa_dh"), ie=s.get("ie"),
            pep_pdb=str((ROOT / r["peptide_pdb"]).resolve()) if r.get("peptide_pdb") else None,
            poc_pdb=str((ROOT / r["pocket_pdb"]).resolve()) if r.get("pocket_pdb") else None,
        ))
    Path("/tmp/e0_rows.json").write_text(json.dumps(rows))
    return rows


def main() -> None:
    rows = build_rows()
    print(f"Computing interface features for {len(rows)} complexes...")
    for i, r in enumerate(rows):
        try:
            r.update(interface_features(r["pep_pdb"], r["poc_pdb"]))
        except Exception as e:  # noqa: BLE001
            print(f"  [{r['pdb']}] FAIL: {e}")
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(rows)}")
    Path("/tmp/e0_features.json").write_text(json.dumps(rows))

    def col(key):
        return np.array([r.get(key, np.nan) for r in rows], float)

    y_all, L_all, bsa_all = col("y"), col("L"), col("bsa")

    def run(mask, label):
        yy, Lm, bm = y_all[mask], L_all[mask], bsa_all[mask]
        print(f"\n=== {label}  (n={int(mask.sum())}) ===")
        print(f"{'feature':<16}{'raw_r':>8}{'spear':>8}{'partial|L':>11}{'partial|BSA':>13}")
        for f in FEATS:
            v = col(f)[mask]
            ok = np.isfinite(v) & np.isfinite(yy)
            if ok.sum() < 5 or np.std(v[ok]) == 0:
                continue
            raw = pearsonr(v[ok], yy[ok]).statistic
            sp = spearmanr(v[ok], yy[ok]).statistic
            if f == "L":
                pl = pb = float("nan")
            else:
                okl = ok & np.isfinite(Lm)
                pl = pearsonr(residualize(v[okl], Lm[okl]),
                              residualize(yy[okl], Lm[okl])).statistic
                okb = ok & np.isfinite(bm)
                if okb.sum() >= 5 and np.std(bm[okb]) > 0:
                    pb = pearsonr(residualize(v[okb], bm[okb]),
                                  residualize(yy[okb], bm[okb])).statistic
                else:
                    pb = float("nan")
            flag = "  <==" if (f != "L" and abs(pl) >= 0.3) else ""
            print(f"{f:<16}{raw:>8.3f}{sp:>8.3f}{pl:>11.3f}{pb:>13.3f}{flag}")

    run(np.ones(len(rows), bool), "ALL (Kd+Ki)")
    run(np.array([r["aff"] == "Kd" for r in rows]), "Kd only")


if __name__ == "__main__":
    main()
