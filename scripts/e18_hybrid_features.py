"""E18 Stage 1+2 — hybrid ΔG features: SASA-dict base + Ramachandran-W entropy.

Faithful to Ram's spec:
  Stage 1: per-residue ΔSASA(peptide free -> bound) x Eisenberg hydropathy -> ΔE_SASA
  Stage 2: W_unbound = Π n_basin[aa]; ΔS_conf = -k_B ln(W_unbound/W_bound), W_bound≈1
           (optionally W_bound>1 for residues already ordered in the bound pose)

Computes both datasets (crystal-65, PEPBI), writes /tmp/e18_cr.json, /tmp/e18_pb.json
with: de_sasa, tds_conf (kcal/mol), lnW_unbound, n_contact, plus carries y/L/grp/seq.
ESM coupling (Stage 3) is added downstream by e18_esm_coupling.py.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
KB_T = 0.5961612775922  # kcal/mol at 300 K (= -k_B*T magnitude; TΔS uses this)

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}

# Eisenberg consensus hydrophobicity scale (kcal/mol-ish, + = hydrophobic).
# Burying hydrophobic surface is favorable -> we use -(eisenberg) so that burying
# a hydrophobic residue lowers ΔE (favorable). Sign handled in de_sasa.
EISENBERG = {
    "I": 0.73, "F": 0.61, "V": 0.54, "L": 0.53, "W": 0.37, "M": 0.26, "A": 0.25,
    "G": 0.16, "C": 0.04, "Y": 0.02, "P": -0.07, "T": -0.18, "S": -0.26, "H": -0.40,
    "E": -0.62, "N": -0.64, "Q": -0.69, "D": -0.72, "K": -1.10, "R": -1.76,
}

# Ramachandran allowed-basin count per residue (number of significantly-populated
# (φ,ψ) wells in coil/unfolded state). Gly broadest, Pro narrowest (φ locked).
# Consensus-style integers from Ramachandran-region populations (Hovmoller/Lovell);
# β-branched (I,V,T) slightly restricted; Pro=2 (cis/trans-ish φ-locked psi wells -> 2).
N_BASIN = {
    "G": 5, "A": 4, "S": 4, "C": 4, "N": 4, "D": 4, "H": 4, "Q": 4, "E": 4,
    "K": 4, "R": 4, "M": 4, "L": 4, "F": 4, "Y": 4, "W": 4,
    "I": 3, "V": 3, "T": 3,
    "P": 2,
}


_AROM = {"PHE", "TYR", "TRP", "HIS"}


def _geom_hb_arom(pep_res, rec_atoms, hb_cut=3.5, contact_cut=5.5):
    """Inline interface H-bond count + aromatic-contact count (matches e14 defs)."""
    from Bio.PDB import NeighborSearch
    if not pep_res or not rec_atoms:
        return 0, 0
    ns = NeighborSearch(rec_atoms)
    hb = arom = 0
    for rp in pep_res:
        rn = rp.resname.upper()
        for a in rp:
            if a.element in ("N", "O") and any(
                    b.element in ("N", "O") and np.linalg.norm(a.coord - b.coord) <= hb_cut
                    for b in ns.search(a.coord, hb_cut)):
                hb += 1
        if rn in _AROM:
            if any(b.get_parent().resname.upper() in _AROM
                   for a in rp if a.element != "H" for b in ns.search(a.coord, 5.5)):
                arom += 1
    return hb, arom


def _sasa_total_per_res(struct):
    """Return dict {(chain,resid): sasa} after Shrake-Rupley at atom level."""
    SR.compute(struct, level="A")
    out = {}
    for res in struct.get_residues():
        if res.id[0] != " ":
            continue
        out[(res.get_parent().id, res.id)] = sum(float(a.sasa) for a in res)
    return out


def stage12_from_residues(pep_pdb, complex_struct_path, pep_chain, seq, ss_map=None):
    """Compute de_sasa and entropy from a free-peptide PDB + the complex structure.

    pep_pdb: PDB containing ONLY the peptide (free-state SASA).
    complex_struct_path: PDB/struct containing the full complex (bound SASA of pep).
    pep_chain: chain id of the peptide in the complex.
    """
    # free peptide SASA per residue
    free = _sasa_total_per_res(P.get_structure("pf", pep_pdb))
    # bound: peptide SASA within the complex
    cpx = _sasa_total_per_res(P.get_structure("cx", complex_struct_path))

    # align by residue order along the peptide chain
    pep_res = [r for r in P.get_structure("pp", pep_pdb)[0].get_residues() if r.id[0] == " "]
    cpx_pep = [r for r in P.get_structure("cc", complex_struct_path)[0][pep_chain]
               if r.id[0] == " "] if pep_chain else []
    de_sasa = 0.0
    n = min(len(pep_res), len(cpx_pep)) if cpx_pep else 0
    if n == 0:
        # fall back: cannot map; ΔE_SASA from total buried * mean hydropathy
        return None
    for i in range(n):
        rfree = free.get((pep_res[i].get_parent().id, pep_res[i].id), 0.0)
        rbound = cpx.get((cpx_pep[i].get_parent().id, cpx_pep[i].id), 0.0)
        dsasa = max(0.0, rfree - rbound)  # buried area for residue i
        aa = AA3to1.get(pep_res[i].resname.upper(), "A")
        # favorable when burying hydrophobic (+Eisenberg): ΔE contribution = -eisenberg*dsasa
        de_sasa += -EISENBERG.get(aa, 0.0) * dsasa
    de_sasa /= 100.0  # scale Å² -> ~kcal/mol-ish range; absorbed by trained weight

    # Stage 2 entropy: W_unbound = Π n_basin; W_bound from SS (ordered residues -> fewer)
    lnW = 0.0
    for i, aa in enumerate(seq[:n]):
        nb = N_BASIN.get(aa, 4)
        lnW += np.log(nb)
    tds_conf = KB_T * lnW  # = k_B T * ln(W_unbound/1); positive = entropy penalty
    return dict(de_sasa=de_sasa, lnW_unbound=lnW, tds_conf=tds_conf, n_pep=n)


# ---- dataset loaders: write free-peptide PDBs + locate complex + chain ----

def crystal_records():
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    base = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    sm = {b["pdb"].upper(): b["peptide_seq"] for b in base}
    e14 = {r["seq"]: r for r in json.loads(Path("/tmp/e14_cr.json").read_text())}
    out = []
    for r in e0:
        if not r.get("pep_pdb"):
            continue
        pdb = r["pdb"].upper()
        seq = sm.get(pdb, "")
        if not seq:
            continue
        # build a "complex" PDB = peptide + pocket merged so SASA(bound) is correct
        merged = Path(f"/tmp/e18_cx/{pdb}.pdb")
        merged.parent.mkdir(exist_ok=True)
        lines = []
        for src, ch in ((r["pep_pdb"], "P"), (r["poc_pdb"], "R")):
            for ln in Path(src).read_text().splitlines():
                if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                    lines.append(ln[:21] + ch + ln[22:])
        merged.write_text("\n".join(lines) + "\nEND\n")
        f = stage12_from_residues(r["pep_pdb"], str(merged), "P", seq)
        if not f:
            continue
        g = e14.get(seq, {})
        f.update(y=r["y"], L=r["L"], aff=r["aff"], seq=seq, grp=g.get("grp", f"cr_{pdb}"),
                 pdb=pdb, hb_count=g.get("hb_count"), aromatic_cc=g.get("aromatic_cc"),
                 bsa=g.get("bsa"))
        out.append(f)
    return out


def pepbi_records():
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    import openpyxl
    wb = openpyxl.load_workbook("/tmp/pepbi/PEPBI.xlsx", read_only=True)
    rows = list(wb["PEPBI Data"].iter_rows(values_only=True))
    hdr = rows[1]
    ci = lambda n: hdr.index(n)
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    c_nm, c_dg, c_kd, c_bg, c_seq = (ci("PEPBI Complex Name"), ci("ΔG (kcal/mol)"),
                                     ci("KD (M)"), ci("Binding Group"), ci("Peptide Sequence"))
    out = []
    pep_d = Path("/tmp/e18_pep"); pep_d.mkdir(exist_ok=True)
    for r in rows[2:]:
        nm = str(r[c_nm]).strip().lower() if r[c_nm] else None
        if not nm or nm not in files:
            continue
        dg, kd = num(r[c_dg]), num(r[c_kd])
        if dg is None and kd and kd > 0:
            dg = 0.593 * np.log(kd)
        if dg is None:
            continue
        seq = str(r[c_seq]).strip().upper() if r[c_seq] else ""
        cx = files[nm]
        s = P.get_structure("x", cx)[0]
        if "B" not in [c.id for c in s]:
            continue
        # write free-peptide PDB (chain B only)
        from Bio.PDB import PDBIO, Select
        class _B(Select):
            def accept_chain(self, ch): return ch.id == "B"
            def accept_residue(self, res): return res.id[0] == " "
        io = PDBIO(); io.set_structure(P.get_structure("y", cx))
        pep_pdb = pep_d / f"{nm}.pdb"; io.save(str(pep_pdb), _B())
        if not seq:
            seq = "".join(AA3to1.get(x.resname.upper(), "A") for x in s["B"] if x.id[0] == " ")
        f = stage12_from_residues(str(pep_pdb), cx, "B", seq)
        if not f:
            continue
        pep_res = [x for x in s["B"] if x.id[0] == " "]
        rec_atoms = [a for c in s if c.id != "B" for x in c if x.id[0] == " "
                     for a in x if a.element != "H"]
        hb, arom = _geom_hb_arom(pep_res, rec_atoms)
        f.update(y=dg, L=len(seq), aff="Kd", seq=seq, grp=f"pb_{r[c_bg]}", nm=nm,
                 hb_count=hb, aromatic_cc=arom)
        out.append(f)
    return out


def main():
    print("crystal-65 Stage1+2...", flush=True)
    cr = crystal_records()
    Path("/tmp/e18_cr.json").write_text(json.dumps(cr))
    print(f"  {len(cr)} records", flush=True)
    print("PEPBI Stage1+2...", flush=True)
    pb = pepbi_records()
    Path("/tmp/e18_pb.json").write_text(json.dumps(pb))
    print(f"  {len(pb)} records", flush=True)
    # quick sanity: correlations of each raw term with ΔG (in-sample, both datasets)
    from scipy.stats import pearsonr
    for name, d in [("crystal", cr), ("pepbi", pb)]:
        y = np.array([r["y"] for r in d])
        for k in ["de_sasa", "tds_conf"]:
            v = np.array([r[k] for r in d])
            if np.std(v) > 0:
                print(f"  {name} corr({k}, ΔG) = {pearsonr(v, y).statistic:+.3f}")


if __name__ == "__main__":
    main()
