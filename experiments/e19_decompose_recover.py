"""E19 — three unexplored attacks on the per-protein-baseline wall (instant, no GPU).

Decompose ΔG_ij = alpha_j (receptor baseline) + sum beta·x_ij (peptide interface).

Emits per complex a RICH feature dict so the eval can test:
  (1) BASELINE RECOVERY: predict per-receptor baseline alpha_j from POCKET descriptors
      (composition fractions, charge, aromatics, size) -> add it back for absolute ΔG.
  (2) LEARNED FAVORABILITY: hb·ΔSASA, saltbridge·ΔSASA, unsat_polar·ΔSASA, apolar·ΔSASA
      as SEPARATE features (regression learns the signs) instead of hand weights.
  (3) LIGAND EFFICIENCY: every interface feature also divided by L (size normalization).

Features written to /tmp/e19_{cr,pb}.json (crash-safe). Eval is e19_eval.py.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

import warnings
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from e18_hybrid_features import AA3to1, EISENBERG  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
POS = {"ARG", "LYS", "HIS"}
NEG = {"ASP", "GLU"}
CHARGED = POS | NEG
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "GLY"}
AROM = {"PHE", "TYR", "TRP", "HIS"}
HPHOBIC_AA = set("AVLIMFWC")


def _per_res_sasa(struct):
    SR.compute(struct, level="A")
    return {(r.get_parent().id, r.id): sum(float(a.sasa) for a in r)
            for r in struct.get_residues() if r.id[0] == " "}


def pocket_descriptors(cx_model, pep_chain, radius=8.0):
    """Composition of receptor residues within `radius` of the peptide (size-normalized)."""
    pep_xyz = np.array([a.coord for r in cx_model[pep_chain] if r.id[0] == " "
                        for a in r if a.element != "H"])
    if len(pep_xyz) == 0:
        return None
    r2 = radius * radius
    poc = []
    for ch in cx_model:
        if ch.id == pep_chain:
            continue
        for res in ch:
            if res.id[0] != " ":
                continue
            for a in res:
                if a.element != "H" and np.min(((pep_xyz - a.coord) ** 2).sum(1)) <= r2:
                    poc.append(res.resname.upper()); break
    n = len(poc)
    if n == 0:
        return None
    names = [AA3to1.get(x, "A") for x in poc]
    f_hyd = sum(a in HPHOBIC_AA for a in names) / n
    f_pos = sum(x in POS for x in poc) / n
    f_neg = sum(x in NEG for x in poc) / n
    f_arom = sum(x in AROM for x in poc) / n
    f_pol = sum(x in POLAR for x in poc) / n
    mean_eis = float(np.mean([EISENBERG.get(a, 0.0) for a in names]))
    return dict(poc_n=float(n), poc_f_hyd=f_hyd, poc_f_pos=f_pos, poc_f_neg=f_neg,
                poc_net=f_pos - f_neg, poc_f_arom=f_arom, poc_f_pol=f_pol, poc_eis=mean_eis)


def interface_features(pep_pdb, cx_path, pep_chain, L):
    """Decomposed ΔSASA-by-favorability + counts; instant geometry."""
    free = _per_res_sasa(P.get_structure("f", pep_pdb))
    cpx = _per_res_sasa(P.get_structure("c", cx_path))
    cx = P.get_structure("cc", cx_path)[0]
    pep_res = [r for r in cx[pep_chain] if r.id[0] == " "]
    rec_atoms = [a for ch in cx if ch.id != pep_chain for r in ch if r.id[0] == " "
                 for a in r if a.element != "H"]
    if not pep_res or not rec_atoms:
        return None
    ns = NeighborSearch(rec_atoms)
    pf = [r for r in P.get_structure("pp", pep_pdb)[0].get_residues() if r.id[0] == " "]
    n = min(len(pep_res), len(pf))
    # decomposed buried-SASA accumulators
    bsa_tot = bsa_hyd = bsa_pol = 0.0
    sasa_hb = sasa_sb = sasa_apolar = sasa_unsat = 0.0
    hb_count = sb_count = arom_cc = clash = 0
    for i in range(n):
        rc = pep_res[i]; rn = rc.resname.upper(); aa = AA3to1.get(rn, "A")
        rfree = free.get((pf[i].get_parent().id, pf[i].id), 0.0)
        rbound = cpx.get((rc.get_parent().id, rc.id), 0.0)
        dsasa = max(0.0, rfree - rbound)
        if dsasa < 1.0:
            continue
        bsa_tot += dsasa
        if aa in HPHOBIC_AA:
            bsa_hyd += dsasa
        if rn in POLAR or rn in CHARGED:
            bsa_pol += dsasa
        has_hb = has_sb = has_apolar = has_arom = False
        unsat_polar = (rn in POLAR or rn in CHARGED)
        for a in rc:
            if a.element == "H":
                continue
            for b in ns.search(a.coord, 5.5):
                d = float(np.linalg.norm(a.coord - b.coord))
                brn = b.get_parent().resname.upper()
                if d < 2.5:
                    clash += 1
                if a.element in ("N", "O") and b.element in ("N", "O") and d <= 3.5:
                    has_hb = True; unsat_polar = False
                if rn in CHARGED and a.element in ("N", "O") and d <= 4.0:
                    want = NEG if rn in POS else POS
                    if brn in want:
                        has_sb = True; unsat_polar = False
                if rn in APOLAR and brn in APOLAR and d <= 5.0:
                    has_apolar = True
                if rn in AROM and brn in AROM and d <= 5.5:
                    has_arom = True
        if has_hb:
            sasa_hb += dsasa; hb_count += 1
        if has_sb:
            sasa_sb += dsasa; sb_count += 1
        if has_apolar:
            sasa_apolar += dsasa
        if has_arom:
            arom_cc += 1
        if unsat_polar:
            sasa_unsat += dsasa
    f = dict(bsa_tot=bsa_tot/100, bsa_hyd=bsa_hyd/100, bsa_pol=bsa_pol/100,
             sasa_hb=sasa_hb/100, sasa_sb=sasa_sb/100, sasa_apolar=sasa_apolar/100,
             sasa_unsat=sasa_unsat/100, hb_count=float(hb_count), sb_count=float(sb_count),
             arom_cc=float(arom_cc), clash=float(clash), L=float(L))
    # ligand-efficiency (per-residue) versions
    Lf = max(1.0, float(L))
    for k in ["bsa_tot", "bsa_hyd", "bsa_pol", "sasa_hb", "sasa_sb", "sasa_apolar",
              "sasa_unsat", "hb_count", "sb_count", "arom_cc"]:
        f[k + "_pr"] = f[k] / Lf
    return f


# ---------------- loaders ----------------

def crystal_records():
    out_path = Path("/tmp/e19_cr.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["pdb"] for r in out}
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    v1 = {r["pdb"]: r for r in json.loads(Path("/tmp/e18_cr.json").read_text())}
    for r in e0:
        pdb = r["pdb"].upper()
        if not r.get("pep_pdb") or pdb in done or pdb not in v1:
            continue
        b = v1[pdb]; L = len(b["seq"])
        merged = Path(f"/tmp/e18v3_cx/{pdb}.pdb")
        if not merged.exists():
            lines = []
            for src, ch in ((r["pep_pdb"], "P"), (r["poc_pdb"], "R")):
                for ln in Path(src).read_text().splitlines():
                    if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                        lines.append(ln[:21] + ch + ln[22:])
            merged.parent.mkdir(exist_ok=True)
            merged.write_text("\n".join(lines) + "\nEND\n")
        try:
            cxm = P.get_structure("m", str(merged))[0]
            fi = interface_features(r["pep_pdb"], str(merged), "P", L)
            pk = pocket_descriptors(cxm, "P")
        except Exception as e:  # noqa: BLE001
            print(f"  cr {pdb} FAIL {type(e).__name__}", flush=True); continue
        if not fi or not pk:
            continue
        rec = dict(pdb=pdb, y=b["y"], seq=b["seq"], grp=f"cr_{pdb}", **fi, **pk)
        out.append(rec); out_path.write_text(json.dumps(out))
    return out


class _B(Select):
    def accept_chain(self, ch): return ch.id == "B"
    def accept_residue(self, res): return res.id[0] == " "


def pepbi_records():
    out_path = Path("/tmp/e19_pb.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["nm"] for r in out}
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    v1 = json.loads(Path("/tmp/e18_pb.json").read_text())
    pep_d = Path("/tmp/e19_pep"); pep_d.mkdir(exist_ok=True)
    for b in v1:
        nm = b["nm"]
        if nm in done or nm not in files:
            continue
        cx = files[nm]; L = len(b["seq"])
        try:
            s = P.get_structure("x", cx)[0]
            if "B" not in [c.id for c in s]:
                continue
            io = PDBIO(); io.set_structure(P.get_structure("y", cx))
            pep_pdb = pep_d / f"{nm}.pdb"; io.save(str(pep_pdb), _B())
            fi = interface_features(str(pep_pdb), cx, "B", L)
            pk = pocket_descriptors(s, "B")
        except Exception as e:  # noqa: BLE001
            print(f"  pb {nm} FAIL {type(e).__name__}", flush=True); continue
        if not fi or not pk:
            continue
        out.append(dict(nm=nm, y=b["y"], seq=b["seq"], grp=b["grp"], **fi, **pk))
        if len(out) % 25 == 0:
            print(f"  pb {len(out)}", flush=True)
        out_path.write_text(json.dumps(out))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "cr"):
        print("=== crystal features ===", flush=True); print(len(crystal_records()), "cr")
    if which in ("both", "pb"):
        print("=== pepbi features ===", flush=True); print(len(pepbi_records()), "pb")


if __name__ == "__main__":
    main()
