"""E31 — FIX the sign-flipping features: intensive + desolvation-aware encodings.

Hypothesis (Ram): physics is universal; flips are a Simpson's/size confound in our EXTENSIVE
encodings. Recompute INTENSIVE + desolvation-aware features on BOTH crystal-65 and the 98,
and test whether they STOP flipping (consistent sign across datasets) — then they can be added
back instead of letting hydrophobic burial carry everything.

Intensive / physics-correct features:
  mj_per_contact   : mean MJ contact energy (composition quality, not size)
  f_hyd_iface      : hydrophobic / total buried SASA (intensive hydrophobicity)
  frac_pol_satisfied: buried peptide polar atoms that ARE H-bonded / all buried polar atoms
  net_polar_perL   : (satisfied - unsatisfied buried polar atoms)/L  (desolvation-aware)
  unsat_pol_perL   : buried UNSATISFIED polar atoms / L  (desolvation PENALTY, expect +)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from hybridock_pep.scoring.mj_potential import MJ_ENERGY  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
HYD = set("AVLIMFWC")


def intensive_features(pep_pdb, rec_pdb):
    """Compute intensive + desolvation-aware features from peptide + receptor PDBs."""
    # merged complex
    tmp = Path(f"/tmp/_e31_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (rec_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    try:
        cx = P.get_structure("c", str(tmp))[0]
        # Free peptide SASA keyed by (residue_index, atom_name) — chain-agnostic so it
        # matches the complex peptide (chain P) regardless of the source chain id.
        pep_free = P.get_structure("f", str(pep_pdb))[0]
        SR.compute(P.get_structure("ff", str(pep_pdb)), level="A")
        pf = P.get_structure("ff2", str(pep_pdb)); SR.compute(pf, level="A")
        free = {}
        for i, r in enumerate(rr for rr in pf.get_residues() if rr.id[0] == " "):
            for a in r:
                free[(i, a.name)] = float(a.sasa)
        cxs = P.get_structure("cc2", str(tmp)); SR.compute(cxs, level="A")
        bound = {}
        for i, r in enumerate(rr for rr in cxs[0]["P"] if rr.id[0] == " "):
            for a in r:
                bound[(i, a.name)] = float(a.sasa)
        pep = [r for r in cx["P"] if r.id[0] == " "]
        rec_atoms = [a for ch in cx if ch.id != "P" for r in ch if r.id[0] == " "
                     for a in r if a.element != "H"]
        if not pep or not rec_atoms:
            return None
        ns = NeighborSearch(rec_atoms)
        L = len(pep)
        # contacts + MJ
        seen = set(); mj_vals = []
        hyd_bur = 0.0; tot_bur = 0.0
        pol_buried = 0; pol_satisfied = 0
        for idx, rp in enumerate(pep):
            a1 = AA3to1.get(rp.resname.upper(), "A")
            nbr = set()
            for atom in rp:
                if atom.element == "H":
                    continue
                d_b = max(0.0, free.get((idx, atom.name), 0.0) - bound.get((idx, atom.name), 0.0))
                tot_bur += d_b
                if a1 in HYD:
                    hyd_bur += d_b
                # polar atom satisfaction (desolvation): buried N/O of peptide
                if atom.element in ("N", "O") and d_b > 1.0:
                    pol_buried += 1
                    if any(b.element in ("N", "O") and np.linalg.norm(atom.coord-b.coord) <= 3.5
                           for b in ns.search(atom.coord, 3.5)):
                        pol_satisfied += 1
                for b in ns.search(atom.coord, 6.5):
                    nbr.add(b.get_parent())
            for rr in nbr:
                k = (id(rp), id(rr))
                if k in seen:
                    continue
                seen.add(k)
                mj_vals.append(MJ_ENERGY.get((a1, AA3to1.get(rr.resname.upper(), "A")), -1.5))
        nC = max(1, len(mj_vals))
        unsat = pol_buried - pol_satisfied
        return dict(
            mj_per_contact=float(np.mean(mj_vals)) if mj_vals else 0.0,
            f_hyd_iface=hyd_bur / (tot_bur + 1e-6),
            frac_pol_satisfied=pol_satisfied / max(1, pol_buried),
            net_polar_perL=(pol_satisfied - unsat) / L,
            unsat_pol_perL=unsat / L,
            bsa_hyd=hyd_bur / 100.0,
        )
    finally:
        tmp.unlink(missing_ok=True)


def build(which):
    if which == "cr":
        e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
        geo = json.loads(Path("/tmp/e19_cr.json").read_text())
        out = []
        for g in geo:
            pdb = g["pdb"].upper()
            if pdb not in e0 or not e0[pdb].get("pep_pdb"):
                continue
            merged = Path(f"/tmp/e18v3_cx/{pdb}.pdb")
            # crystal: peptide=pep_pdb, receptor=poc_pdb
            f = intensive_features(e0[pdb]["pep_pdb"], e0[pdb]["poc_pdb"])
            if f:
                out.append(dict(f, y=g["y"]))
        return out
    else:
        b98 = json.loads(Path("/tmp/e28_feats.json").read_text())
        work = Path("/tmp/ppep_work")
        out = []
        for key, r in b98.items():
            pepf = work / f"{key}_pep.pdb"; recf = work / f"{key}_rec.pdb"
            if pepf.exists() and recf.exists():
                f = intensive_features(pepf, recf)
                if f:
                    out.append(dict(f, y=r["y"]))
        return out


def main():
    print("computing intensive features (crystal-65 + 98)...", flush=True)
    cr = build("cr"); b98 = build("b98")
    json.dump(dict(cr=cr, b98=b98), open("/tmp/e31_intensive.json", "w"))
    print(f"cr={len(cr)} b98={len(b98)}\n")
    feats = ["bsa_hyd", "mj_per_contact", "f_hyd_iface", "frac_pol_satisfied",
             "net_polar_perL", "unsat_pol_perL"]
    ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])
    print("=== SIGN-CONSISTENCY: raw corr(feature, ΔG), crystal-65 vs 98 ===")
    print(f"  {'feature':<20}{'crystal-65':>12}{'the-98':>10}{'consistent?':>13}")
    keep = []
    for f in feats:
        rc = pearsonr([r[f] for r in cr], ycr).statistic
        r9 = pearsonr([r[f] for r in b98], y98).statistic
        ok = (rc * r9 > 0) and abs(rc) > 0.1 and abs(r9) > 0.1
        if ok:
            keep.append(f)
        print(f"  {f:<20}{rc:>+12.3f}{r9:>+10.3f}{('YES' if ok else 'flip/weak'):>13}")
    print(f"\n  universal (consistent) intensive features: {keep}")

    # transfer test with the fixed universal set vs hydrophobic-burial-only
    def transfer(feats_):
        Xtr = np.array([[r[f] for f in feats_] for r in cr])
        Xte = np.array([[r[f] for f in feats_] for r in b98])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        w, *_ = np.linalg.lstsq(A, ycr, rcond=None)
        pred = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
        return pearsonr(pred, y98).statistic, np.sqrt(((pred - y98) ** 2).mean())
    print("\n=== cr -> 98 TRANSFER: did fixing the features recover signal? ===")
    r0, e0_ = transfer(["bsa_hyd"]); print(f"  bsa_hyd only:           r={r0:+.3f} RMSE={e0_:.2f}")
    if keep:
        rk, ek = transfer(keep); print(f"  fixed universal set:    r={rk:+.3f} RMSE={ek:.2f}")
    rall, eall = transfer(feats); print(f"  all intensive (6):      r={rall:+.3f} RMSE={eall:.2f}")
    print("  >> if fixed set > bsa_hyd alone, the intensive encoding RECOVERED the lost features")


if __name__ == "__main__":
    main()
