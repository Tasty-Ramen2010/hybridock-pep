"""E32 — proper PHYSICS terms (Ram: physics never lies; capture it correctly, no proxies).

Our counts flip because they capture only the FAVORABLE half of each interaction, missing the
DESOLVATION penalty that opposes it. Implement the real net free-energy terms and test whether
they are UNIVERSAL (sign-consistent + strong on BOTH crystal-65 and the independent 98):

  g_desolv  : Σ_buried σ_atomtype · ΔSASA   (Eisenberg-McLachlan atomic solvation;
              apolar burial FAVORABLE, polar/charged burial PENALIZED — the net, not a count)
  g_elec    : screened Coulomb between peptide & receptor formal charges (dist-dependent ε)
  g_vdw_cmpl: shape/packing complementarity (buried-area / gap proxy)
These are net energies; the desolvation penalty should make them NOT flip.
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
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()

# Eisenberg-McLachlan (1986) atomic solvation parameters, cal/mol/Å^2.
# POSITIVE = burial favorable (apolar); NEGATIVE = burial unfavorable (polar/charged).
SIGMA_C = 16.0       # carbon
SIGMA_S = 21.0       # sulfur
SIGMA_NO = -6.0      # neutral N/O
SIGMA_O_MINUS = -24.0  # carboxylate O (Asp/Glu)
SIGMA_N_PLUS = -50.0   # charged N (Lys/Arg)

NEG_RES = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
POS_RES = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"]}
FORMAL = {"ASP": -1.0, "GLU": -1.0, "LYS": +1.0, "ARG": +1.0, "HIS": +0.5}


def atom_sigma(resname, atom):
    rn = resname.upper(); nm = atom.name
    if atom.element == "C":
        return SIGMA_C
    if atom.element == "S":
        return SIGMA_S
    if rn in NEG_RES and nm in NEG_RES[rn]:
        return SIGMA_O_MINUS
    if rn in POS_RES and nm in POS_RES[rn]:
        return SIGMA_N_PLUS
    if atom.element in ("N", "O"):
        return SIGMA_NO
    return 0.0


def charge_center(res):
    rn = res.resname.upper()
    if rn in NEG_RES:
        ats = [a for a in res if a.name in NEG_RES[rn]]
    elif rn in POS_RES:
        ats = [a for a in res if a.name in POS_RES[rn]]
    else:
        return None
    if not ats:
        return None
    return np.mean([a.coord for a in ats], axis=0), FORMAL.get(rn, 0.0)


def physics(pep_pdb, rec_pdb):
    tmp = Path(f"/tmp/_e32_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (rec_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    try:
        # per-atom buried SASA of the peptide (free vs bound), keyed by index
        SR.compute((pf := P.get_structure("f", str(pep_pdb))), level="A")
        free = {}
        for i, r in enumerate(rr for rr in pf.get_residues() if rr.id[0] == " "):
            for a in r:
                free[(i, a.name)] = float(a.sasa)
        SR.compute((cxs := P.get_structure("c", str(tmp))), level="A")
        cx = cxs[0]
        pep = [r for r in cx["P"] if r.id[0] == " "]
        bound = {}
        for i, r in enumerate(pep):
            for a in r:
                bound[(i, a.name)] = float(a.sasa)
        # desolvation: Σ σ · buried SASA (peptide side)
        g_desolv = 0.0
        for i, r in enumerate(pep):
            for a in r:
                if a.element == "H":
                    continue
                db = max(0.0, free.get((i, a.name), 0.0) - bound.get((i, a.name), 0.0))
                g_desolv += atom_sigma(r.resname, a) * db
        g_desolv /= 1000.0  # cal -> kcal-ish scale
        # electrostatics: screened Coulomb between peptide & receptor charged centers
        rec_res = [r for ch in cx if ch.id != "P" for r in ch if r.id[0] == " "]
        pc = [c for c in (charge_center(r) for r in pep) if c]
        rc = [c for c in (charge_center(r) for r in rec_res) if c]
        g_elec = 0.0
        for (xp, qp) in pc:
            for (xr, qr) in rc:
                d = float(np.linalg.norm(xp - xr))
                if d < 12.0:
                    eps = 4.0 * d  # distance-dependent dielectric
                    g_elec += 332.0 * qp * qr / (eps * d + 1e-6)
        return dict(g_desolv=g_desolv, g_elec=g_elec)
    finally:
        tmp.unlink(missing_ok=True)


def build(which):
    if which == "cr":
        e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
        geo = json.loads(Path("/tmp/e19_cr.json").read_text())
        out = []
        for g in geo:
            pdb = g["pdb"].upper()
            if pdb in e0 and e0[pdb].get("pep_pdb"):
                f = physics(e0[pdb]["pep_pdb"], e0[pdb]["poc_pdb"])
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
                f = physics(pepf, recf)
                if f:
                    out.append(dict(f, y=r["y"]))
        return out


def main():
    print("computing real-physics terms (desolvation + electrostatics)...", flush=True)
    cr = build("cr"); b98 = build("b98")
    json.dump(dict(cr=cr, b98=b98), open("/tmp/e32_physics.json", "w"))
    inten = json.load(open("/tmp/e31_intensive.json"))
    # attach intensive bsa for combined test (align by index/y)
    ycr = np.array([r["y"] for r in cr]); y98 = np.array([r["y"] for r in b98])
    print(f"cr={len(cr)} b98={len(b98)}\n")
    print("=== SIGN-CONSISTENCY of REAL physics terms (do they flip?) ===")
    print(f"  {'term':<12}{'crystal-65':>12}{'the-98':>10}{'universal?':>12}")
    for f in ["g_desolv", "g_elec"]:
        rc = pearsonr([r[f] for r in cr], ycr).statistic
        r9 = pearsonr([r[f] for r in b98], y98).statistic
        ok = rc * r9 > 0 and abs(rc) > 0.1 and abs(r9) > 0.1
        print(f"  {f:<12}{rc:>+12.3f}{r9:>+10.3f}{('YES' if ok else 'flip/weak'):>12}")
    # transfer with physics terms + intensive bsa
    crm = [dict(a, **b) for a, b in zip(cr, inten["cr"])]
    b98m = [dict(a, **b) for a, b in zip(b98, inten["b98"])]

    def transfer(feats):
        Xtr = np.array([[r[f] for f in feats] for r in crm])
        Xte = np.array([[r[f] for f in feats] for r in b98m])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); w, *_ = np.linalg.lstsq(A, ycr, rcond=None)
        pr = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
        return pearsonr(pr, y98).statistic, np.sqrt(((pr - y98) ** 2).mean())

    def loo(rows, feats, y):
        X = np.array([[r[f] for f in feats] for r in rows]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic
    print("\n=== does adding real physics IMPROVE transfer & pooled? ===")
    pooled = crm + b98m; yp = np.concatenate([ycr, y98])
    for nm, fs in [("bsa_hyd (best universal)", ["bsa_hyd"]),
                   ("g_desolv only", ["g_desolv"]),
                   ("g_desolv + g_elec", ["g_desolv", "g_elec"]),
                   ("desolv+elec+bsa+mjpc", ["g_desolv", "g_elec", "bsa_hyd", "mj_per_contact"])]:
        rt, et = transfer(fs); rp = loo(pooled, fs, yp)
        print(f"  {nm:<26} transfer r={rt:+.3f}  pooled-163 LOO r={rp:+.3f}")
    print("  PPI-Affinity 0.554 | prior best universal pooled 0.421")


if __name__ == "__main__":
    main()
