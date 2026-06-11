"""E42 — NET salt-bridge electrostatic term, done properly (the remaining gap, docs E41).

Physics: a buried charged group's net ΔG contribution =
    E_coulomb (screened attraction to receptor charges, FAVORABLE if complementary)
  + ΔG_desolvation (Born penalty for burying the charge, UNFAVORABLE, ∝ burial·q²)
The discriminating, universal signal is the BURIED UNSATISFIED charge: a charged residue
desolvated at the interface with NO complementary partner pays a large pure penalty (binds
weaker). This is the electrostatic analog of the hydrophobic effect (paired interaction+
desolvation), which is exactly why it should NOT sign-flip like a bare Coulomb count.

Per peptide charged group (Asp/Glu carboxylate −1, Lys/Arg/His amine +1/+0.5):
  burial_i  = ΔSASA(charged atoms, free→bound) / ref         how desolvated (0..1)
  coul_i    = Σ_recCharge 332·q_i·q_j/(4·r²)   (screened)     signed Coulomb (− favorable)
  desolv_i  = k·burial_i·q_i²                                 desolvation penalty (+)
  net_i     = coul_i + desolv_i
Aggregate (intensive + the killer count):
  e_sb_net      = Σ net_i / L          mean net electrostatic per residue
  e_coul        = Σ coul_i / L         favorable attraction only
  e_desolv      = Σ desolv_i / L       desolvation penalty only
  n_buried_unsat= # buried charged residues with NO partner (the discriminating penalty)
  n_salt_bridge = # satisfied salt bridges (buried + complementary partner)
Tests sign-consistency (cr vs 98) + whether it lifts the charged/helix complexes we fail on.
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

# Charged groups: residue -> (atom names, formal charge per group).
NEG = {"ASP": (["OD1", "OD2"], -1.0), "GLU": (["OE1", "OE2"], -1.0)}
POS = {"LYS": (["NZ"], +1.0), "ARG": (["NH1", "NH2", "NE"], +1.0), "HIS": (["ND1", "NE2"], +0.5)}
CHARGED = {**NEG, **POS}
K_DESOLV = 30.0  # Born-like desolvation scale (kcal/mol per unit q² at full burial); fit absorbs it


def _charge_groups(residues):
    """Return list of (center_xyz, q, atom_list) for charged groups among residues."""
    out = []
    for r in residues:
        rn = r.resname.upper()
        if rn in CHARGED:
            names, q = CHARGED[rn]
            ats = [a for a in r if a.name in names]
            if ats:
                out.append((np.mean([a.coord for a in ats], axis=0), q, r, ats))
    return out


def electrostatics(pep_pdb, rec_pdb):
    tmp = Path(f"/tmp/_e42_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (rec_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    try:
        # per-atom buried SASA of peptide (free vs bound), keyed by (res_index, atom_name)
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
        rec_res = [r for ch in cx if ch.id != "P" for r in ch if r.id[0] == " "]
        rec_charges = _charge_groups(rec_res)  # (xyz, q, res, atoms)
        rec_charge_atoms = [(g[0], g[1]) for g in rec_charges]
        # for partner detection use all receptor charged atom coords
        rec_chg_xyz = np.array([g[0] for g in rec_charges]) if rec_charges else np.zeros((0, 3))
        rec_chg_q = np.array([g[1] for g in rec_charges]) if rec_charges else np.zeros(0)

        L = len(pep)
        e_coul = e_desolv = e_net = 0.0
        n_unsat = n_sb = 0
        n_charged = 0
        for i, r in enumerate(pep):
            rn = r.resname.upper()
            if rn not in CHARGED:
                continue
            n_charged += 1
            names, q = CHARGED[rn]
            ats = [a for a in r if a.name in names]
            if not ats:
                continue
            center = np.mean([a.coord for a in ats], axis=0)
            # burial of the charged atoms (fraction desolvated)
            bur = 0.0; freesum = 0.0
            for a in ats:
                f = free.get((i, a.name), 0.0); b = bound.get((i, a.name), 0.0)
                bur += max(0.0, f - b); freesum += f
            bur_frac = bur / (freesum + 1e-6)
            # screened Coulomb to all receptor charged groups (distance-dependent dielectric 4r)
            coul = 0.0
            partner = False
            if len(rec_chg_xyz):
                d = np.linalg.norm(rec_chg_xyz - center, axis=1)
                m = d < 12.0
                coul = float(np.sum(332.0 * q * rec_chg_q[m] / (4.0 * d[m] ** 2 + 1e-6)))
                # complementary partner within salt-bridge distance (opposite sign, <4.5Å)
                partner = bool(np.any((d < 4.5) & (rec_chg_q * q < 0)))
            desolv = K_DESOLV * bur_frac * (q * q)
            net = coul + desolv
            e_coul += coul; e_desolv += desolv; e_net += net
            if bur_frac > 0.4 and not partner:
                n_unsat += 1                 # buried unsatisfied charge = pure penalty (killer)
            if bur_frac > 0.3 and partner:
                n_sb += 1                    # satisfied salt bridge
        return dict(
            e_sb_net=e_net / L, e_coul=e_coul / L, e_desolv=e_desolv / L,
            n_buried_unsat=float(n_unsat), n_salt_bridge=float(n_sb),
            frac_unsat=n_unsat / max(1, n_charged),
        )
    finally:
        tmp.unlink(missing_ok=True)


def build(which):
    out_path = Path(f"/tmp/e42_{which}.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if which == "cr":
        e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
        geo = json.loads(Path("/tmp/e19_cr.json").read_text())
        items = [(g["pdb"].upper(), e0[g["pdb"].upper()].get("pep_pdb"),
                  e0[g["pdb"].upper()].get("poc_pdb"), g["y"])
                 for g in geo if g["pdb"].upper() in e0 and e0[g["pdb"].upper()].get("pep_pdb")]
    else:
        e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
        work = Path("/tmp/ppep_work")
        items = [(k, str(work / f"{k}_pep.pdb"), str(work / f"{k}_rec.pdb"), r["y"])
                 for k, r in e28.items() if (work / f"{k}_pep.pdb").exists()]
    for key, pep, rec, y in items:
        if key in out or not pep:
            continue
        try:
            f = electrostatics(pep, rec)
            out[key] = dict(f, y=y)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}: {str(e)[:40]}", flush=True)
        out_path.write_text(json.dumps(out))
    return out


def main():
    print("=== computing electrostatics (crystal-65 + 98) ===", flush=True)
    cr = build("cr"); b98 = build("b98")
    ycr = np.array([cr[k]["y"] for k in cr]); y98 = np.array([b98[k]["y"] for k in b98])
    FEATS = ["e_sb_net", "e_coul", "e_desolv", "n_buried_unsat", "n_salt_bridge", "frac_unsat"]
    print(f"\ncr={len(cr)} b98={len(b98)}")
    print("=== sign-consistency (cr vs 98) — does NET electrostatics avoid the flip? ===")
    print(f"  {'feature':<16}{'crystal-65':>12}{'the-98':>10}{'universal?':>12}")
    keep = []
    for f in FEATS:
        vc = np.array([cr[k][f] for k in cr]); v9 = np.array([b98[k][f] for k in b98])
        rc = pearsonr(vc, ycr).statistic if vc.std() > 0 else 0
        r9 = pearsonr(v9, y98).statistic if v9.std() > 0 else 0
        ok = rc * r9 > 0 and min(abs(rc), abs(r9)) > 0.1
        if ok:
            keep.append(f)
        print(f"  {f:<16}{rc:>+12.3f}{r9:>+10.3f}{('YES' if ok else 'flip/weak'):>12}")
    print(f"  universal electrostatic features: {keep}")

    # does it improve the pooled model on top of geometry+entropy?
    inten = json.loads(Path("/tmp/e31_intensive.json").read_text())
    geo = json.loads(Path("/tmp/e19_cr.json").read_text())
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    s_cr = json.loads(Path("/tmp/e40_cr.json").read_text())
    s_b = json.loads(Path("/tmp/e40_b98.json").read_text())
    UNI = ["bsa_hyd", "mj_per_contact", "f_hyd_iface", "frac_pol_satisfied"]

    def join(rows_inten, keys, sf, el):
        out = []
        for it, k in zip(rows_inten, keys):
            if k in sf and k in el:
                d = dict(it)
                d["s_free_bur"] = sf[k]["s_free"] * min(1.0, it.get("f_hyd_iface", 0.5))
                for f in FEATS:
                    d[f] = el[k][f]
                d["y"] = el[k]["y"]
                out.append(d)
        return out
    crk = [r["pdb"].upper() for r in geo]; bk = list(e28)
    poolc = join(inten["cr"], crk, s_cr, cr) + join(inten["b98"], bk, s_b, b98)
    y = np.array([r["y"] for r in poolc])

    def loo(feats):
        X = np.array([[r.get(f, 0.0) for f in feats] for r in poolc]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic, np.sqrt(((p - y) ** 2).mean())
    print(f"\n=== pooled n={len(poolc)}: does electrostatics lift geometry+entropy? ===")
    base = UNI + ["s_free_bur"]
    for nm, fs in [("geometry+entropy [baseline]", base), ("+ e_sb_net", base + ["e_sb_net"]),
                   ("+ n_buried_unsat", base + ["n_buried_unsat"]),
                   ("+ frac_unsat", base + ["frac_unsat"]),
                   ("+ all universal elec", base + keep)]:
        r, e = loo(fs); print(f"  {nm:<30} r={r:+.3f} RMSE={e:.2f}")
    print("  baseline geometry+entropy 0.488 | Rosetta cross-target ~0.42")


if __name__ == "__main__":
    main()
