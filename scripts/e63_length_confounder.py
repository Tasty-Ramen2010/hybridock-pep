"""E63 — find the LURKING VARIABLE behind length's sign-flip (Ram's hypothesis).

Length isn't physics; it ALIASES different things in each dataset. Hypothesis: cr65 = compact strong
binders (length = real contacts); the-98 = long floppy tails (length = disorder/entropy cost). Find the
feature whose coupling to LENGTH is most OPPOSITE between datasets — that's what length stands for — then
model THAT directly (reward contacts, penalize tails) so it's sign-stable.

Plan:
 1. Rich feature catalog computed uniformly on both datasets (seq + per-residue burial + shape + physics).
 2. corr(feature, LENGTH) on cr65 vs the-98 -> rank by |difference| = the confounders riding length.
 3. corr(feature, ΔG) per dataset -> which flip, which transfer.
 4. dataset 'personality': mean feature values cr65 vs the-98 (is 98 really tail-heavy?).
 5. CONSTRUCTIVE: model ΔG ~ contact-reward + tail-penalty; does it transfer where raw L fails?
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
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e63_catalog.json")
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def burials_and_shape(pep_pdb, rec_pdb):
    pep = P.get_structure("pep", str(pep_pdb))[0]
    SR.compute(pep, level="R")
    free = [(r.resname, r.id[1], r.sasa, r) for r in pep.get_residues() if r.id[0] == " "]
    rec = P.get_structure("rec", str(rec_pdb))[0]
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    cx = Structure("cx"); m = Model(0); cx.add(m); used = set()
    for ch in list(pep.get_chains()) + list(rec.get_chains()):
        cid = ch.id
        while cid in used:
            cid = chr((ord(cid) + 1) % 90 + 33)
        used.add(cid); ch2 = ch.copy(); ch2.id = cid; m.add(ch2)
    SR.compute(cx, level="R")
    comp = {}
    for ch in cx.get_chains():
        for r in ch.get_residues():
            if r.id[0] == " ":
                comp.setdefault((r.resname, r.id[1]), r.sasa)
    bur = []
    for rn, seq, fs, r in free:
        cs = comp.get((rn, seq), fs)
        bur.append((rn, max(0.0, fs - cs)))
    # shape: Rg from CA
    cas = np.array([a.coord for r in pep.get_residues() if r.id[0] == " " for a in r if a.name == "CA"])
    if len(cas) >= 2:
        rg = float(np.sqrt(((cas - cas.mean(0)) ** 2).sum(1).mean()))
        e2e = float(np.linalg.norm(cas[0] - cas[-1]))
    else:
        rg = e2e = 0.0
    return bur, rg, e2e


def seqfeat(seq):
    L = max(1, len(seq))
    return dict(
        charged_frac=sum(c in "DEKR" for c in seq) / L,
        net_charge=seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
        abs_net_charge=abs(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")),
        hyd_frac=sum(c in "AILMFVWC" for c in seq) / L,
        arom_frac=sum(c in "FWYH" for c in seq) / L,
        bulky_frac=sum(c in "FWYLIM" for c in seq) / L,
        pro_frac=seq.count("P") / L, gly_frac=seq.count("G") / L,
        progly_frac=(seq.count("P") + seq.count("G")) / L,
        small_frac=sum(c in "AGS" for c in seq) / L,
        polar_frac=sum(c in "STNQHY" for c in seq) / L,
        gravy=float(np.mean([KD.get(c, 0) for c in seq])),
    )


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    rows = {}
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    e49c = json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text())
    # crystal-65
    for r in bench:
        try:
            seq = r["peptide_seq"]
            bur, rg, e2e = burials_and_shape(r["peptide_pdb"], r["pocket_pdb"])
            rows[f"cr_{r['pdb']}"] = mk(seq, bur, rg, e2e, r["dg_exp"], "cr65",
                                        e49c.get(r["pdb"].upper(), {}))
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} fail {str(e)[:40]}", flush=True)
    # the-98
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if not (pep.exists() and rec.exists()):
            continue
        try:
            bur, rg, e2e = burials_and_shape(pep, rec)
            rows[f"98_{k}"] = mk(v["seq"], bur, rg, e2e, v["y"], "the98", v)
        except Exception as e:  # noqa: BLE001
            print(f"  98 {k} fail {str(e)[:40]}", flush=True)
    CACHE.write_text(json.dumps(rows))
    return rows


def mk(seq, bur, rg, e2e, y, ds, phys):
    L = len(bur) if bur else len(seq)
    areas = np.array([b for _, b in bur]) if bur else np.zeros(1)
    f = dict(ds=ds, y=y, L=L, seq=seq, rg=rg, rg_per_L=rg / max(1, L),
             e2e=e2e, e2e_per_L=e2e / max(1, L),
             total_bsa=float(areas.sum()), mean_burial=float(areas.mean()),
             max_burial=float(areas.max()),
             n_anchor=int((areas > 40).sum()),
             n_nonbinding=int((areas <= 40).sum()),          # tail/wasted residues (COUNT)
             nonbind_frac=float((areas <= 40).mean()),        # tail fraction (INTENSIVE)
             n_dangling=int((areas <= 10).sum()),             # truly exposed
             dangling_frac=float((areas <= 10).mean()),
             buried_frac=float((areas > 40).mean()))
    f.update(seqfeat(seq))
    f["mmgbsa"] = phys.get("dg_single", np.nan)
    f["eint"] = phys.get("e_int_mean", np.nan)
    f["mtds"] = phys.get("minus_tds", np.nan)
    return f


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[m], b[m]).statistic if m.sum() > 5 else np.nan


def main():
    rows = list(build().values())
    cr = [r for r in rows if r["ds"] == "cr65"]
    t98 = [r for r in rows if r["ds"] == "the98"]
    print(f"=== E63 length-confounder hunt.  cr65={len(cr)}  the98={len(t98)} ===")

    feats = ["total_bsa", "mean_burial", "max_burial", "n_anchor", "n_nonbinding", "nonbind_frac",
             "n_dangling", "dangling_frac", "buried_frac", "rg", "rg_per_L", "e2e_per_L",
             "charged_frac", "net_charge", "hyd_frac", "arom_frac", "bulky_frac", "pro_frac",
             "gly_frac", "progly_frac", "small_frac", "polar_frac", "gravy",
             "mmgbsa", "eint", "mtds"]

    print("\n=== (A) corr(feature, LENGTH) per dataset — what does length STAND FOR? ===")
    print(f"{'feature':<15}{'cr65':>9}{'the98':>9}{'|diff|':>9}   interpretation")
    Lcr = [r["L"] for r in cr]; Lt = [r["L"] for r in t98]
    diffs = []
    for f in feats:
        ccr = sp([r[f] for r in cr], Lcr); ct = sp([r[f] for r in t98], Lt)
        if np.isnan(ccr) or np.isnan(ct):
            continue
        diffs.append((f, ccr, ct, abs(ccr - ct)))
    for f, ccr, ct, d in sorted(diffs, key=lambda t: -t[3])[:12]:
        tag = "<== length aliases this OPPOSITELY" if (ccr * ct < 0 and d > 0.4) else ""
        print(f"  {f:<13}{ccr:>+9.2f}{ct:>+9.2f}{d:>9.2f}   {tag}")

    print("\n=== (B) corr(feature, ΔG) per dataset — which flip vs which transfer ===")
    print(f"{'feature':<15}{'cr65':>9}{'the98':>9}  stable?")
    for f in feats:
        ccr = sp([r[f] for r in cr], [r["y"] for r in cr])
        ct = sp([r[f] for r in t98], [r["y"] for r in t98])
        if np.isnan(ccr) or np.isnan(ct):
            continue
        st = "YES" if ccr * ct > 0 else "flip"
        mark = "  <== TRANSFERS" if (ccr * ct > 0 and min(abs(ccr), abs(ct)) > 0.15) else ""
        print(f"  {f:<13}{ccr:>+9.2f}{ct:>+9.2f}  {st}{mark}")

    print("\n=== (C) DATASET PERSONALITY: mean feature value (is 98 tail-heavy, 65 compact-strong?) ===")
    for f in ["L", "y", "mean_burial", "total_bsa", "buried_frac", "nonbind_frac", "dangling_frac",
              "n_nonbinding", "rg_per_L", "e2e_per_L", "hyd_frac"]:
        mc = np.nanmean([r[f] for r in cr]); mt = np.nanmean([r[f] for r in t98])
        print(f"  {f:<14} cr65={mc:>8.2f}   the98={mt:>8.2f}   {'<-- 98 higher' if mt>mc else '<-- 65 higher'}")

    print("\n=== (D) CONSTRUCTIVE: reward contacts + penalize tails -> transferable? ===")

    def fp(tr, te, cols):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1); Xe = Xe[oke]
        ye = np.array([r["y"] for r in te])[oke]
        return pearsonr(np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w, ye)[0]
    for nm, cols in [("raw L", ["L"]),
                     ("total_bsa (contacts)", ["total_bsa"]),
                     ("n_nonbinding (tail penalty)", ["n_nonbinding"]),
                     ("contacts + tail", ["total_bsa", "n_nonbinding"]),
                     ("mean_burial + nonbind_frac", ["mean_burial", "nonbind_frac"]),
                     ("mmgbsa + n_nonbinding", ["mmgbsa", "n_nonbinding"]),
                     ("mmgbsa + total_bsa + n_nonbinding", ["mmgbsa", "total_bsa", "n_nonbinding"])]:
        print(f"  {nm:<34} 98→cr65={fp(t98,cr,cols):+.3f}  cr65→98={fp(cr,t98,cols):+.3f}")
    print("\n  >> a contacts+tail model that's POSITIVE both directions = Ram's decomposition works.")


if __name__ == "__main__":
    main()
