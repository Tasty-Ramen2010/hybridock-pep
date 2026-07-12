"""E62 — Ram's effective-length hypothesis: raw L flips sign because it counts floppy tails. Replace it
with the number of residues that ACTUALLY bind (high per-residue burial). A 20-mer with 5 buried anchors
should score like a 5-mer, not a 20-mer. Pool the-98 + crystal-65 into ONE unbiased set (pooling cancels
the per-dataset size↔affinity confound), and test whether a BURIAL-SPLIT length correlates consistently
where raw L does not.

Per-residue burial = SASA(peptide residue, peptide alone) − SASA(same residue, in complex), via
Bio.PDB ShrakeRupley. Features: n_anchor (#res buried > thr), buried_frac=n_anchor/L, mean/max burial,
total BSA. Tests: pooled corr vs ΔG, AND cross-dataset transfer (the real test) vs raw L.
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
CACHE = Path("/tmp/e62_burial.json")


def per_residue_burial(pep_pdb, rec_pdb):
    """Return list of (resname, buried_area) for peptide residues. Burial = free − complex SASA."""
    pep = P.get_structure("pep", str(pep_pdb))[0]
    rec = P.get_structure("rec", str(rec_pdb))[0]
    # peptide alone
    SR.compute(pep, level="R")
    free = {(r.get_parent().id, r.id[1]): (r.resname, r.sasa) for r in pep.get_residues() if r.id[0] == " "}
    # build complex = pep + rec in one structure
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    cx = Structure("cx")
    m = Model(0)
    cx.add(m)
    used = set()
    for ch in list(pep.get_chains()) + list(rec.get_chains()):
        cid = ch.id
        while cid in used:
            cid = chr(ord(cid) + 1) if cid.isalpha() else "Z"
        used.add(cid)
        ch2 = ch.copy()
        ch2.id = cid
        m.add(ch2)
    SR.compute(cx, level="R")
    # match peptide residues by (resname, resseq) order
    pep_chain_ids = {c.id for c in pep.get_chains()}
    comp_sasa = {}
    for ch in cx.get_chains():
        for r in ch.get_residues():
            if r.id[0] == " ":
                comp_sasa.setdefault((r.resname, r.id[1]), r.sasa)
    out = []
    for (chid, seq), (rn, fs) in free.items():
        cs = comp_sasa.get((rn, seq), fs)
        out.append((rn, max(0.0, fs - cs)))
    return out


def feats_from_burial(burials):
    L = len(burials)
    areas = np.array([b for _, b in burials])
    return dict(
        L=L,
        n_anchor40=int((areas > 40).sum()),
        n_anchor60=int((areas > 60).sum()),
        buried_frac40=float((areas > 40).mean()) if L else 0.0,
        mean_burial=float(areas.mean()) if L else 0.0,
        max_burial=float(areas.max()) if L else 0.0,
        total_bsa=float(areas.sum()),
        eff_len=float((areas / 80.0).clip(0, 1).sum()),  # soft anchor count (80Å²=fully buried res)
    )


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    rows = {}
    # crystal-65
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        try:
            b = per_residue_burial(r["peptide_pdb"], r["pocket_pdb"])
            f = feats_from_burial(b)
            f.update(ds="cr65", y=r["dg_exp"])
            rows[f"cr_{r['pdb']}"] = f
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} fail {str(e)[:40]}", flush=True)
    # the-98 (need y from e49b cache; structures in ppep_work)
    e49 = json.loads(Path("/tmp/e49b_the98.json").read_text())
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        pep = work / f"{k}_pep.pdb"
        rec = work / f"{k}_rec.pdb"
        if not (pep.exists() and rec.exists()):
            continue
        try:
            b = per_residue_burial(pep, rec)
            f = feats_from_burial(b)
            f.update(ds="the98", y=v["y"])
            rows[f"98_{k}"] = f
        except Exception as e:  # noqa: BLE001
            print(f"  98 {k} fail {str(e)[:40]}", flush=True)
    CACHE.write_text(json.dumps(rows))
    return rows


def corr_block(rows, feats):
    cr = [r for r in rows if r["ds"] == "cr65"]
    t98 = [r for r in rows if r["ds"] == "the98"]
    print(f"\n{'feature':<16}{'pooled':>10}{'cr65':>9}{'the98':>9}   sign-stable?")
    for f in feats:
        allr = [r for r in rows]
        x = np.array([r[f] for r in allr]); y = np.array([r["y"] for r in allr])
        xc = np.array([r[f] for r in cr]); yc = np.array([r["y"] for r in cr])
        xt = np.array([r[f] for r in t98]); yt = np.array([r["y"] for r in t98])
        rp = spearmanr(x, y).statistic
        rc = spearmanr(xc, yc).statistic
        rt = spearmanr(xt, yt).statistic
        stable = "YES" if (rc * rt > 0) else "NO  <-- FLIPS"
        print(f"  {f:<14}{rp:>+10.3f}{rc:>+9.3f}{rt:>+9.3f}   {stable}")


def transfer(rows, feat_sets):
    cr = [r for r in rows if r["ds"] == "cr65"]
    t98 = [r for r in rows if r["ds"] == "the98"]

    def fp(tr, te, cols):
        X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = 1.0 * np.eye(A.shape[1]); R[0, 0] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ y)
        Xe = np.array([[r[c] for c in cols] for r in te], float)
        return pearsonr(np.column_stack([np.ones(len(Xe)), (Xe - mu) / sd]) @ w,
                        np.array([r["y"] for r in te]))[0]
    print(f"\n=== cross-dataset transfer (does the feature SCALE, the real test) ===")
    print(f"{'feature set':<26}{'98→cr65':>10}{'cr65→98':>10}")
    for nm, cols in feat_sets.items():
        try:
            print(f"  {nm:<24}{fp(t98, cr, cols):>+10.3f}{fp(cr, t98, cols):>+10.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  {nm}: {str(e)[:40]}")


def main():
    rows_d = build()
    rows = list(rows_d.values())
    cr = sum(1 for r in rows if r["ds"] == "cr65"); t98 = sum(1 for r in rows if r["ds"] == "the98")
    print(f"=== E62 effective-length, POOLED.  cr65={cr}  the98={t98}  total={len(rows)} ===")
    print("Spearman(feature, exp ΔG) — note ΔG negative so NEGATIVE corr = feature↑ → stronger binding")
    corr_block(rows, ["L", "eff_len", "n_anchor40", "n_anchor60", "buried_frac40",
                      "mean_burial", "max_burial", "total_bsa"])
    transfer(rows, {
        "raw L (baseline)": ["L"],
        "eff_len (soft anchors)": ["eff_len"],
        "n_anchor40": ["n_anchor40"],
        "total_bsa": ["total_bsa"],
        "mean_burial (intensive)": ["mean_burial"],
        "eff_len+mean_burial": ["eff_len", "mean_burial"],
    })
    print("\n  >> if eff_len / n_anchor is sign-stable (cr65 & the98 same sign) where raw L FLIPS,")
    print("     Ram's split-length hypothesis is RIGHT: count binders, not residues.")


if __name__ == "__main__":
    main()
