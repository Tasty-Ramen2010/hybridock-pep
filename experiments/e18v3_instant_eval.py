"""E18 v3 — honest test of the INSTANT part of v2 (no MD, no GPU).

Isolates the one genuinely-new feature from e18v2: bond_strength_sasa, i.e.
per-residue ΔSASA weighted by contact FAVORABILITY (satisfied H-bond / salt bridge /
apolar packing = more favorable; buried-unsatisfied polar/charge = PENALTY) + clash.
This is instant geometry (same family as the hb+aromatic features that DO replicate
across datasets), so it is the cheapest, most-likely-to-help piece of Ram's spec.

Runs the SAME honest harness as e18_train_eval:
  A) cross-dataset transfer crystal<->PEPBI (Pearson r, sign)
  B) leave-binding-group-out on PEPBI (pooled r, median per-group Spearman, % correct)
Compares: baseline[hb,arom]  vs  +de_strength  vs  de_strength+clash alone.
Falsifiable question: does favorability-weighted SASA add signal the bare
hb+aromatic baseline doesn't already have? Verdict printed at the end.

Checkpoints de_strength to /tmp/e18v3_{cr,pb}.json so it is crash-safe.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from e18_hybrid_features import AA3to1, EISENBERG  # noqa: E402  (no MD import)

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
POS = {"ARG", "LYS", "HIS"}
NEG = {"ASP", "GLU"}
CHARGED = POS | NEG
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "TRP", "HIS"}
APOLAR = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "GLY", "TRP"}


def _per_res_sasa(struct):
    SR.compute(struct, level="A")
    return {(r.get_parent().id, r.id): sum(float(a.sasa) for a in r)
            for r in struct.get_residues() if r.id[0] == " "}


def bond_strength_sasa(pep_pdb, complex_path, pep_chain):
    """Per-residue ΔSASA weighted by contact favorability + clash penalty (instant)."""
    free = _per_res_sasa(P.get_structure("f", pep_pdb))
    cpx = _per_res_sasa(P.get_structure("c", complex_path))
    cx = P.get_structure("cc", complex_path)[0]
    pep_res = [r for r in cx[pep_chain] if r.id[0] == " "]
    rec_atoms = [a for ch in cx if ch.id != pep_chain for r in ch if r.id[0] == " "
                 for a in r if a.element != "H"]
    if not pep_res or not rec_atoms:
        return None
    ns = NeighborSearch(rec_atoms)
    pf = [r for r in P.get_structure("pp", pep_pdb)[0].get_residues() if r.id[0] == " "]
    n = min(len(pep_res), len(pf))
    de = 0.0
    clash = 0
    for i in range(n):
        rc = pep_res[i]
        rn = rc.resname.upper()
        aa = AA3to1.get(rn, "A")
        rfree = free.get((pf[i].get_parent().id, pf[i].id), 0.0)
        rbound = cpx.get((rc.get_parent().id, rc.id), 0.0)
        dsasa = max(0.0, rfree - rbound)
        if dsasa < 1.0:
            continue
        has_hb = has_sb = has_apolar = False
        unsat_polar = (rn in POLAR or rn in CHARGED)
        for a in rc:
            if a.element == "H":
                continue
            for b in ns.search(a.coord, 4.5):
                d = float(np.linalg.norm(a.coord - b.coord))
                if d < 2.5:
                    clash += 1
                brn = b.get_parent().resname.upper()
                if a.element in ("N", "O") and b.element in ("N", "O") and d <= 3.5:
                    has_hb = True
                    unsat_polar = False
                if rn in CHARGED and a.element in ("N", "O") and d <= 4.0:
                    want = NEG if rn in POS else POS
                    if brn in want:
                        has_sb = True
                        unsat_polar = False
                if rn in APOLAR and brn in APOLAR and d <= 5.0:
                    has_apolar = True
        fav = -EISENBERG.get(aa, 0.0)
        if has_hb:
            fav -= 0.3
        if has_sb:
            fav -= 0.6
        if has_apolar:
            fav -= 0.2
        if unsat_polar:
            fav += 1.0
        de += fav * dsasa
    de /= 100.0
    return dict(de_strength=de, clash_pen=float(clash))


# ---------------- loaders (reuse v1 datasets, add de_strength) ----------------

def crystal_records():
    out_path = Path("/tmp/e18v3_cr.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["pdb"] for r in out}
    e0 = json.loads(Path("/tmp/e0_rows.json").read_text())
    v1 = {r["pdb"]: r for r in json.loads(Path("/tmp/e18_cr.json").read_text())}
    for r in e0:
        pdb = r["pdb"].upper()
        if not r.get("pep_pdb") or pdb in done or pdb not in v1:
            continue
        merged = Path(f"/tmp/e18v3_cx/{pdb}.pdb"); merged.parent.mkdir(exist_ok=True)
        lines = []
        for src, ch in ((r["pep_pdb"], "P"), (r["poc_pdb"], "R")):
            for ln in Path(src).read_text().splitlines():
                if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                    lines.append(ln[:21] + ch + ln[22:])
        merged.write_text("\n".join(lines) + "\nEND\n")
        try:
            s = bond_strength_sasa(r["pep_pdb"], str(merged), "P")
        except Exception as e:  # noqa: BLE001
            print(f"  cr {pdb} FAIL {type(e).__name__}", flush=True); continue
        if not s:
            continue
        b = v1[pdb]
        out.append(dict(pdb=pdb, y=b["y"], seq=b["seq"], grp=b["grp"],
                        hb_count=b.get("hb_count") or 0, aromatic_cc=b.get("aromatic_cc") or 0,
                        de_sasa=b["de_sasa"], **s))
        out_path.write_text(json.dumps(out))
    return out


class _B(Select):
    def accept_chain(self, ch): return ch.id == "B"
    def accept_residue(self, res): return res.id[0] == " "


def pepbi_records():
    out_path = Path("/tmp/e18v3_pb.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["nm"] for r in out}
    files = {os.path.basename(f)[:-4].lower(): f
             for f in glob.glob("/tmp/pepbi/struct/**/*.pdb", recursive=True)}
    v1 = json.loads(Path("/tmp/e18_pb.json").read_text())
    pep_d = Path("/tmp/e18v3_pep"); pep_d.mkdir(exist_ok=True)
    for b in v1:
        nm = b["nm"]
        if nm in done or nm not in files:
            continue
        cx = files[nm]
        try:
            s = P.get_structure("x", cx)[0]
            if "B" not in [c.id for c in s]:
                continue
            io = PDBIO(); io.set_structure(P.get_structure("y", cx))
            pep_pdb = pep_d / f"{nm}.pdb"; io.save(str(pep_pdb), _B())
            sc = bond_strength_sasa(str(pep_pdb), cx, "B")
        except Exception as e:  # noqa: BLE001
            print(f"  pb {nm} FAIL {type(e).__name__}", flush=True); continue
        if not sc:
            continue
        out.append(dict(nm=nm, y=b["y"], seq=b["seq"], grp=b["grp"],
                        hb_count=b.get("hb_count") or 0, aromatic_cc=b.get("aromatic_cc") or 0,
                        de_sasa=b["de_sasa"], **sc))
        if len(out) % 25 == 0:
            print(f"  pb {len(out)}", flush=True)
        out_path.write_text(json.dumps(out))
    return out


# ---------------- honest harness (same as e18_train_eval) ----------------

FEAT = ["de_strength", "clash_pen", "de_sasa", "hb_count", "aromatic_cc"]
MODELS = {
    "baseline hb+arom": ["hb_count", "aromatic_cc"],
    "de_strength alone": ["de_strength"],
    "de_strength+clash": ["de_strength", "clash_pen"],
    "baseline + de_strength": ["hb_count", "aromatic_cc", "de_strength"],
    "ALL (str+clash+sasa+base)": ["de_strength", "clash_pen", "de_sasa", "hb_count", "aromatic_cc"],
}


def _mat(recs):
    X = np.array([[r[f] for f in FEAT] for r in recs], float)
    meta = [dict(y=r["y"], grp=r["grp"]) for r in recs]
    return X, meta


def _cols(names):
    return [FEAT.index(n) for n in names]


def _fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w


def transfer(Xtr, mtr, Xte, mte, names):
    c = _cols(names)
    ytr = np.array([m["y"] for m in mtr]); yte = np.array([m["y"] for m in mte])
    pred = _fit_predict(Xtr[:, c], ytr, Xte[:, c])
    return pearsonr(pred, yte).statistic, float(np.sqrt(np.mean((pred - yte) ** 2)))


def logo(X, meta, names):
    c = _cols(names)
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["grp"], []).append(i)
    multi = {g: idx for g, idx in groups.items() if len(idx) >= 4}
    rhos, pp, py = [], [], []
    for gid, te in multi.items():
        tr = [i for i in range(len(meta)) if meta[i]["grp"] != gid]
        ytr = np.array([meta[i]["y"] for i in tr])
        pred = _fit_predict(X[np.ix_(tr, c)], ytr, X[np.ix_(te, c)])
        yte = np.array([meta[i]["y"] for i in te])
        if np.std(pred) > 0:
            rhos.append(spearmanr(pred, yte).statistic)
            pp.append(pred - pred.mean()); py.append(yte - yte.mean())
    rhos = np.array(rhos)
    pr = pearsonr(np.concatenate(pp), np.concatenate(py)).statistic if pp else float("nan")
    return rhos, pr, len(multi)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "cr"):
        print("=== crystal de_strength ===", flush=True); cr = crystal_records()
    else:
        cr = json.loads(Path("/tmp/e18v3_cr.json").read_text())
    if which in ("both", "pb"):
        print("=== PEPBI de_strength ===", flush=True); pb = pepbi_records()
    else:
        pb = json.loads(Path("/tmp/e18v3_pb.json").read_text())
    print(f"\ncrystal={len(cr)} pepbi={len(pb)}")

    Xc, mc = _mat(cr); Xp, mp = _mat(pb)
    print("\n=== A) CROSS-DATASET TRANSFER (Pearson r; want > baseline & positive) ===")
    print(f"{'model':<28}{'cr->pb':>9}{'pb->cr':>9}{'cr->pb RMSE':>13}")
    for name, names in MODELS.items():
        r_cp, rmse = transfer(Xc, mc, Xp, mp, names)
        r_pc, _ = transfer(Xp, mp, Xc, mc, names)
        print(f"{name:<28}{r_cp:>9.3f}{r_pc:>9.3f}{rmse:>13.2f}")

    print("\n=== B) LEAVE-GROUP-OUT on PEPBI (within-target) ===")
    print(f"{'model':<28}{'pooled r':>10}{'median rho':>11}{'%correct':>10}{'n_grp':>7}")
    base_pr = None
    for name, names in MODELS.items():
        rhos, pr, ng = logo(Xp, mp, names)
        if name == "baseline hb+arom":
            base_pr = pr
        if len(rhos):
            print(f"{name:<28}{pr:>10.3f}{np.median(rhos):>11.2f}{np.mean(rhos > 0):>9.0%}{ng:>7}")

    print("\n=== VERDICT ===")
    b_cp, _ = transfer(Xc, mc, Xp, mp, ["hb_count", "aromatic_cc"])
    add_cp, _ = transfer(Xc, mc, Xp, mp, ["hb_count", "aromatic_cc", "de_strength"])
    add_rhos, add_pr, _ = logo(Xp, mp, ["hb_count", "aromatic_cc", "de_strength"])
    print(f"  cross-dataset: baseline {b_cp:+.3f} -> +de_strength {add_cp:+.3f}  "
          f"({'HELPS' if add_cp > b_cp + 0.03 else 'no gain'})")
    print(f"  within-target LOGO pooled r: baseline {base_pr:+.3f} -> +de_strength {add_pr:+.3f}  "
          f"({'HELPS' if add_pr > base_pr + 0.03 else 'no gain'})")


if __name__ == "__main__":
    main()
