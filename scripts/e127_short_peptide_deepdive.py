"""E127 — SHORT peptide (≤8) failure deep-dive: where we fail, the physics limit, how to fix it.

Short peptides score r=0.33 (worst with vlong). Hypothesis: short binding is ANCHOR / SPECIFIC-INTERACTION
dominated, not interface-SIZE dominated — but our 16 features are mostly extensive interface sums whose
dynamic range collapses for small interfaces. So the size features go near-constant and the few specific
interactions (a key salt bridge, an aromatic anchor, a deep hydrophobic plug) that actually drive short
binding aren't captured.

PART 1  FAILURE MAP — per-short-peptide error; worst cases; what they share.
PART 2  FEATURE DEATH — corr(feature,ΔG) and variance WITHIN short vs med: which features die on short.
PART 3  PHYSICS FIX — compute ANCHOR / specific-interaction features from the bound complex (max single-
        residue burial, deepest anchor, best salt bridge, aromatic anchor, terminal charge, rigidity) and
        test whether they rescue short (5-fold CV, short only).
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
PLROOT = ROOT / "data/drive_pull/pl/P-L"
POS, NEG, AROM, HYD = set("KR"), set("DE"), set("FWY"), set("AILMFWVC")


def cc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    return pearsonr(x[m], y[m])[0] if m.sum() > 4 and np.std(x[m]) > 0 else np.nan


def peptide_residues(mol2):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9 or f[1][0] == "H":
            continue
        try:
            atoms.append((f[1], "".join(c for c in f[7] if c.isalpha()).upper()[:3],
                          np.array([float(f[2]), float(f[3]), float(f[4])])))
        except ValueError:
            continue
    res, cur = [], None
    for nm, rn, xyz in atoms:
        if nm == "N":
            cur = {"rn": rn, "xyz": [], "at": {}}
            res.append(cur)
        if cur is None:
            cur = {"rn": rn, "xyz": [], "at": {}}
            res.append(cur)
        cur["xyz"].append(xyz)
        cur["at"][nm] = xyz
    return res


def rec_heavy_charged(pdb):
    heavy, charged = [], []
    rows = {}
    for ln in pdb.read_text().splitlines():
        if not ln.startswith("ATOM") or ln[12:16].strip()[:1] == "H":
            continue
        try:
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
        heavy.append(xyz)
        rn = ln[17:20].strip()
        if rn in ("LYS", "ARG", "ASP", "GLU"):
            rows.setdefault((ln[21], ln[22:27], rn), {})[ln[12:16].strip()] = xyz
    for (c, n, rn), at in rows.items():
        if rn in ("LYS", "ARG") and ("NZ" in at or "CZ" in at):
            charged.append((+1, at.get("NZ", at.get("CZ"))))
        elif rn in ("ASP", "GLU"):
            o = [at[k] for k in ("OD1", "OD2", "OE1", "OE2", "CG", "CD") if k in at]
            if o:
                charged.append((-1, np.mean(o, 0)))
    return (np.array(heavy) if heavy else np.zeros((1, 3))), charged


def anchor_features(pid, seq):
    d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{pid}/{pid}_ligand.mol2"))), None)
    if d is None:
        return None
    res = peptide_residues(d / f"{pid}_ligand.mol2")
    if not res:
        return None
    rec, rcharge = rec_heavy_charged(d / f"{pid}_protein.pdb")
    contacts, salt, arom_anchor = [], 0, 0
    for r in res:
        rx = np.array(r["xyz"])
        nc = int((np.linalg.norm(rec[:, None, :] - rx[None, :, :], axis=2).min(0) < 4.5).sum()) if rx.size else 0
        nb = int((np.linalg.norm(rec - rx.mean(0), axis=1) < 8.0).sum()) if rx.size else 0
        contacts.append(nb)
        # best salt bridge from this residue's charge center
        cc_ = None
        if r["rn"] in ("LYS",) and "NZ" in r["at"]:
            cc_ = (+1, r["at"]["NZ"])
        elif r["rn"] == "ARG" and "CZ" in r["at"]:
            cc_ = (+1, r["at"]["CZ"])
        elif r["rn"] == "ASP":
            o = [r["at"][k] for k in ("OD1", "OD2", "CG") if k in r["at"]]
            cc_ = (-1, np.mean(o, 0)) if o else None
        elif r["rn"] == "GLU":
            o = [r["at"][k] for k in ("OE1", "OE2", "CD") if k in r["at"]]
            cc_ = (-1, np.mean(o, 0)) if o else None
        if cc_ and rcharge:
            for sr, xr in rcharge:
                if cc_[0] * sr < 0 and np.linalg.norm(cc_[1] - xr) < 4.5:
                    salt += 1
                    break
        if r["rn"] in ("PHE", "TRP", "TYR") and nb > 12:
            arom_anchor += 1
    contacts = np.array(contacts)
    return {"max_burial": float(contacts.max()), "deep_anchors": int((contacts > 15).sum()),
            "best_salt_bridge": float(salt), "arom_anchor": float(arom_anchor),
            "burial_concentration": float(contacts.max() / (contacts.sum() + 1e-9))}


def seq_short_feats(seq):
    L = max(1, len(seq))
    nterm = 1.0  # N-term +
    cterm = -1.0
    return {"term_netq": nterm + cterm + sum(c in POS for c in seq) - sum(c in NEG for c in seq),
            "rigid_frac": (seq.count("P") + seq.count("G")) / L,
            "arom_frac": sum(c in AROM for c in seq) / L,
            "hyd_frac": sum(c in HYD for c in seq) / L}


ANCHORF = ["max_burial", "deep_anchors", "best_salt_bridge", "arom_anchor", "burial_concentration"]
SEQF = ["term_netq", "rigid_frac", "arom_frac", "hyd_frac"]


def main():
    # load pooled (need short subset; PDBbind has structures for anchor features)
    pdbb = [json.loads(ln) for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    short = [r for r in pdbb if r["length"] <= 8]
    med = [r for r in pdbb if 9 <= r["length"] <= 12]
    print(f"=== E127 SHORT peptide deep-dive (PDBbind short={len(short)}, med={len(med)}) ===\n")

    # PART 2: feature death — corr + variance within short vs med
    print("PART 2 — feature behavior WITHIN short vs med  [corr(feat,ΔG) | coeff-of-variation]:")
    ys, ym = np.array([r["y"] for r in short]), np.array([r["y"] for r in med])
    print(f"{'feature':<14}{'short corr':>12}{'med corr':>10}{'short CV':>10}{'med CV':>9}  note")
    for c in PROD:
        vs = np.array([r[c] for r in short]); vm = np.array([r[c] for r in med])
        cvs = np.std(vs) / (abs(np.mean(vs)) + 1e-9)
        cvm = np.std(vm) / (abs(np.mean(vm)) + 1e-9)
        note = "DEAD on short" if cvs < 0.25 * cvm or abs(cc(vs, ys)) < 0.05 else ""
        print(f"{c:<14}{cc(vs,ys):>+12.2f}{cc(vm,ym):>+10.2f}{cvs:>10.2f}{cvm:>9.2f}  {note}")

    # PART 3: anchor / specific-interaction features
    print("\nPART 3 — compute ANCHOR features for short, test if they rescue short:")
    rows = []
    for r in short:
        af = anchor_features(r["pdb"], r["seq"])
        if af is None:
            continue
        rows.append({"y": r["y"], "feat": [r[c] for c in PROD],
                     "anchor": dict(af, **seq_short_feats(r["seq"]))})
    y = np.array([r["y"] for r in rows])
    print(f"  short with anchor features: n={len(rows)}")
    print("  anchor/specific feature → corr(feat, ΔG) within short:")
    for a in ANCHORF + SEQF:
        v = np.array([r["anchor"][a] for r in rows])
        print(f"     {a:<22} {cc(v,y):+.3f}")

    def cv(add):
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(rows))
        X = np.array([r["feat"] + ([r["anchor"][a] for a in ANCHORF + SEQF] if add else []) for r in rows], float)
        pred = np.full(len(rows), np.nan)
        for f in range(5):
            tr = fold != f
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=2.0, min_samples_leaf=12, random_state=0).fit(X[tr], y[tr])
            pred[fold == f] = m.predict(X[fold == f])
        ok = ~np.isnan(pred)
        return pearsonr(pred[ok], y[ok])[0], float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))
    rb, rmb = cv(False)
    ra, rma = cv(True)
    print(f"\n  SHORT GBT 5-fold:  16-feat r={rb:+.3f} (RMSE {rmb:.2f}) → +anchor/specific r={ra:+.3f} (RMSE {rma:.2f})  Δ={ra-rb:+.3f}")
    print("\n  reading: if anchor/specific features lift short, the physics limit = our SIZE features collapse")
    print("  on small interfaces; short binding needs SPECIFIC-INTERACTION (anchor/salt/aromatic) features.")


if __name__ == "__main__":
    main()
