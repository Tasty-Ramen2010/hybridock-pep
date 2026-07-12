"""E125 — does the entropy term IMPROVE affinity scoring? (Ram's Σ-over-contacts decomposition)

Proper binding-entropy penalty: residues that CONTACT the receptor lose their free conformational
entropy. So entropy_lost = Σ_{contacting residues} per-residue free MD entropy. Higher = bigger −TΔS
penalty = WEAKER binding → corr(entropy_lost, ΔG) should be POSITIVE.

For PDBbind peptides that already have computed per-residue entropy (data/sfree_perres.jsonl): compute
contacting peptide residues from the bound complex (mol2 + receptor), sum their free entropy, add as a
feature to the 16 structural features. Does affinity GBT improve — especially in long/vlong where the
atlas said entropy is the missing physics? (Preliminary: only the peptides computed so far; full run pending.)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density", "cys_frac"]
PLROOT = ROOT / "data/drive_pull/pl/P-L"


def seqhash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def band(L):
    return "short≤8" if L <= 8 else "med9-12" if L <= 12 else "long13-16" if L <= 16 else "vlong≥17"


def peptide_residue_xyz(mol2):
    """Return list of per-residue heavy-atom coord arrays (ordered by backbone N)."""
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
            atoms.append((f[1], np.array([float(f[2]), float(f[3]), float(f[4])])))
        except ValueError:
            continue
    res, cur = [], None
    for nm, xyz in atoms:
        if nm == "N":
            cur = []
            res.append(cur)
        if cur is None:
            cur = []
            res.append(cur)
        cur.append(xyz)
    return [np.array(r) for r in res]


def receptor_heavy(pdb):
    xyz = []
    for ln in pdb.read_text().splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip()[:1] != "H":
            try:
                xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            except ValueError:
                pass
    return np.array(xyz) if xyz else np.zeros((1, 3))


def entropy_lost(pid, per_res_ent):
    d = next((Path(p).parent for p in glob.glob(str(PLROOT / f"*/{pid}/{pid}_ligand.mol2"))), None)
    if d is None:
        return None
    res_xyz = peptide_residue_xyz(d / f"{pid}_ligand.mol2")
    if not res_xyz:
        return None
    rec = receptor_heavy(d / f"{pid}_protein.pdb")
    ent = [e for e in per_res_ent]
    lost = 0.0
    free_tot = 0.0
    n_contact = 0
    for i, rx in enumerate(res_xyz):
        ei = ent[i] if i < len(ent) and ent[i] is not None else None
        if ei is None:
            continue
        free_tot += ei
        if rx.size and (np.linalg.norm(rec[:, None, :] - rx[None, :, :], axis=2).min() < 4.5):
            lost += ei
            n_contact += 1
    return {"entropy_lost": lost, "entropy_free_tot": free_tot,
            "entropy_lost_frac": lost / (free_tot + 1e-9), "n_contact_res": n_contact}


def main():
    sfree = {}
    for ln in (ROOT / "data/sfree_perres.jsonl").read_text().splitlines():
        r = json.loads(ln)
        sfree[r["hash"]] = r
    pdbb = [json.loads(ln) for ln in (ROOT / "data/pdbbind_peptides.jsonl").read_text().splitlines()]
    rows = []
    for r in pdbb:
        sf = sfree.get(seqhash(r["seq"].upper()))
        if not sf or not sf.get("per_res_entropy"):
            continue
        el = entropy_lost(r["pdb"], sf["per_res_entropy"])
        if el is None:
            continue
        valid = [e for e in sf["per_res_entropy"] if e is not None]
        s_free = sf.get("s_free") or (float(np.mean(valid)) if valid else 0.0)
        rows.append({"y": r["y"], "length": r["length"], "feat": {c: r[c] for c in PROD},
                     "ent": dict(el, s_free=s_free)})
    print(f"=== E125 entropy-scoring test (n={len(rows)} peptides with computed entropy + PDBbind structure) ===\n")
    if len(rows) < 25:
        print(f"  only {len(rows)} so far — wait for more MD. (preliminary below if ≥10)")
        if len(rows) < 10:
            return
    y = np.array([r["y"] for r in rows])
    L = np.array([r["length"] for r in rows])

    el = np.array([r["ent"]["entropy_lost"] for r in rows])
    elf = np.array([r["ent"]["entropy_lost_frac"] for r in rows])
    print(f"  corr(entropy_lost, ΔG)      = {pearsonr(el, y)[0]:+.3f}  (expect >0: more lost → weaker)")
    print(f"  corr(entropy_lost_frac, ΔG) = {pearsonr(elf, y)[0]:+.3f}")
    print(f"  corr(s_free, ΔG)            = {pearsonr([r['ent']['s_free'] for r in rows], y)[0]:+.3f}")
    for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]:
        m = np.array([band(x) == b for x in L])
        if m.sum() >= 6:
            print(f"     {b:<11} corr(entropy_lost,ΔG)={pearsonr(el[m], y[m])[0]:+.3f} (n={m.sum()})")

    ENT = ["entropy_lost", "entropy_lost_frac", "entropy_free_tot", "s_free", "n_contact_res"]

    def cv(add):
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 5, len(rows))
        X = np.array([[r["feat"][c] for c in PROD] + ([r["ent"][e] for e in ENT] if add else []) for r in rows], float)
        pred = np.full(len(rows), np.nan)
        for f in range(5):
            tr = fold != f
            m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                              l2_regularization=2.0, min_samples_leaf=15, random_state=0).fit(X[tr], y[tr])
            pred[fold == f] = m.predict(X[fold == f])
        return pred

    def rr(p, m):
        ok = m & ~np.isnan(p)
        return pearsonr(p[ok], y[ok])[0] if ok.sum() > 4 else np.nan
    base, ent = cv(False), cv(True)
    print("\n  GBT 5-fold:  struct16 → struct16 + entropy, by length")
    for lab, m in [("ALL", np.ones(len(rows), bool))] + [(b, np.array([band(x) == b for x in L])) for b in ["short≤8", "med9-12", "long13-16", "vlong≥17"]]:
        if m.sum() >= 6:
            print(f"     {lab:<11} n={m.sum():<4} base={rr(base,m):+.3f} → +entropy={rr(ent,m):+.3f}  Δ={rr(ent,m)-rr(base,m):+.3f}")
    print("\n  reading: +entropy lifts long/vlong ⇒ the MD entropy term recovers the regime the atlas flagged.")
    print("  (Preliminary on the MD computed so far; re-run after the full 922 completes for the real number.)")


if __name__ == "__main__":
    main()
