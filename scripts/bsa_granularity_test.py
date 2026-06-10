#!/usr/bin/env python3
"""
bsa_granularity_test.py — Does tail/per-residue contact beat total BSA?

Ram's hypothesis: total interface BSA is dominated by the (identical) core in
two poses that differ only in tail docking — so it dilutes the loose-vs-bound
tail signal that actually drives global RMSD. Per-residue / terminal contact
should isolate that signal.

Cheap contact-distance proxy for BSA (no Shrake-Rupley): per-complex Kendall τ
of each contact aggregation vs ref_rmsds, over the 57 gen_n100 complexes.
Higher contact → tighter pose → lower RMSD, so we report τ(feature, -rmsd);
positive = correctly signed. Baseline to beat: ref2015 / total-BSA τ ≈ 0.14.

Features:
  total_contacts   total peptide-receptor atom contacts < 6 Å  (∝ total BSA)
  n_contacting     # peptide residues with any receptor contact < 5 Å
  contact_frac     fraction of peptide residues making contact
  tail_contacts    contacts made by the 3 N-term + 3 C-term residues
  term_buried      mean(min_dist<5Å) over the 4 terminal residues
  mean_min_dist    mean over residues of (min distance to receptor)   [neg signal]

Run (any env with numpy/scipy):
  python3 scripts/bsa_granularity_test.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
from scipy import stats as sp
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"


def read_heavy_by_res(pdb: str):
    """Return list of per-residue heavy-atom coord arrays, in chain order."""
    res, order = {}, []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        key = (ln[21], ln[22:27])
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        except ValueError:
            continue
        if key not in res:
            res[key] = []
            order.append(key)
        res[key].append(xyz)
    return [np.array(res[k], dtype=np.float32) for k in order]


def features(pep_res, rec_xyz):
    """Compute contact aggregations for one pose. rec_xyz: [Nrec,3] all heavy."""
    P = len(pep_res)
    if P < 2 or len(rec_xyz) < 4:
        return None
    per_res_mindist = np.empty(P, np.float32)
    per_res_ncontact = np.empty(P, np.float32)
    total_contacts = 0.0
    for i, pr in enumerate(pep_res):
        d = cdist(pr, rec_xyz)              # [n_atoms_i, Nrec]
        per_res_mindist[i] = d.min()
        per_res_ncontact[i] = (d < 6.0).sum()
        total_contacts += (d < 6.0).sum()

    contacting = per_res_mindist < 5.0     # bool per residue
    term_idx = [0, 1, P - 2, P - 1] if P >= 4 else list(range(P))
    tail_contacts = float(sum(per_res_ncontact[i] for i in term_idx))
    return {
        "total_contacts": float(total_contacts),
        "n_contacting":   float(contacting.sum()),
        "contact_frac":   float(contacting.mean()),
        "tail_contacts":  tail_contacts,
        "term_buried":    float(np.mean([contacting[i] for i in term_idx])),
        "mean_min_dist":  float(-per_res_mindist.mean()),  # neg: closer = better
    }


def main():
    import pickle
    bjson = json.load(open(GEN_JSON))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))
    feat_names = ["total_contacts", "n_contacting", "contact_frac",
                  "tail_contacts", "term_buried", "mean_min_dist"]
    per_cx_tau = {f: [] for f in feat_names}
    n_done = 0

    for cn in cxs:
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        poses_dir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue
        rec_res = read_heavy_by_res(str(rec_pdb))
        rec_xyz = np.vstack(rec_res) if rec_res else np.empty((0, 3))

        rows, rmsds = [], []
        for pi in range(len(rr)):
            pp = poses_dir / f"pose_{pi}.pdb"
            if not pp.exists():
                continue
            f = features(read_heavy_by_res(str(pp)), rec_xyz)
            if f is None:
                continue
            rows.append(f)
            rmsds.append(rr[pi])
        if len(rows) < 10:
            continue
        rmsds = np.array(rmsds)
        for fn in feat_names:
            vals = np.array([r[fn] for r in rows])
            t, _ = sp.kendalltau(vals, -rmsds)   # higher feature → lower RMSD
            if not np.isnan(t):
                per_cx_tau[fn].append(t)
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} complexes...", flush=True)

    print(f"\n{'='*60}")
    print(f"CONTACT GRANULARITY TEST  ({n_done} complexes)")
    print(f"{'='*60}")
    print(f"\n  per-complex Kendall τ vs RMSD (higher feature = better pose)")
    print(f"  baseline ref2015 / total-BSA ≈ +0.14\n")
    print(f"  {'feature':<18} {'τ mean':>8} {'τ std':>8} {'τ>0 %':>7}")
    print(f"  {'-'*44}")
    ranked = sorted(feat_names, key=lambda f: -np.mean(per_cx_tau[f]))
    for fn in ranked:
        t = np.array(per_cx_tau[fn])
        print(f"  {fn:<18} {t.mean():>+8.4f} {t.std():>8.4f} {100*np.mean(t>0):>6.0f}%")

    best = ranked[0]
    print(f"\n  Best: {best}  (τ={np.mean(per_cx_tau[best]):+.4f})")
    if np.mean(per_cx_tau[best]) > np.mean(per_cx_tau["total_contacts"]) + 0.02:
        print(f"  → tail/per-residue granularity BEATS total BSA by "
              f"{np.mean(per_cx_tau[best])-np.mean(per_cx_tau['total_contacts']):+.4f}. "
              f"Worth real Shrake-Rupley test.")
    else:
        print(f"  → total_contacts is ~best; finer granularity does NOT help. "
              f"Over-insertion cap dominates.")

    out = {f: {"tau_mean": float(np.mean(per_cx_tau[f])),
               "tau_std": float(np.std(per_cx_tau[f])),
               "n": len(per_cx_tau[f])} for f in feat_names}
    (REPO / "logs" / "training_campaign" / "bsa_granularity.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
