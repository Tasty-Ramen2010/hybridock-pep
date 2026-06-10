#!/usr/bin/env python3
"""
bsa_tail_test.py — Per-residue / tail-localized SASA-BSA + clash ranker.

Tests Ram's tail insight RIGOROUSLY (real Shrake-Rupley, not contact counts):
  Total interface BSA is dominated by the core. A pose's loose-vs-bound TAIL is
  a small fraction of total BSA but drives global RMSD. Localizing the
  BSA(+clash) signal to the terminal residues may isolate it.

Per-residue peptide-side burial = SASA(pep atom, free) - SASA(pep atom, complex),
summed by residue. SASA saturates (buried atom can't add area) and we keep the
clash penalty — so this is the over-insertion-safe version, unlike contact counts.

Strategies (within-complex z, lower = better → ascending rank):
  global   -z(total_bsa) + z(clash)                 ≈ production BSA+clash
  tail     -z(tail_bsa)  + z(tail_clash)            terminal 4 residues only
  core     -z(core_bsa)  + z(core_clash)            non-terminal residues
  glob+tail -z(total_bsa)+z(clash) -0.5 z(tail_bsa) global + extra tail weight

Reports per-complex τ and top-1/5 Hit@2Å. Baseline ref2015 τ≈0.14, Hit@2Å 10.5%.

Run in score-env or rapidock (needs Biopython):
  python3 scripts/bsa_tail_test.py [--n N]   # N complexes (default all 57)
"""
from __future__ import annotations

import argparse, io, json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
ENC_PKL  = REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"
OUT = REPO / "logs" / "training_campaign" / "bsa_tail.json"

CLASH_DIST = 3.0
CROP = 10.0
N_TERM = 4   # terminal residues counted as "tail" (2 N-term + 2 C-term)


def read_heavy(pdb: str):
    """Return (lines, xyz[N,3], res_id_per_atom[N])."""
    lines, xyz, rid = [], [], []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        lines.append(ln)
        rid.append((ln[21], ln[22:27]))
    return lines, (np.array(xyz, np.float32) if xyz else np.empty((0, 3), np.float32)), rid


def per_atom_sasa(lines):
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    if not lines:
        return np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = PDBParser(QUIET=True).get_structure("x", io.StringIO("\n".join(lines) + "\nEND\n"))
        ShrakeRupley().compute(st, level="A")
        return np.array([a.sasa for a in st.get_atoms()], np.float32)


def pose_scores(pep_pdb, rec_lines, rec_xyz):
    pep_lines, pep_xyz, pep_rid = read_heavy(pep_pdb)
    if len(pep_xyz) < 4 or len(rec_xyz) < 4:
        return None
    # crop receptor near peptide
    d2 = ((rec_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1)  # [R,P]
    near = d2.min(1) <= CROP ** 2
    crop_lines = [rec_lines[i] for i in np.where(near)[0]]

    s_free = per_atom_sasa(pep_lines)                       # peptide alone
    s_cx_all = per_atom_sasa(pep_lines + crop_lines)        # complex
    s_bound = s_cx_all[:len(pep_lines)]                     # peptide portion
    if len(s_free) != len(pep_lines) or len(s_bound) != len(pep_lines):
        return None
    atom_bsa = np.maximum(s_free - s_bound, 0.0)            # per-atom burial

    # peptide atom→clash count (distance to nearest receptor atom)
    pd2 = ((pep_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1)  # [P,R]
    atom_clash = (pd2.min(1) < CLASH_DIST ** 2).astype(np.float32)

    # group atoms by residue, in order
    res_order, res_atoms = [], {}
    for i, k in enumerate(pep_rid):
        if k not in res_atoms:
            res_atoms[k] = []; res_order.append(k)
        res_atoms[k].append(i)
    P = len(res_order)
    term = set(res_order[:2] + res_order[-2:]) if P >= 4 else set(res_order)

    tot_bsa = float(atom_bsa.sum()); tot_clash = float(atom_clash.sum())
    tail_bsa = core_bsa = tail_clash = core_clash = 0.0
    for k in res_order:
        idx = res_atoms[k]
        b = float(atom_bsa[idx].sum()); c = float(atom_clash[idx].sum())
        if k in term:
            tail_bsa += b; tail_clash += c
        else:
            core_bsa += b; core_clash += c
    return dict(tot_bsa=tot_bsa, tot_clash=tot_clash,
                tail_bsa=tail_bsa, tail_clash=tail_clash,
                core_bsa=core_bsa, core_clash=core_clash)


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    bjson = json.load(open(GEN_JSON))
    cxs = sorted(set(k[0] for k in pickle.load(open(ENC_PKL, "rb"))))
    if a.n:
        cxs = cxs[:a.n]

    strat = ["global", "tail", "core", "glob_tail"]
    taus = {s: [] for s in strat}
    hit1 = {s: [] for s in strat}; hit5 = {s: [] for s in strat}
    oracle_hit1 = []
    t0 = time.time()

    for ci, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rr = entry.get("ref_rmsds", [])
        poses_dir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue
        rec_lines, rec_xyz, _ = read_heavy(str(rec_pdb))

        feats, rmsds = [], []
        for pi in range(len(rr)):
            pp = poses_dir / f"pose_{pi}.pdb"
            if not pp.exists():
                continue
            f = pose_scores(str(pp), rec_lines, rec_xyz)
            if f is None:
                continue
            feats.append(f); rmsds.append(rr[pi])
        if len(feats) < 10:
            continue
        rmsds = np.array(rmsds)
        g = {k: np.array([f[k] for f in feats]) for k in feats[0]}

        scores = {
            "global":    -_z(g["tot_bsa"])  + _z(g["tot_clash"]),
            "tail":      -_z(g["tail_bsa"]) + _z(g["tail_clash"]),
            "core":      -_z(g["core_bsa"]) + _z(g["core_clash"]),
            "glob_tail": -_z(g["tot_bsa"])  + _z(g["tot_clash"]) - 0.5*_z(g["tail_bsa"]),
        }
        for s, sc in scores.items():
            t, _ = sp.kendalltau(-sc, -rmsds)   # lower score = better; want τ(-sc,-rmsd)>0
            if not np.isnan(t):
                taus[s].append(t)
            order = np.argsort(sc)              # ascending: best first
            hit1[s].append(float(rmsds[order[0]] <= 2.0))
            hit5[s].append(float(rmsds[order[:5]].min() <= 2.0))
        oracle_hit1.append(float(rmsds.min() <= 2.0))

        if (ci + 1) % 5 == 0:
            el = time.time() - t0
            run = {s: np.mean(taus[s]) for s in strat}
            print(f"  {ci+1}/{len(cxs)} cx  {el:.0f}s  "
                  f"τ: global={run['global']:+.3f} tail={run['tail']:+.3f} "
                  f"glob_tail={run['glob_tail']:+.3f}", flush=True)

    n = len(oracle_hit1)
    print(f"\n{'='*64}")
    print(f"SASA-BSA TAIL TEST  ({n} complexes, {time.time()-t0:.0f}s)")
    print(f"{'='*64}")
    print(f"  baseline ref2015 τ≈+0.14, top-1 Hit@2Å 10.5%, top-5 21.1%")
    print(f"  oracle top-1 Hit@2Å = {100*np.mean(oracle_hit1):.1f}%\n")
    print(f"  {'strategy':<12} {'τ mean':>8} {'τ std':>8} {'top1 H2':>8} {'top5 H2':>8}")
    print(f"  {'-'*48}")
    for s in sorted(strat, key=lambda s: -np.mean(taus[s])):
        print(f"  {s:<12} {np.mean(taus[s]):>+8.4f} {np.std(taus[s]):>8.4f} "
              f"{100*np.mean(hit1[s]):>7.1f}% {100*np.mean(hit5[s]):>7.1f}%")

    OUT.write_text(json.dumps(
        {s: {"tau": float(np.mean(taus[s])), "hit1": float(np.mean(hit1[s])),
             "hit5": float(np.mean(hit5[s])), "n": len(taus[s])} for s in strat},
        indent=2))
    print(f"\n  Saved → {OUT}")


if __name__ == "__main__":
    main()
