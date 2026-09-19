#!/usr/bin/env python
"""The probe said Boltz's pair block locates the interface at AUC 0.909. Two controls decide
whether that is worth anything.

WHY CONTROLS AND NOT A CELEBRATION. AUC 0.909 against a shuffled control of 0.551 is a large,
clean effect -- but "the pair block knows where the peptide binds" is only interesting if BOTH of
these hold, and neither is implied by the AUC:

CONTROL A -- IS IT REDUNDANT? A peptide binds in a groove, and grooves are geometrically
distinctive from the receptor side alone. If a neighbour-count burial score computed from the
receptor structure with no network at all recovers the same contact map, then z is re-deriving
something we can compute in a millisecond, and conditioning RAPiDock on it imports nothing. This
is the control that actually threatens the result, so it is run first. The comparison is
like-for-like: same reduction to a per-residue score, same AUC, same shuffled null.

CONTROL B -- IS IT ACTIONABLE? The probe scores z against the TRUE contact map, which we only
have because the paper gave us the answer. The question that matters for the tool is whether the
map can pick a good pose out of a pool we already generate. We have 24 poses per pair and we know
the oracle over that pool is far better than what our scoring selects, so there is headroom to
capture. Scoring each pose by its agreement with Boltz's contact map, and comparing the pose it
picks against the pool's best and against random selection, measures the part we could actually
use -- and needs no retraining at all.

A pass on A and B together means conditioning is worth building. A pass on A alone means the
information is real but we have not shown we can act on it. A fail on A means option 3 is
measuring groove geometry.

Usage: boltz_z_controls.py
"""
from __future__ import annotations

import csv
import os
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
EMB = ROOT / "datasets/boltz_emb"
POOL = ROOT / "runs/coventry/hybridock_ft"
CONTACT = 8.0
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def auc_of(score: np.ndarray, true: np.ndarray) -> float | None:
    s, t = score.ravel(), true.ravel()
    pos, neg = s[t], s[~t]
    if len(pos) < 5 or len(neg) < 5:
        return None
    a = float(np.mean([(x > neg).mean() + 0.5 * (x == neg).mean() for x in pos]))
    return max(a, 1 - a)


def shuffled_null(score, true, nr, npep, reps=20) -> float:
    rng = np.random.default_rng(0)
    return float(np.mean([auc_of(score, true[rng.permutation(nr)][:, rng.permutation(npep)])
                          or 0.5 for _ in range(reps)]))


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def main() -> None:
    targets = sorted(d.name for d in EMB.iterdir() if (d / "emb.npz").exists())
    print(f"{len(targets)} complexes with cached embeddings\n")

    print("CONTROL A -- is the pair block telling us anything a burial count does not?\n")
    print(f"  {'target':<8}{'boltz z':>10}{'burial':>10}{'shuffled':>10}{'z - burial':>12}")
    zs, bs, ns = [], [], []
    keep = []
    for p in targets:
        recf, pepf = PAPER / f"{p}_1b1.pdb", PAPER / f"{p}_peptide.pdb"
        if not (recf.exists() and pepf.exists()):
            continue
        rec_xyz, _ = ca_seq(recf)
        pep_xyz, _ = ca_seq(pepf)
        nr, npep = len(rec_xyz), len(pep_xyz)
        d = np.load(EMB / p / "emb.npz", allow_pickle=True)
        arrs = {k: np.squeeze(d[k], axis=0) if d[k].shape[0] == 1 else d[k] for k in d}
        key = next((k for k in arrs if arrs[k].ndim >= 3
                    and arrs[k].shape[0] == arrs[k].shape[1] > 5), None)
        if key is None or arrs[key].shape[0] < nr + npep:
            continue
        Z = arrs[key]
        zscore = np.linalg.norm(Z[:nr, nr:nr + npep], axis=-1)
        true = (np.linalg.norm(rec_xyz[:, None, :] - pep_xyz[None, :, :], axis=2) < CONTACT)

        # burial: neighbours within 10 A among receptor CAs, broadcast across peptide columns.
        # Deliberately receptor-only and structure-only -- no network, no peptide information.
        nb = (np.linalg.norm(rec_xyz[:, None, :] - rec_xyz[None, :, :], axis=2) < 10.0).sum(1)
        bscore = np.repeat(nb[:, None].astype(float), npep, axis=1)

        az, ab = auc_of(zscore, true), auc_of(bscore, true)
        an = shuffled_null(zscore, true, nr, npep)
        if az is None or ab is None:
            continue
        zs.append(az); bs.append(ab); ns.append(an); keep.append(p)
        print(f"  {p:<8}{az:>10.3f}{ab:>10.3f}{an:>10.3f}{az - ab:>+12.3f}")

    if not zs:
        print("  nothing scored"); return
    print(f"\n  {'mean':<8}{st.mean(zs):>10.3f}{st.mean(bs):>10.3f}{st.mean(ns):>10.3f}"
          f"{st.mean(zs) - st.mean(bs):>+12.3f}")
    wins = sum(1 for a, b in zip(zs, bs) if a > b)
    print(f"  z beats plain burial on {wins}/{len(zs)} complexes")
    a_pass = st.mean(zs) - st.mean(bs) > 0.10
    print("\n  -> " + ("z carries interface information that receptor geometry alone does NOT. "
                       "The signal is not just groove shape."
                       if a_pass else
                       "a neighbour count reproduces most of it. The pair block is largely "
                       "re-deriving groove geometry we can compute for free, so conditioning on "
                       "it imports far less than the raw AUC suggested."))

    print("\n\nCONTROL B -- can that map PICK a pose out of our own pool?\n")
    print(f"  {'target':<8}{'z-picked':>11}{'our rank1':>11}{'pool best':>11}"
          f"{'pool mean':>11}{'z pctile':>10}")
    picked, ours, best, rnd = [], [], [], []
    for p in keep:
        pool = sorted((POOL / f"{p}__{p}B").glob("rank*.pdb"),
                      key=lambda f: int(f.stem[4:])) if (POOL / f"{p}__{p}B").exists() else []
        if len(pool) < 5:
            continue
        rec_xyz, _ = ca_seq(PAPER / f"{p}_1b1.pdb")
        truth, _ = ca_seq(PAPER / f"{p}_peptide.pdb")
        nr, npep = len(rec_xyz), len(truth)
        d = np.load(EMB / p / "emb.npz", allow_pickle=True)
        arrs = {k: np.squeeze(d[k], axis=0) if d[k].shape[0] == 1 else d[k] for k in d}
        key = next(k for k in arrs if arrs[k].ndim >= 3
                   and arrs[k].shape[0] == arrs[k].shape[1] > 5)
        zmap = np.linalg.norm(arrs[key][:nr, nr:nr + npep], axis=-1)

        scores, rms = [], []
        for f in pool:
            q, _ = ca_seq(f)
            n = min(len(q), npep)
            if n < 3:
                continue
            obs = (np.linalg.norm(rec_xyz[:, None, :] - q[None, :n, :], axis=2) < CONTACT)
            # agreement = mean z-affinity over the contacts this pose actually makes
            scores.append(float(zmap[:, :n][obs].mean()) if obs.any() else -1e9)
            rms.append(rmsd(q, truth))
        if len(scores) < 5:
            continue
        scores, rms = np.array(scores), np.array(rms)
        sel = int(np.argmax(scores))
        pct = 100.0 * (rms > rms[sel]).mean()
        picked.append(rms[sel]); ours.append(rms[0]); best.append(rms.min()); rnd.append(rms.mean())
        print(f"  {p:<8}{rms[sel]:>11.2f}{rms[0]:>11.2f}{rms.min():>11.2f}"
              f"{rms.mean():>11.2f}{pct:>9.0f}%")

    if not picked:
        print("  no pools matched"); return
    picked, ours, best, rnd = map(np.array, (picked, ours, best, rnd))
    print(f"\n  {'median':<8}{np.median(picked):>11.2f}{np.median(ours):>11.2f}"
          f"{np.median(best):>11.2f}{np.median(rnd):>11.2f}")
    print(f"\n  z-selection beats our own rank1 on {(picked < ours).sum()}/{len(picked)}")
    print(f"  z-selection beats random-from-pool on {(picked < rnd).sum()}/{len(picked)}")
    gap = np.median(rnd) - np.median(best)
    got = np.median(rnd) - np.median(picked)
    print(f"  oracle headroom in the pool: {gap:.2f} A;  z recovers {got:.2f} A "
          f"({100 * got / gap if gap > 0 else 0:.0f}%)")
    b_pass = (picked < rnd).mean() > 0.6 and got > 0.5
    print("\n  -> " + ("the map can act on our own pool without retraining anything. "
                       "Conditioning is worth building."
                       if b_pass else
                       "knowing the contact map does NOT let us pick a better pose from what we "
                       "already generate. The information is real but our pool may not contain a "
                       "pose good enough for it to find -- which is a GENERATION problem, not a "
                       "conditioning one."))

    print("\n\nVERDICT ON OPTION 3")
    print(f"  A (not redundant with geometry): {'PASS' if a_pass else 'FAIL'}")
    print(f"  B (actionable on our own poses): {'PASS' if b_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
