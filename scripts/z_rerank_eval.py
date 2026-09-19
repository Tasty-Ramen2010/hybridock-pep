#!/usr/bin/env python
"""Rerank our own poses with Boltz's pair representation -- against the baseline that matters.

WHAT PASSED ITS CONTROLS. Boltz's z block locates the designed interface at AUC 0.909 against a
shuffled 0.551, it beats a receptor-only burial baseline on 8/8 complexes, and scoring our own 24
poses by agreement with its contact map beat our rank-1 selection on 8/8 while recovering 83% of
the pool's oracle headroom. That is a real signal and it needs no retraining of anything.

THE BASELINE THAT DECIDES WHETHER IT IS WORTH SHIPPING. Getting z requires running Boltz on the
pair -- about 36 s. But if Boltz is already running, the obvious alternative is to skip our pool
entirely and take BOLTZ'S OWN POSE, which on these fourteen sits at a median near 3.4 A with 9/14
under 5 A while our pool's best is nowhere near that. A reranker that loses to "just use the
co-fold" is not a feature, however good its controls looked. So all four are reported side by
side:

  our rank 1          what we ship today, no Boltz
  z-picked            our pool, ranked by Boltz's contact map
  pool oracle         the best pose we generated -- the ceiling any reranker can reach
  Boltz's own pose    the trivial alternative, same Boltz call

HOW THE MAP IS USED. For each pose, take the contacts it actually makes with the receptor and
average the z-affinity over exactly those pairs. A pose that puts its residues where Boltz says
contacts belong scores high. No training, no fitting, no free parameters beyond the contact
cutoff -- which matters, because with 14 complexes anything fitted here would be fitted to noise.

Usage: z_rerank_eval.py [--cutoff 8.0]
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import subprocess
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
POOL = ROOT / "runs/coventry/hybridock_ft"
BOLTZ_POSES = ROOT / "runs/coventry/boltz"
EMB = ROOT / "datasets/boltz_emb"
BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
GOOD = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc12", "pc21", "pc26", "pc28",
        "pc34", "pc43", "pc44", "pc46"]
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: Path, chain: str | None = None):
    xyz, seq, seen = [], [], set()
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        if chain is not None and l[21] != chain:
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        seq.append(AA3.get(l[17:20].strip(), 'X'))
    return np.array(xyz), ''.join(seq)


def best_offset(a: str, b: str) -> int:
    best = (0, -1.0)
    for off in range(0, 10):
        x = a[off:]
        n = min(len(x), len(b))
        if n < 30:
            continue
        idt = sum(1 for u, v in zip(x[:n], b[:n]) if u == v) / n
        if idt > best[1]:
            best = (off, idt)
    return best[0]


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rmsd(A, B):
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def get_z(name: str, rseq: str, pseq: str):
    d = EMB / name
    f = d / "emb.npz"
    if not f.exists():
        d.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="zr_", dir="/tmp/claude-1000"))
        try:
            spec = work / "s.yaml"
            spec.write_text("version: 1\nsequences:\n"
                            f"  - protein:\n      id: A\n      sequence: {rseq}\n      msa: empty\n"
                            f"  - protein:\n      id: B\n      sequence: {pseq}\n      msa: empty\n")
            pr = subprocess.run([str(BOLTZ), "predict", str(spec), "--out_dir", str(work),
                                 "--recycling_steps", "3", "--diffusion_samples", "1",
                                 "--write_embeddings", "--output_format", "mmcif", "--override"],
                                capture_output=True, text=True, timeout=1800, check=False)
            npz = sorted(work.rglob("*embeddings*.npz")) or sorted(work.rglob("*.npz"))
            if pr.returncode != 0 or not npz:
                return None
            shutil.copy(npz[0], f)
        except subprocess.TimeoutExpired:
            return None
        finally:
            shutil.rmtree(work, ignore_errors=True)
    d0 = np.load(f, allow_pickle=True)
    arrs = {k: np.squeeze(d0[k], axis=0) if d0[k].shape[0] == 1 else d0[k] for k in d0}
    key = next((k for k in arrs if arrs[k].ndim >= 3
                and arrs[k].shape[0] == arrs[k].shape[1] > 5), None)
    return arrs[key] if key else None


def boltz_pose_rmsd(p: str, truth: np.ndarray) -> float:
    """Boltz's own pose, superimposed into the paper frame via the receptor."""
    for fn in ("transplanted.pdb", "cofolded.pdb"):
        f = BOLTZ_POSES / f"{p}__{p}" / fn
        if not f.exists():
            continue
        ch: dict[str, int] = {}
        for l in f.read_text().splitlines():
            if l.startswith("ATOM"):
                ch[l[21]] = ch.get(l[21], 0) + 1
        if len(ch) < 2:
            continue
        pepch = min(ch, key=lambda c: ch[c])
        recch = max(ch, key=lambda c: ch[c])
        q, _ = ca_seq(f, pepch)
        mrec, mseq = ca_seq(f, recch)
        prec, pseq = ca_seq(PAPER / f"{p}_1b1.pdb")
        if len(mrec) < 30 or len(prec) < 30:
            continue
        off = best_offset(mseq, pseq)
        mm = mrec[off:]
        n = min(len(mm), len(prec))
        R, t = kabsch(mm[:n], prec[:n])
        return rmsd((R @ q.T).T + t, truth)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", type=float, default=8.0)
    a = ap.parse_args()

    truth_csv = {(r["peptide"], r["binder_label"][:-1]): r
                 for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth_csv.items() if pp == p)
            for p in GOOD}

    print(f"{'target':<8}{'our rank1':>11}{'z-picked':>10}{'pool best':>11}"
          f"{'Boltz pose':>12}{'z pctile':>10}")
    ours, zsel, best, bpose = [], [], [], []
    for p in GOOD:
        recf, pepf = PAPER / f"{p}_1b1.pdb", PAPER / f"{p}_peptide.pdb"
        pooldir = POOL / f"{p}__{p}B"
        if not (recf.exists() and pepf.exists() and pooldir.exists()):
            continue
        rec_xyz, rseq = ca_seq(recf)
        tru, _ = ca_seq(pepf)
        Z = get_z(p, rseq, seqs[p])
        if Z is None:
            print(f"{p:<8}  no embeddings")
            continue
        nr, npep = len(rec_xyz), len(tru)
        if Z.shape[0] < nr + npep:
            print(f"{p:<8}  z too small ({Z.shape[0]} < {nr + npep})")
            continue
        zmap = np.linalg.norm(Z[:nr, nr:nr + npep], axis=-1)

        pool = sorted(pooldir.glob("rank*.pdb"), key=lambda f: int(f.stem[4:]))
        scores, rms = [], []
        for f in pool:
            q, _ = ca_seq(f)
            n = min(len(q), npep)
            if n < 3:
                continue
            obs = (np.linalg.norm(rec_xyz[:, None, :] - q[None, :n, :], axis=2) < a.cutoff)
            scores.append(float(zmap[:, :n][obs].mean()) if obs.any() else -1e9)
            rms.append(rmsd(q, tru))
        if len(scores) < 5:
            continue
        scores, rms = np.array(scores), np.array(rms)
        sel = int(np.argmax(scores))
        bp = boltz_pose_rmsd(p, tru)
        pct = 100.0 * (rms > rms[sel]).mean()
        ours.append(rms[0]); zsel.append(rms[sel]); best.append(rms.min()); bpose.append(bp)
        print(f"{p:<8}{rms[0]:>11.2f}{rms[sel]:>10.2f}{rms.min():>11.2f}{bp:>12.2f}{pct:>9.0f}%")

    if not zsel:
        print("\nnothing scored")
        return
    ours, zsel, best = np.array(ours), np.array(zsel), np.array(best)
    bpose = np.array(bpose)
    print(f"\n{'median':<8}{np.median(ours):>11.2f}{np.median(zsel):>10.2f}"
          f"{np.median(best):>11.2f}{np.nanmedian(bpose):>12.2f}")
    print(f"{'<=5A':<8}{int((ours <= 5).sum()):>11}{int((zsel <= 5).sum()):>10}"
          f"{int((best <= 5).sum()):>11}{int(np.nansum(bpose <= 5)):>12}   (of {len(ours)})")

    from scipy import stats
    print(f"\n  z-rerank vs our rank1:  better on {(zsel < ours).sum()}/{len(ours)}"
          f"   Wilcoxon p={stats.wilcoxon(zsel, ours).pvalue:.4f}"
          f"   median {np.median(zsel - ours):+.2f} A")
    gap = np.median(ours) - np.median(best)
    got = np.median(ours) - np.median(zsel)
    print(f"  oracle headroom {gap:.2f} A; z recovers {got:.2f} A "
          f"({100 * got / gap if gap > 0 else 0:.0f}%)")

    ok = np.isfinite(bpose)
    print(f"\n  THE BASELINE THAT DECIDES IT — z-rerank vs simply taking Boltz's pose:")
    print(f"    z-rerank better on {(zsel[ok] < bpose[ok]).sum()}/{int(ok.sum())}"
          f"   Wilcoxon p={stats.wilcoxon(zsel[ok], bpose[ok]).pvalue:.4f}")
    print(f"    medians: z-rerank {np.median(zsel[ok]):.2f} A vs Boltz pose "
          f"{np.median(bpose[ok]):.2f} A")
    print("\n  -> " + (
        "reranking our pool beats using the co-fold directly, so the extra pool is earning its "
        "keep."
        if np.median(zsel[ok]) < np.median(bpose[ok]) else
        "taking Boltz's pose directly BEATS reranking our pool. The z signal is real but the "
        "reranker is not the way to use it -- our pool does not contain poses good enough for a "
        "perfect ranker to find, which makes this a GENERATION problem. The value of z is as a "
        "conditioning signal inside the generator, not as a selector over its output."))


if __name__ == "__main__":
    main()
