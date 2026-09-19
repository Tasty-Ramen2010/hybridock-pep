#!/usr/bin/env python
"""Option 3, tested before it is built: do Boltz's internal embeddings carry usable information?

THE PROPOSAL was to feed Boltz's learned representations into RAPiDock as conditioning -- importing
what ten million structures taught it without needing ten million structures ourselves. Wiring that
into the architecture and retraining is weeks. Whether the embeddings carry anything we can use is
answerable in an afternoon, so it is answered first.

WHAT BOLTZ EXPOSES. `--write_embeddings` dumps two tensors:
  s   per-residue ("single") representation, one vector per residue
  z   per-PAIR representation, one vector for every residue pair i,j -- this is the thing that
      makes co-folding work, because "residue i of the peptide belongs against residue j of the
      binder" is something it can state directly, and our graph network cannot.

THE PROBE, and why it is the right one. Rather than asking "can a model be built on these", ask
the prior question: do the embeddings already know where the peptide goes? For each of the
designed complexes we take the z block between peptide and binder residues, reduce it to a
predicted contact score per residue pair, and check it against the contacts in the paper's OWN
designed structure. If Boltz's pair representation recovers the true contact map, the information
is present and conditioning on it is worth building. If it does not, option 3 dies here for the
price of a GPU afternoon.

Then the harder, more useful question: can that contact map PICK a good pose out of our own pool?
Our 24 poses already contain better and worse placements -- on the Coventry grid the oracle over
our pool is far better than what our scoring picks. If Boltz's contacts rank our own poses well,
conditioning would help even without changing a single weight of RAPiDock.

The direction-prior experiment is the cautionary precedent: importing ONE bit from co-folding (the
N-to-C axis) was real but useless, 0.634 -> 0.591. A contact map is far richer than one bit, which
is why it is worth a second look -- but the same standard applies, and the controls are the same.

Usage: boltz_embedding_probe.py [--n 8]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path

import numpy as np

# Derive the repo root instead of hardcoding it: this script also runs on the DGX, where
# the tree lives at ~/Ram_Work/hybridock-pep. A hardcoded /home/igem path has broken a
# DGX run three times before.
ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "datasets/coventry/binders_paper"
POOL = ROOT / "runs/coventry/hybridock_ft"
OUT = ROOT / "datasets/boltz_emb"
BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc12", "pc21", "pc26", "pc28",
         "pc34", "pc43", "pc44", "pc46"]          # the 14 with a verified identity
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


def run_boltz(name: str, rec: str, pep: str) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    if (d / "emb.npz").exists():
        return {"npz": str(d / "emb.npz"), "cached": True}
    work = Path(tempfile.mkdtemp(prefix="emb_", dir="/tmp/claude-1000"))
    try:
        spec = work / "s.yaml"
        spec.write_text("version: 1\nsequences:\n"
                        f"  - protein:\n      id: A\n      sequence: {rec}\n      msa: empty\n"
                        f"  - protein:\n      id: B\n      sequence: {pep}\n      msa: empty\n")
        t0 = time.time()
        pr = subprocess.run([str(BOLTZ), "predict", str(spec), "--out_dir", str(work),
                             "--recycling_steps", "3", "--diffusion_samples", "1",
                             "--write_embeddings", "--output_format", "mmcif", "--override"],
                            capture_output=True, text=True, timeout=1800, check=False)
        if pr.returncode != 0:
            return {"error": pr.stderr[-300:]}
        npz = sorted(work.rglob("*embeddings*.npz")) or sorted(work.rglob("*.npz"))
        if not npz:
            return {"error": "no embeddings written"}
        shutil.copy(npz[0], d / "emb.npz")
        return {"npz": str(d / "emb.npz"), "seconds": round(time.time() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import csv
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}

    targets = ORDER[:a.n]
    print(f"extracting Boltz embeddings for {len(targets)} designed complexes\n", flush=True)
    got = []
    for p in targets:
        recf = PAPER / f"{p}_1b1.pdb"
        if not recf.exists():
            continue
        _, rseq = ca_seq(recf)
        r = run_boltz(p, rseq, seqs[p])
        if "npz" in r:
            got.append((p, r["npz"]))
            tag = "(cached)" if r.get("cached") else f"{r.get('seconds', 0):.0f}s"
            print(f"  {p:<7} ok  {tag}", flush=True)
        else:
            print(f"  {p:<7} FAILED: {r.get('error', '')[:90]}", flush=True)
    if not got:
        print("\nno embeddings extracted — option 3 cannot be tested this way")
        return

    print(f"\n{len(got)} embedding sets. Inspecting what Boltz actually writes:\n")
    p0, f0 = got[0]
    z = np.load(f0, allow_pickle=True)
    for k in list(z)[:8]:
        arr = z[k]
        print(f"  {k:<24}{str(arr.shape):<22}{arr.dtype}")

    print("\nDOES THE PAIR REPRESENTATION KNOW THE INTERFACE?\n")
    print(f"  {'target':<8}{'pep':>5}{'rec':>6}{'AUC vs contacts':>17}{'shuffled':>12}")
    aucs = []
    for p, f in got:
        try:
            d = np.load(f, allow_pickle=True)
            # Boltz writes both tensors with a leading batch axis of 1: s is (1, L, 384) and z
            # is (1, L, L, 128). Squeezing it is not cosmetic -- without it every candidate key
            # has min(shape[:2]) == 1, the selector below rejects both, and the probe prints its
            # header and nothing else, which is exactly what the first run did.
            arrs = {k: np.squeeze(d[k], axis=0) if d[k].shape[0] == 1 else d[k] for k in d}
            key = next((k for k in arrs if arrs[k].ndim >= 3
                        and arrs[k].shape[0] == arrs[k].shape[1] > 5), None)
            if key is None:
                print(f"  {p:<8}  no square pair tensor in {list(arrs)}")
                continue
            Z = arrs[key]
            rec_xyz, _ = ca_seq(PAPER / f"{p}_1b1.pdb")
            pep_xyz, _ = ca_seq(PAPER / f"{p}_peptide.pdb")
            nr, npep = len(rec_xyz), len(pep_xyz)
            if Z.shape[0] < nr + npep:
                continue
            # cross block: receptor rows x peptide columns, reduced to a scalar per pair
            block = Z[:nr, nr:nr + npep]
            score = np.linalg.norm(block, axis=-1) if block.ndim == 3 else block
            true = (np.linalg.norm(rec_xyz[:, None, :] - pep_xyz[None, :, :], axis=2) < 8.0)
            def auc_of(sc, tr):
                pos, neg = sc.ravel()[tr.ravel()], sc.ravel()[~tr.ravel()]
                if len(pos) < 5 or len(neg) < 5:
                    return None
                a = float(np.mean([(x > neg).mean() + 0.5 * (x == neg).mean() for x in pos]))
                return max(a, 1 - a)         # sign of the reduction is arbitrary

            auc = auc_of(score, true)
            if auc is None:
                continue
            # CONTROL. Taking max(a, 1-a) means the null is NOT 0.5 -- it is whatever that
            # folding gives on a contact map this sparse, which is why a shuffled map is scored
            # the identical way rather than assumed. The direction-prior experiment is the
            # precedent: a real effect that was still worth nothing.
            rng = np.random.default_rng(0)
            null = float(np.mean([auc_of(score, true[rng.permutation(nr)][:, rng.permutation(npep)])
                                  or 0.5 for _ in range(20)]))
            aucs.append((auc, null))
            print(f"  {p:<8}{npep:>5}{nr:>6}{auc:>17.3f}{null:>12.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {p:<8}  probe failed: {type(exc).__name__}")
    if aucs:
        import statistics as st
        real = st.mean(a for a, _ in aucs)
        null = st.mean(n for _, n in aucs)
        print(f"\n  mean AUC {real:.3f} over {len(aucs)} complexes, shuffled control {null:.3f}")
        print(f"  lift over the control: {real - null:+.3f}")
        print("  -> " + ("the pair representation LOCATES the interface. Conditioning on it is "
                         "worth building."
                         if real - null > 0.10 and real > 0.65 else
                         "the raw pair block does not locate the interface under this reduction "
                         "-- either the wrong tensor, or the information is not accessible as a "
                         "vector norm. Note this rules out the CHEAP version of option 3, not "
                         "the learned one: a trained projection of z could still carry it."))


if __name__ == "__main__":
    main()
