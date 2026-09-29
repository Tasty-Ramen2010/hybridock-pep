#!/usr/bin/env python
"""Phase B scoring of the Boltz-2 arm: superimpose into the crystal frame, emit rank*.pdb.

THE CO-FOLD IS IN BOLTZ'S OWN FRAME. A co-folding model predicts the whole complex from
sequence, so its coordinates have no relationship to the crystal's. An earlier script in this
repo compared raw co-fold coordinates against a crystal peptide and reported 0/14 under 5 A
against an established 9/14; the contradiction is the only reason the omission was caught. So
every pose here is superimposed by its RECEPTOR before the peptide is touched.

The alignment is exact rather than fuzzy, and that is worth stating. The YAML for each complex
was built from ca_seq() of the crystal receptor itself, so Boltz chain A residue i IS the i-th
CA of the crystal receptor. There is no MSG-tag offset problem here. That is asserted, not
assumed: a length mismatch means the spec and the crystal drifted apart and the complex is
dropped rather than silently scored against a shifted frame.

OUTPUT SHAPE IS DELIBERATE. Superimposed peptides are written as runs/<set>/boltz2/<name>/
rank{1..N}.pdb, Boltz's own confidence order, which is exactly the layout our other arms use.
That means dockq_rs.py, the RMSD pass and the figure code all work on this arm unchanged, with
one metric and one crystal across every arm.

LEAKAGE: Boltz-2's cutoff is 2023-06-01 and this bench is 2020-2023. This arm is a memorised
upper bound, not a fair head-to-head, and must be labelled as such wherever it is plotted.

Usage: boltz2_bench_score.py [--pred datasets/boltz2_bench/pred] [--out runs/balanced_length/boltz2]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")


def chains(p: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for l in p.read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")):
            out.setdefault(l[21], []).append(l)
    return out


def ca(lines: list[str]) -> np.ndarray:
    """CA coordinates, deduplicated by (chain, residue), in file order.

    THE CHAIN MUST BE PART OF THE KEY. An earlier version keyed on the residue number alone.
    boltz2_bench_prep.py builds each YAML's receptor sequence with a chain-AWARE dedup, so a
    multi-chain pocket whose chains reuse residue numbers came out shorter here than the sequence
    Boltz was actually given. The length assertion below then read that as a frame mismatch and
    dropped the complex: 194 of 387 were silently skipped, and the 193 that survived were the
    single-chain (smaller, easier) ones, which biased the arm upward.
    """
    seen, xyz = set(), []
    for l in lines:
        if l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(xyz)


def kabsch(P: np.ndarray, Q: np.ndarray):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def transform(lines: list[str], R: np.ndarray, t: np.ndarray) -> list[str]:
    out = []
    for l in lines:
        v = R @ np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) + t
        out.append(f"{l[:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{l[54:]}")
    return out


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default=str(ROOT / "data/bench_balanced_length.csv"))
    ap.add_argument("--pred", default=str(ROOT / "datasets/boltz2_bench/pred"))
    ap.add_argument("--out", default=str(ROOT / "runs/balanced_length/boltz2"))
    a = ap.parse_args()

    pred, out = Path(a.pred), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.bench)))

    done = skipped = 0
    best1, bestN = [], []
    for r in rows:
        name = r["name"]
        # boltz writes predictions/<stem>/<stem>_model_N.pdb; accept either nesting depth
        cands = sorted(pred.rglob(f"{name}_model_*.pdb"))
        if not cands:
            skipped += 1
            continue
        crec = ca([l for l in Path(r["receptor"]).read_text().splitlines()
                   if l.startswith("ATOM")])
        cpep = ca([l for l in Path(r["peptide_pdb"]).read_text().splitlines()
                   if l.startswith("ATOM")])
        d = out / name
        d.mkdir(exist_ok=True)
        rs = []
        for i, f in enumerate(sorted(cands, key=lambda p: int(p.stem.rsplit("_", 1)[1])), 1):
            ch = chains(f)
            if len(ch) < 2:
                continue
            rc = max(ch, key=lambda c: len(ch[c]))
            pc = min(ch, key=lambda c: len(ch[c]))
            brec, bpep = ca(ch[rc]), ca(ch[pc])
            # exact correspondence or nothing -- see docstring
            if len(brec) != len(crec):
                break
            R, t = kabsch(brec, crec)
            (d / f"rank{i}.pdb").write_text(
                "\n".join(transform(ch[pc], R, t)) + "\nEND\n")
            rs.append(rmsd((R @ bpep.T).T + t, cpep))
        if not rs:
            skipped += 1
            continue
        done += 1
        best1.append(rs[0])
        bestN.append(min(rs))

    b1, bn = np.array(best1), np.array(bestN)
    print(f"boltz2 arm: scored {done}, skipped {skipped}")
    if done:
        print(f"  top-1  median {np.median(b1):.2f} A   <=5A {int((b1 <= 5).sum())}/{done}"
              f"   <=2A {int((b1 <= 2).sum())}/{done}")
        print(f"  oracle median {np.median(bn):.2f} A   <=5A {int((bn <= 5).sum())}/{done}"
              f"   <=2A {int((bn <= 2).sum())}/{done}")
        print("  NOTE: 2020-2023 bench is inside Boltz-2's 2023-06-01 training cutoff. "
              "Memorised upper bound, not a fair head-to-head.")


if __name__ == "__main__":
    main()
