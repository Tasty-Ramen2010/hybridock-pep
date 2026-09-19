#!/usr/bin/env python
"""The first honest measurement of our pose error — against the paper's own designed complexes.

WHAT ARRIVED. Brian sent the AF2-multimer structures behind the disordered-peptide paper, and all
eighteen of our targets are in them as two-chain complexes: the designed binder AND the peptide in
its intended binding mode, sequences matching our grid exactly. Until now every pose number we had
was relative to something we built ourselves -- our ESMFold binder models and either our docking
or Boltz's co-folding -- so "5.87 A on designed grooves" was an estimate against a moving target.
This is the target standing still.

THREE THINGS IT SETTLES, each of which we have only been able to guess at:

  1  WHICH BINDER MODELS WERE WRONG, and by how much. We already found n3/n7/pc21/pc26 to be
     outliers because Boltz and AF3 agreed against our ESMFold; that was circumstantial. The
     paper's own structure is the arbiter.

  2  HOW FAR OUR DOCKED PEPTIDE ACTUALLY IS. Superimpose on the binder, then measure the peptide.
     This is peptide RMSD in the receptor frame -- the quantity that matters for scoring, and the
     one we have never had a ground truth for on this set.

  3  WHETHER WE THREAD IT BACKWARDS. The N-to-C axis cosine against the designed pose answers the
     79-81% backwards claim directly instead of inferring it from geometric proxies.

Both of our arms are measured the same way, so the co-folded arm's advantage -- or lack of one --
is finally on an absolute scale rather than relative to itself.

Usage: coventry_truepose_audit.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
TRUE = ROOT / "datasets/coventry/paper_af2"
ESM = ROOT / "datasets/coventry/binders"
AF3 = ROOT / "datasets/coventry/binders_af3"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
REBUILT = {"n3", "n7", "pc21", "pc26"}


def ca(pdb: Path, chain: str | None = None) -> np.ndarray:
    out, seen = [], set()
    for l in pdb.read_text().splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        if chain and l[21] != chain:
            continue
        key = (l[21], l[22:27])
        if key in seen:
            continue
        seen.add(key)
        out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(out)


def split_complex(p: Path):
    """(binder CA, peptide CA) — the peptide is the shorter chain, as in every one of these."""
    chains = {}
    for l in p.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            chains.setdefault(l[21], []).append(
                (float(l[30:38]), float(l[38:46]), float(l[46:54])))
    ks = sorted(chains, key=lambda c: -len(chains[c]))
    return np.array(chains[ks[0]]), np.array(chains[ks[1]])


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation+translation putting P onto Q (equal length)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def rmsd(A: np.ndarray, B: np.ndarray) -> float:
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean()))


def axis(X: np.ndarray) -> np.ndarray:
    v = X[-1] - X[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(3)


def main() -> None:
    print("PART 1 — how wrong were our binder models? (CA RMSD to the paper's AF2 binder)\n")
    print(f"  {'binder':<8}{'our model':<10}{'ESMFold':>10}{'AF3':>10}   verdict")
    esm_r, af3_r = {}, {}
    for b in ORDER:
        tc = TRUE / f"{b}_complex.pdb"
        if not tc.exists():
            continue
        tb, _ = split_complex(tc)
        row = []
        for tag, d in (("ESMFold", ESM), ("AF3", AF3)):
            f = d / f"{b}_1b1.pdb"
            if not f.exists():
                row.append(None); continue
            m = ca(f)
            n = min(len(m), len(tb))
            if n < 10:
                row.append(None); continue
            R, t = kabsch(m[:n], tb[:n])
            row.append(rmsd((R @ m[:n].T).T + t, tb[:n]))
        esm_r[b], af3_r[b] = row[0], row[1]
        used = "AF3" if b in REBUILT else "ESMFold"
        e = f"{row[0]:.2f}" if row[0] is not None else "  -  "
        a = f"{row[1]:.2f}" if row[1] is not None else "  -  "
        flag = ""
        if row[0] is not None and row[0] > 4.0:
            flag = "ESMFold WRONG"
            if row[1] is not None and row[1] < row[0] - 1:
                flag += " — AF3 was the right call"
        print(f"  {b:<8}{used:<10}{e:>10}{a:>10}   {flag}")
    ok = [v for k, v in esm_r.items() if v is not None and k not in REBUILT]
    bad = [v for k, v in esm_r.items() if v is not None and k in REBUILT]
    if ok:
        print(f"\n  ESMFold on the 14 we kept:   median {st.median(ok):.2f} A")
    if bad:
        print(f"  ESMFold on the 4 we replaced: median {st.median(bad):.2f} A")
    a4 = [af3_r[k] for k in REBUILT if af3_r.get(k) is not None]
    if a4:
        print(f"  AF3 on those same 4:          median {st.median(a4):.2f} A")

    print("\n\nPART 2 — how far is our docked peptide from the designed pose?")
    print("Superimposed on the binder first, so this is peptide error in the receptor frame.\n")
    print(f"  {'target':<8}{'our dock':>10}{'co-folded':>11}{'axis ours':>11}"
          f"{'axis cofold':>13}   threading")
    ours_r, cof_r, ours_ax, cof_ax = [], [], [], []
    for p in ORDER:
        tc = TRUE / f"{p}_complex.pdb"
        if not tc.exists():
            continue
        tb, tp = split_complex(tc)
        rec_model = (AF3 if p in REBUILT else ESM) / f"{p}_1b1.pdb"
        if not rec_model.exists():
            continue
        mb = ca(rec_model)
        n = min(len(mb), len(tb))
        R, t = kabsch(mb[:n], tb[:n])          # our receptor frame -> paper frame

        def score(pep_pdb: Path):
            if not pep_pdb.exists():
                return None, None
            q = ca(pep_pdb)
            if len(q) < 3:
                return None, None
            q = (R @ q.T).T + t
            return rmsd(q, tp), float(np.dot(axis(q), axis(tp)))

        d_dock = sorted((ROOT / f"runs/coventry/hybridock_ft/{p}__{p}B").glob("rank*.pdb"))
        r1, a1 = score(d_dock[0]) if d_dock else (None, None)
        r2, a2 = score(ROOT / f"runs/coventry/boltz/{p}__{p}/peptide.pdb")
        for v, acc in ((r1, ours_r), (r2, cof_r), (a1, ours_ax), (a2, cof_ax)):
            if v is not None:
                acc.append(v)
        f = lambda x: f"{x:.2f}" if x is not None else "  -  "
        thread = ""
        if a1 is not None:
            thread = "ours BACKWARDS" if a1 < 0 else "ours forward"
        print(f"  {p:<8}{f(r1):>10}{f(r2):>11}{f(a1):>11}{f(a2):>13}   {thread}")

    if ours_r:
        print(f"\n  our docked peptide:  median {st.median(ours_r):5.2f} A   "
              f"<=5A {sum(r <= 5 for r in ours_r)}/{len(ours_r)}   "
              f"<=2A {sum(r <= 2 for r in ours_r)}/{len(ours_r)}")
    if cof_r:
        print(f"  co-folded peptide:   median {st.median(cof_r):5.2f} A   "
              f"<=5A {sum(r <= 5 for r in cof_r)}/{len(cof_r)}   "
              f"<=2A {sum(r <= 2 for r in cof_r)}/{len(cof_r)}")
    if ours_ax:
        back = sum(1 for a in ours_ax if a < 0)
        print(f"\n  THREADING DIRECTION, measured against the designed pose for the first time:")
        print(f"    our docked pose runs backwards on {back}/{len(ours_ax)} "
              f"({100 * back / len(ours_ax):.0f}%)   mean cosine {st.mean(ours_ax):+.3f}")
    if cof_ax:
        back = sum(1 for a in cof_ax if a < 0)
        print(f"    co-folded pose runs backwards on {back}/{len(cof_ax)} "
              f"({100 * back / len(cof_ax):.0f}%)   mean cosine {st.mean(cof_ax):+.3f}")


if __name__ == "__main__":
    main()
