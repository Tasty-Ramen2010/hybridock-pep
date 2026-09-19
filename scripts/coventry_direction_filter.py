#!/usr/bin/env python
"""Stage 1.4 on the real grid: keep Boltz's ONE BIT, keep our own poses and our own scoring.

THE POINT. Panel D of the final figure uses co-folded COORDINATES, and that is a fair thing to
show but it is Boltz's structure doing the work. This asks a narrower question: if we import from
the co-folding model only the DIRECTION the peptide runs through the groove -- one bit, a unit
vector, no atoms -- and use it to filter OUR OWN docked pose pool, how much of the gain survives?

The bit is worth importing because it is the thing we provably cannot get alone: 79-81% of our
pool threads the groove backwards, ref2015 prefers the forward pose only 41% of the time, and the
best purely geometric descriptor we have reaches AUC 0.603 at telling them apart. Meanwhile on
9CCE the co-folded axis identified the forward poses in our pool with 98% precision.

Everything else here is ours: our docked poses, our per-pose ref2015 interface energies, our
median-polish cancellation. Nothing is refitted, so there is no training set to leak.

THREE CONTROLS, because a filter that discards poses will change a score whether or not the
direction means anything:

  shuffled prior   each cell is filtered by ANOTHER cell's co-folded axis. Same number of poses
                   discarded, same selection pressure, no matching information. If this scores
                   as well as the real prior, the gain is pose-pruning and not direction.
  reversed prior   the axis is negated, so the filter keeps exactly the poses the real prior
                   rejects. This should be worse than no filter at all, and if it is not, the
                   axis is not carrying threading direction.
  ours alone       filter on our own axis_quality with no prior at all -- the fully independent
                   version, and the honest measure of what we can do today without Boltz.

Usage: coventry_direction_filter.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
N = len(ORDER)
POSES = ROOT / "runs/coventry/hybridock_ft"
#: Binders whose co-folded pose was re-transplanted onto the AF3 model: for these the prior axis
#: lives in the AF3 frame while our docked poses live in the ESMFold frame, so the cosine between
#: them is not meaningful. Reported separately rather than silently mixed in.
BROKEN = {"n3", "n7", "pc21", "pc26"}


def ca_coords(pdb: Path) -> np.ndarray:
    out = []
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(out)


def grade(S, keep, label) -> tuple:
    from hybridock_pep.scoring.cancellation import two_way
    P = two_way(S, robust=True)
    cog = np.array([P[i, i] for i in range(N)])
    non = np.array([P[i, j] for i in range(N) for j in range(N) if i != j and keep[i, j]])
    a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
    rr = [sorted(range(N), key=lambda j: P[i, j]).index(i) + 1 for i in range(N)]
    print(f"  {label:<46}{a:>7.3f}{np.mean(rr):>7.2f}{sum(r <= 3 for r in rr):>5}"
          f"{sum(r <= 5 for r in rr):>5}")
    return a, float(np.mean(rr))


def main() -> None:
    from hybridock_pep.analysis.direction import agreement, axis_quality

    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    keep = np.zeros((N, N), dtype=bool)
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            keep[i, j] = (i == j) or truth.get((p, b), {}).get("measured") == "0"

    prior = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("axis"):
                prior[(r["peptide"], r["binder"])] = np.array(r["axis"], dtype=float)

    # per-pose interface energies, already computed -- nothing is rescored here
    energy = {}
    for line in (ROOT / "logs/coventry_full_terms.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            energy[r["name"]] = {row["pose"]: float(row["total"]) for row in r["rows"]}

    print("Loading pose geometry for 324 cells...", flush=True)
    cells = {}
    for i, p in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            name = f"{p}__{b}B"
            e = energy.get(name, {})
            recs = []
            for pdb in sorted((POSES / name).glob("rank*.pdb")):
                if pdb.name not in e:
                    continue
                ca = ca_coords(pdb)
                if len(ca) < 4:
                    continue
                recs.append({"e": e[pdb.name], "ca": ca, "q": axis_quality(ca)})
            cells[(i, j)] = recs
    npose = [len(v) for v in cells.values()]
    print(f"  {sum(npose)} poses, {np.mean(npose):.1f} per cell\n")

    def build(select) -> np.ndarray:
        """Grid of the best surviving interface energy; falls back to the pool if all are cut."""
        S = np.zeros((N, N))
        for (i, j), recs in cells.items():
            kept = [r for r in recs if select(i, j, r)] or recs
            S[i, j] = min(r["e"] for r in kept)
        return S

    print(f"  {'filter on our own pose pool':<46}{'AUC':>7}{'rank':>7}{'t3':>5}{'t5':>5}")
    base = grade(build(lambda i, j, r: True), keep, "no filter  (all poses, best energy)")

    rng = np.random.default_rng(0)
    keys = list(prior)
    for cut in (0.0, 0.3, 0.5):
        def real(i, j, r, c=cut):
            a = prior.get((ORDER[i], ORDER[j]))
            return a is None or agreement(r["ca"], a) > c
        grade(build(real), keep, f"co-folded direction prior, cos > {cut:+.1f}")

    print()
    shuf = {k: prior[keys[(n + 7) % len(keys)]] for n, k in enumerate(keys)}
    for label, table, cut in (("shuffled prior (control)", shuf, 0.3),
                              ("reversed prior (control)", {k: -v for k, v in prior.items()}, 0.3)):
        def f(i, j, r, t=table, c=cut):
            a = t.get((ORDER[i], ORDER[j]))
            return a is None or agreement(r["ca"], a) > c
        grade(build(f), keep, f"{label}, cos > {cut:+.1f}")

    print()
    for q in (0.55, 0.70):
        grade(build(lambda i, j, r, t=q: r["q"] >= t), keep,
              f"OURS ALONE: axis_quality >= {q:.2f}, no prior")

    # the prior is only in a common frame for the 14 binders we did not re-transplant
    ok = [j for j, b in enumerate(ORDER) if b not in BROKEN]
    print(f"\n  restricted to the {len(ok)} binders whose prior shares our frame "
          f"(the other 4 were re-transplanted onto AF3):")
    sub = np.ix_(range(N), ok)

    def sub_grade(S, label):
        from hybridock_pep.scoring.cancellation import two_way
        P = two_way(S[sub], robust=True)
        cog = np.array([P[i, ok.index(i)] for i in range(N) if i in ok])
        non = np.array([P[i, jj] for i in range(N) for jj, j in enumerate(ok)
                        if i != j and keep[i, j]])
        a = float(np.mean([(c < non).mean() + 0.5 * (c == non).mean() for c in cog]))
        print(f"  {label:<46}{a:>7.3f}")

    sub_grade(build(lambda i, j, r: True), "no filter")
    def real3(i, j, r):
        a = prior.get((ORDER[i], ORDER[j]))
        return a is None or agreement(r["ca"], a) > 0.3
    sub_grade(build(real3), "co-folded direction prior, cos > +0.3")


if __name__ == "__main__":
    main()
