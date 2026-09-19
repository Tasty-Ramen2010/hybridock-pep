#!/usr/bin/env python
"""Are protein-protein interfaces small enough to train a peptide interface scorer on?

THE IDEA, AND WHY IT IS THE RIGHT SHAPE OF ANSWER TO OUR ACTUAL PROBLEM. Every failure this
project has hit traces to the same root: 1,500 usable protein-peptide complexes is not enough
data, and Coventry said the quiet part out loud -- "the architecture matters less than the data
you give it", and "you might consider your thing to be like an interface scorer rather than a
full complex predictor". Those two remarks together point somewhere specific. If the thing we
score is an INTERFACE rather than a complex, then a protein-protein interface is the same kind of
object as a protein-peptide interface, and SKEMPI alone carries 7,086 measured affinities over
323 structures -- several times our entire peptide corpus.

THE FEASIBILITY QUESTION THAT DECIDES IT, which is Ram's: are they small enough? A peptide buries
one short chain against a groove. If a typical PPI buries ten times that area across a flat
patch, the features will not transfer, because burial, enclosure and contact counts will all sit
in a completely different regime and a model trained there will be extrapolating when it meets a
peptide. So this measures both populations on the SAME definitions before any training is
proposed, and reports the overlap rather than assuming one.

WHAT IS MEASURED PER COMPLEX. Interface residues on each side (heavy atoms within 5 A across the
partner), total cross-interface contacts, and the size of the SMALLER side -- which is the number
that matters, because the smaller side is what plays the role of the peptide. A PPI whose smaller
side is 8-25 residues is, for our purposes, a peptide complex that happens to come attached to
more protein.

No cropping or repacking happens here. Coventry, 9:05: "take the interface section. Again, don't
touch it after that point, because you'll change it in a way that's probably bad."

Usage: ppi_interface_survey.py [max_structures]
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SKEMPI = ROOT / "datasets/skempi"
PAPER = ROOT / "datasets/coventry/binders_paper"
CUT = 5.0
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def chains(p: Path) -> dict[str, list]:
    out, cur = defaultdict(list), None
    for l in p.read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM") or l[76:78].strip() == "H":
            continue
        alt = l[16]
        if alt not in (" ", "A"):
            continue
        k = (l[21], l[22:27])
        if k != cur:
            out[l[21]].append([])
            cur = k
        out[l[21]][-1].append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return {c: [np.array(r) for r in v if len(r)] for c, v in out.items()}


def interface(A: list, B: list, cut: float = CUT):
    """(n interface residues in A, n in B, total atom-atom contacts)."""
    ia = ib = 0
    contacts = 0
    bflat = np.vstack(B) if B else np.zeros((0, 3))
    hit_b = np.zeros(len(bflat), dtype=bool)
    idx, off = [], 0
    for r in B:
        idx.append((off, off + len(r))); off += len(r)
    for ra in A:
        if not len(bflat):
            break
        d = np.linalg.norm(ra[:, None, :] - bflat[None, :, :], axis=2)
        m = d < cut
        if m.any():
            ia += 1
            contacts += int(m.sum())
            hit_b |= m.any(0)
    for s, e in idx:
        if hit_b[s:e].any():
            ib += 1
    return ia, ib, contacts


def main() -> None:
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 120

    # partner chain definitions from SKEMPI, so we split the real biological interface
    pair = {}
    with open(ROOT / "data/skempi_v2.csv") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            pdb = r["#Pdb"]
            if "_" in pdb:
                bits = pdb.split("_")
                if len(bits) >= 3:
                    pair.setdefault(bits[0].upper(), (bits[1], bits[2]))

    print("PEPTIDE reference — the eighteen designed complexes (our actual target regime)\n")
    pep_small, pep_ct = [], []
    for t in ORDER:
        b, p = PAPER / f"{t}_1b1.pdb", PAPER / f"{t}_peptide.pdb"
        if not (b.exists() and p.exists()):
            continue
        A = list(chains(b).values())[0]
        B = list(chains(p).values())[0]
        ia, ib, ct = interface(A, B)
        pep_small.append(ib); pep_ct.append(ct)
    print(f"  smaller side (the peptide): median {st.median(pep_small):.0f} interface residues, "
          f"range {min(pep_small)}-{max(pep_small)}")
    print(f"  cross-interface contacts:   median {st.median(pep_ct):.0f}")

    print(f"\nPROTEIN-PROTEIN — SKEMPI structures (up to {lim})\n")
    print(f"  {'pdb':<7}{'chA if':>8}{'chB if':>8}{'smaller':>9}{'contacts':>10}")
    small, ct_all, rows = [], [], []
    files = sorted(SKEMPI.glob("*.pdb"))[:lim]
    for f in files:
        pdb = f.stem.upper()
        ch = chains(f)
        if len(ch) < 2:
            continue
        p1, p2 = pair.get(pdb, (None, None))
        if p1 and p2 and all(c in ch for c in p1) and all(c in ch for c in p2):
            A = [r for c in p1 for r in ch[c]]
            B = [r for c in p2 for r in ch[c]]
        else:
            ks = sorted(ch, key=lambda c: -len(ch[c]))[:2]
            A, B = ch[ks[0]], ch[ks[1]]
        ia, ib, ct = interface(A, B)
        if ia == 0 or ib == 0:
            continue
        sm = min(ia, ib)
        small.append(sm); ct_all.append(ct)
        rows.append((pdb, ia, ib, sm, ct))
    for r in rows[:14]:
        print(f"  {r[0]:<7}{r[1]:>8}{r[2]:>8}{r[3]:>9}{r[4]:>10}")
    print(f"  ... {len(rows)} structures measured")

    print(f"\n  smaller side: median {st.median(small):.0f} residues, "
          f"range {min(small)}-{max(small)}, IQR "
          f"{np.percentile(small, 25):.0f}-{np.percentile(small, 75):.0f}")
    print(f"  contacts:     median {st.median(ct_all):.0f}")

    lo, hi = min(pep_small), max(pep_small)
    inband = sum(1 for s in small if lo <= s <= hi)
    print(f"\nTHE VERDICT ON SIZE")
    print(f"  peptide interfaces span {lo}-{hi} residues on the smaller side.")
    print(f"  PPI interfaces inside that same band: {inband}/{len(small)} "
          f"({100 * inband / len(small):.0f}%)")
    print(f"  PPI interfaces within 2x the peptide max (<= {2 * hi}): "
          f"{sum(1 for s in small if s <= 2 * hi)}/{len(small)} "
          f"({100 * sum(1 for s in small if s <= 2 * hi) / len(small):.0f}%)")
    ratio = st.median(small) / st.median(pep_small)
    print(f"  median PPI interface is {ratio:.1f}x the peptide one")
    print("\n  -> " + ("SAME REGIME. Crop and train; the features will transfer."
                       if ratio < 2.0 else
                       "LARGER REGIME. Usable, but only the small tail sits where peptides live "
                       "-- filter by interface size rather than training on all of it."))


if __name__ == "__main__":
    main()
