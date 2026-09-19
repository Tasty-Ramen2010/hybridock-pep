#!/usr/bin/env python
"""Rebuild the binder set from the paper's own structures, and document what was wrong with ours.

WHY THIS EXISTS. Brian sent the AF2-multimer complexes behind the paper. Matching our binder
models against them by SEQUENCE -- not by filename -- turned up identity errors that no amount of
better modelling would have fixed, because the files were not the proteins their names claimed:

  pc17 / pc18   SWAPPED. Our pc17_1b1.pdb is 100% identical to the paper's pc18 binder and vice
                versa. Two columns of every result so far were each other's.
  pc11          NOT A GRID PROTEIN AT ALL. Our pc11_1b1.pdb is 100% identical to APE_1b2, a
                binder for a different target that does not appear in Fig. 2B.
  pc35          66% identity to the paper's pc35 -- the right target, a different design.

The other fourteen are 95.7-100% identical once an N-terminal MSG tag is accounted for, so they
were right all along -- including n3/n7/pc21/pc26, which we had flagged as "wrong models" on the
strength of Boltz and AF3 disagreeing with our ESMFold. That inference was correct about the
COORDINATES being poor and wrong about the cause.

THE MSG TAG MATTERS FOR MORE THAN BOOKKEEPING. Our sequences carry a three-residue MSG prefix the
paper's do not. Any residue-indexed comparison that ignores it is off by three, which silently
inflates every superposition RMSD -- it made all eighteen of our models look 5-6 A wrong on the
first pass of this audit, when most are correct.

WHAT THIS WRITES. datasets/coventry/binders_paper/<t>_1b1.pdb -- chain A of each designed complex,
which is the binder as the authors modelled it. Also <t>_peptide.pdb, the designed peptide pose,
which is the first ground truth we have had for what our docking is supposed to reproduce.

Usage: coventry_fix_binders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
SRC = ROOT / "datasets/coventry/paper_af2"
DST = ROOT / "datasets/coventry/binders_paper"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def chain_lengths(p: Path) -> dict[str, int]:
    n, seen = {}, set()
    for l in p.read_text().splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen:
                continue
            seen.add(k)
            n[l[21]] = n.get(l[21], 0) + 1
    return n


def write_chain(src: Path, dst: Path, chain: str) -> int:
    keep = [l for l in src.read_text().splitlines()
            if l.startswith(("ATOM", "TER")) and (len(l) > 21 and l[21] == chain)]
    dst.write_text("\n".join(keep) + "\nTER\nEND\n")
    return sum(1 for l in keep if l.startswith("ATOM") and l[12:16].strip() == "CA")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    print(f"  {'target':<8}{'binder ch':>10}{'res':>6}{'peptide ch':>12}{'res':>6}")
    ok = 0
    for t in ORDER:
        src = SRC / f"{t}_complex.pdb"
        if not src.exists():
            print(f"  {t:<8}  MISSING {src.name}")
            continue
        ln = chain_lengths(src)
        if len(ln) < 2:
            print(f"  {t:<8}  only {len(ln)} chain(s) — skipped")
            continue
        chains = sorted(ln, key=lambda c: -ln[c])
        b, p = chains[0], chains[1]
        nb = write_chain(src, DST / f"{t}_1b1.pdb", b)
        np_ = write_chain(src, DST / f"{t}_peptide.pdb", p)
        ok += 1
        print(f"  {t:<8}{b:>10}{nb:>6}{p:>12}{np_:>6}")
    print(f"\n  wrote {ok}/18 binders and their designed peptide poses to "
          f"{DST.relative_to(ROOT)}")
    print("\n  These replace datasets/coventry/binders for any future run. The old set has "
          "pc17/pc18 swapped, pc11 = APE_1b2, and a divergent pc35.")


if __name__ == "__main__":
    main()
