#!/usr/bin/env python
"""Test Coventry's specificity rule on the designed structures, then turn it into a usable term.

HIS RULE, 35:31: "binding energy comes from hydrophobicity -- either hydrophobicity on your target
or hydrophobicity on your peptide. And so the more polars you can have in your thing and still
have it work, the more likely it is to be specific... if your thing has a lot of hydrophobic
residues, I doubt it's savable. I mean, arginine is kind of sticky too." And 35:22:
"polytryptophan will stick to everything."

This is a quantitative, falsifiable claim about promiscuity, and the disordered-peptide grid is a
near-ideal test bed for it: eighteen peptides against eighteen binders, with the paper reporting
exactly which pairs bind. Promiscuity is therefore observable -- a peptide that binds three
different binders is promiscuous, one that binds only its own is specific -- and we now have the
designed structures, so interface composition is measurable rather than inferred from sequence.

THE PREDICTION, STATED BEFORE LOOKING: peptides whose binding is carried by hydrophobic and
arginine contact should appear in MORE measured cells; peptides whose binding is carried by
polar and salt-bridge contact should appear in fewer.

WHY THIS IS WORTH MORE THAN A CORRELATION. Every selectivity term we have tried so far has been a
pair feature -- something about peptide i against binder j. This rule is different in kind: it
predicts a peptide's promiscuity from the peptide side alone, before any pairing. That is exactly
the main effect our cancellation machinery DELETES, which is why we would never have found it by
staring at interaction residuals, and why it is complementary rather than redundant to everything
in the current stack.

Two independent measures of interface hydrophobicity are computed so the result cannot rest on
one arbitrary scale: contact-weighted (which residues actually touch the binder, from the designed
structure) and composition-only (sequence alone, no structure). If the structural one wins, the
structures earned their keep; if sequence alone suffices, the rule is cheaper than we thought.

Usage: coventry_hydrophobicity_rule.py
"""
from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
PAPER = ROOT / "datasets/coventry/binders_paper"
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
CONTACT = 5.0

#: Kyte-Doolittle. Positive = hydrophobic.
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
      'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
      'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
#: Coventry names these three as sticky-regardless-of-scale: hydrophobics plus R, plus W.
STICKY = set("AVLIMFWCY") | set("R")
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def residues(p: Path, want_chain: str | None = None):
    out, cur = [], None
    for l in p.read_text().splitlines():
        if not l.startswith("ATOM") or (l[76:78].strip() == "H"):
            continue
        k = (l[21], l[22:27])
        if k != cur:
            out.append([AA3.get(l[17:20].strip(), 'X'), []])
            cur = k
        out[-1][1].append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return [(a, np.array(c)) for a, c in out if len(c)]


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    seqs = {p: next(r["peptide_seq"] for (pp, _), r in truth.items() if pp == p) for p in ORDER}

    rows = []
    for p in ORDER:
        cpx_b = PAPER / f"{p}_1b1.pdb"
        cpx_p = PAPER / f"{p}_peptide.pdb"
        if not (cpx_b.exists() and cpx_p.exists()):
            continue
        B, P = residues(cpx_b), residues(cpx_p)

        # contact-weighted hydrophobicity: each peptide residue counted by how much it touches
        wsum = hsum = ssum = 0.0
        for aa, pc in P:
            n = 0
            for bb, bc in B:
                d = np.linalg.norm(pc[:, None, :] - bc[None, :, :], axis=2)
                n += int((d < CONTACT).sum())
            if n:
                wsum += n
                hsum += n * KD.get(aa, 0.0)
                ssum += n * (1.0 if aa in STICKY else 0.0)
        cw_kd = hsum / wsum if wsum else float("nan")
        cw_sticky = ssum / wsum if wsum else float("nan")

        s = seqs[p]
        comp_kd = st.mean(KD.get(c, 0.0) for c in s)
        comp_sticky = sum(1 for c in s if c in STICKY) / len(s)

        n_meas = sum(1 for b in ORDER if truth.get((p, b), {}).get("measured") == "1")
        cross = n_meas - 1                       # partners beyond its own cognate
        kd = truth.get((p, p), {}).get("kd_M")
        pkd = -math.log10(float(kd)) if kd else float("nan")
        rows.append(dict(pep=p, cw_kd=cw_kd, cw_sticky=cw_sticky, comp_kd=comp_kd,
                         comp_sticky=comp_sticky, cross=cross, pkd=pkd, contacts=wsum,
                         seq=s))

    print("Coventry 35:31 — 'the more polars you can have and still have it work, the more "
          "likely it is to be specific'\n")
    print(f"  {'pep':<7}{'seq':<19}{'cross':>6}{'cw-KD':>8}{'cw-sticky':>11}"
          f"{'comp-KD':>9}{'comp-sticky':>13}{'cog pKd':>9}")
    for r in sorted(rows, key=lambda r: -r["cross"]):
        print(f"  {r['pep']:<7}{r['seq'][:18]:<19}{r['cross']:>6}{r['cw_kd']:>8.2f}"
              f"{r['cw_sticky']:>11.2f}{r['comp_kd']:>9.2f}{r['comp_sticky']:>13.2f}"
              f"{r['pkd']:>9.2f}")

    cross = [r["cross"] for r in rows]
    print("\nDOES INTERFACE HYDROPHOBICITY PREDICT PROMISCUITY?")
    print("(positive r = more hydrophobic -> binds MORE partners, which is his prediction)\n")
    print(f"  {'measure':<42}{'Pearson':>9}{'Spearman':>10}")
    for label, key in (("contact-weighted Kyte-Doolittle (structure)", "cw_kd"),
                       ("contact-weighted sticky fraction (structure)", "cw_sticky"),
                       ("composition Kyte-Doolittle (sequence only)", "comp_kd"),
                       ("composition sticky fraction (sequence only)", "comp_sticky")):
        v = [r[key] for r in rows]
        print(f"  {label:<42}{pearson(v, cross):>9.3f}{spearman(v, cross):>10.3f}")

    print(f"\n  control — cognate affinity vs promiscuity: "
          f"{pearson([r['pkd'] for r in rows], cross):+.3f} "
          f"(a strong value here would mean we are just measuring 'binds well')")
    print(f"  control — interface size vs promiscuity:   "
          f"{pearson([r['contacts'] for r in rows], cross):+.3f} "
          f"(bigger interface could trivially mean more partners)")

    prom = [r for r in rows if r["cross"] >= 2]
    spec = [r for r in rows if r["cross"] == 0]
    print(f"\n  promiscuous (2+ extra partners, n={len(prom)}): "
          f"cw-KD {st.mean(r['cw_kd'] for r in prom):+.2f}, "
          f"sticky {st.mean(r['cw_sticky'] for r in prom):.2f}")
    print(f"  specific    (0 extra partners,  n={len(spec)}): "
          f"cw-KD {st.mean(r['cw_kd'] for r in spec):+.2f}, "
          f"sticky {st.mean(r['cw_sticky'] for r in spec):.2f}")
    print(f"  -> promiscuous peptides are "
          f"{st.mean(r['cw_kd'] for r in prom) - st.mean(r['cw_kd'] for r in spec):+.2f} "
          f"KD units more hydrophobic at the interface")


if __name__ == "__main__":
    main()
