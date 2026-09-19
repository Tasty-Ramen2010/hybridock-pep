#!/usr/bin/env python
"""Parse Cao 2022 SSM into a clean table, and locate the interface WITHOUT structures.

WHAT THIS DATA IS. Thirteen targets, each with a site-saturation scan over its designed binders:
every position mutated to every amino acid, with SC50 measured by yeast display. Names encode it
directly as <parent>_<position>_<aa>, plus a `_native` row per parent for the unmutated design.
229,961 variants, which is roughly 150x our entire protein-peptide corpus.

THE PROBLEM COVENTRY FLAGGED (41:42): "it's a three helical bundle on top of an interface. You
don't care about mutations up here. You only care about mutations down here." Most positions are
scaffold. Mutating them changes nothing about binding, so training on all 229,961 rows means
training mostly on noise about positions that do not participate.

HIS SOLUTION NEEDS STRUCTURES WE DO NOT HAVE -- the tarball is scores only, and at 41:02 he says
"you probably have to alpha fold all these". So the interface is located from the DATA instead: a
position where substitutions routinely destroy binding is, functionally, an interface position;
one where every substitution is tolerated is scaffold. That is a weaker definition than a distance
cutoff, but it is the one that matters for scoring, and unlike a distance cutoff it cannot be
wrong about a residue that is close but does nothing.

Sensitivity per position = the fraction of its 19 substitutions that lose measurable binding,
given that the parent itself binds. Parents whose native does not bind are dropped: a scan around
a non-binder has nothing to say about what breaks binding.

Coventry's claim is then testable and is tested here: if he is right, sensitivity should be
concentrated in a minority of positions rather than spread evenly.

Usage: cao_ssm_parse.py
Output: data/cao_ssm.csv  (one row per variant)  +  data/cao_ssm_positions.csv
"""
from __future__ import annotations

import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SRC = ROOT / "datasets/cao2022/cao_2022_affinity_updated"
OUT = ROOT / "data/cao_ssm.csv"
POSOUT = ROOT / "data/cao_ssm_positions.csv"
AAS = set("ACDEFGHIKLMNPQRSTVWY")
#: <parent>_<pos>_<aa> ; parent names themselves contain underscores and digits, so anchor on
#: the tail: a number then a single amino-acid letter at the very end.
MUT = re.compile(r"^(?P<parent>.+)_(?P<pos>\d+)_(?P<aa>[A-Z])$")


def validity() -> dict[str, str]:
    """Parent -> strictest validation tier it appears in."""
    tier = {}
    for name, lvl in (("somewhat_valid_ssms.list", "somewhat"),
                      ("reasonably_valid_ssms.list", "reasonable"),
                      ("quite_valid_ssms.list", "quite")):
        f = SRC / name
        if f.exists():
            for l in f.read_text().split():
                if l.strip():
                    tier[l.strip()] = lvl
    return tier


def main() -> None:
    tiers = validity()
    rows, natives = [], {}
    for f in sorted(SRC.glob("*_ssm.sc")):
        target = f.name.replace("_ssm.sc", "")
        lines = f.read_text().splitlines()
        hdr = lines[0].split()
        ix = {c: i for i, c in enumerate(hdr)}
        for l in lines[1:]:
            p = l.split()
            if len(p) < len(hdr):
                continue
            desc = p[ix["description"]]
            try:
                sc50 = float(p[ix["sc50_est"]])
            except (ValueError, KeyError):
                sc50 = float("inf")
            unmeas = p[ix["sc50_unmeasurable"]].lower() in ("true", "1")
            try:
                pbind = float(p[ix["p_bind"]])
            except (ValueError, KeyError):
                pbind = float("nan")
            lowconf = p[ix.get("low_conf", 0)].lower() in ("true", "1") if "low_conf" in ix else False

            if desc.endswith("_native"):
                parent = desc[: -len("_native")]
                natives[(target, parent)] = (sc50, unmeas)
                rows.append(dict(target=target, parent=parent, pos=0, mut_aa="native",
                                 sc50=sc50, unmeasurable=int(unmeas), p_bind=pbind,
                                 low_conf=int(lowconf), tier=tiers.get(parent, "")))
                continue
            m = MUT.match(desc)
            if not m or m.group("aa") not in AAS:
                continue
            rows.append(dict(target=target, parent=m.group("parent"),
                             pos=int(m.group("pos")), mut_aa=m.group("aa"),
                             sc50=sc50, unmeasurable=int(unmeas), p_bind=pbind,
                             low_conf=int(lowconf), tier=tiers.get(m.group("parent"), "")))

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"parsed {len(rows)} variants over "
          f"{len({(r['target'], r['parent']) for r in rows})} parent designs, "
          f"{len({r['target'] for r in rows})} targets -> {OUT.relative_to(ROOT)}")
    for lvl in ("quite", "reasonable", "somewhat", ""):
        n = len({(r["target"], r["parent"]) for r in rows if r["tier"] == lvl})
        if n:
            print(f"    {n:>4} parents at validity tier '{lvl or 'unlisted'}'")

    # ---------------------------------------------------------------- position sensitivity
    per = defaultdict(list)
    for r in rows:
        if r["pos"] > 0:
            per[(r["target"], r["parent"], r["pos"])].append(r)

    prows = []
    for (t, parent, pos), muts in per.items():
        nat = natives.get((t, parent))
        if not nat or nat[1]:                 # parent must itself be a measurable binder
            continue
        n = len(muts)
        if n < 10:
            continue
        killed = sum(1 for m in muts if m["unmeasurable"])
        finite = [m["sc50"] for m in muts if np.isfinite(m["sc50"])]
        prows.append(dict(target=t, parent=parent, pos=pos, n_muts=n,
                          frac_killed=killed / n,
                          med_sc50=float(np.median(finite)) if finite else float("nan"),
                          native_sc50=nat[0]))

    with POSOUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prows[0]))
        w.writeheader(); w.writerows(prows)

    fk = np.array([p["frac_killed"] for p in prows])
    print(f"\nPOSITION SENSITIVITY — {len(prows)} positions on parents that themselves bind\n")
    print("  fraction of the 19 substitutions that abolish measurable binding:")
    for lo, hi, lbl in ((0.0, 0.1, "0-10%   tolerant (scaffold)"),
                        (0.1, 0.3, "10-30%  mildly sensitive"),
                        (0.3, 0.6, "30-60%  sensitive"),
                        (0.6, 0.9, "60-90%  hot"),
                        (0.9, 1.01, "90-100% critical")):
        n = int(((fk >= lo) & (fk < hi)).sum())
        print(f"    {lbl:<30}{n:>6}  ({100 * n / len(fk):>5.1f}%)")
    print(f"\n  median {np.median(fk):.2f}, mean {fk.mean():.2f}")
    hot = fk >= 0.6
    print(f"  positions at 60%+ kill rate: {int(hot.sum())}/{len(fk)} "
          f"({100 * hot.mean():.1f}%)")
    print("\n  -> " + ("CONCENTRATED, as Coventry predicted: a minority of positions carry the "
                       "binding. Training on all positions dilutes the signal."
                       if hot.mean() < 0.4 else
                       "sensitivity is NOT concentrated — the scaffold/interface split is less "
                       "clean than the 'three-helix bundle on top of an interface' picture."))

    bypar = defaultdict(list)
    for p in prows:
        bypar[(p["target"], p["parent"])].append(p["frac_killed"])
    conc = [sum(1 for v in vs if v >= 0.6) / len(vs) for vs in bypar.values() if len(vs) >= 20]
    if conc:
        print(f"  per design, share of its own positions that are hot: "
              f"median {st.median(conc) * 100:.0f}% (n={len(conc)} designs)")


if __name__ == "__main__":
    main()
