#!/usr/bin/env python
"""Three-way table on RAPiDock's own test set (leak-free RecentSet, 345 complexes).

One metric for everyone: DockQ with capri_peptide=True, the standard the RAPiDock paper
reports. CAPRI levels: acceptable >=0.23, medium >=0.49, high >=0.80.

Reads:
  logs/dockq_recentset.jsonl  {name, arm, dockq:[...]}   arms: rapidock_og, hybridock_ft
  logs/dockq_adcp.jsonl       {name, dockq:[...]}        ADCP ranked modes 1..10

Reporting rules that keep this honest:
  * OUR poses are UNRANKED (exp_runner uses --scoring_function none --confidence_ckpt null),
    so there is no top-1 for RAPiDock-og or ours. Only best-of-24 (an ORACLE) is defined.
  * ADCP's modes ARE ranked by its own affinity, so top-1 / top-5 / best-of-10 are real.
  * Never compare our oracle to a published ranked top-1. The paper's numbers are printed
    as context only, on 523 complexes (ours: 345 after dropping leaks), never as a row.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
LEVELS = (("acceptable", 0.23), ("medium", 0.49), ("high", 0.80))


def load(path: Path, key: str = "arm", default: str = "adcp") -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(dict)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vals = [v for v in r["dockq"] if not (v is None or math.isnan(v))]
        out[r.get(key, default)][r["name"]] = vals
    return out


def rate(vals: list[float], thr: float) -> bool:
    return bool(vals) and max(vals) >= thr


def summarise(label: str, per_complex: dict[str, list[float]], k: int | None) -> str:
    """k = how many top poses are allowed; None = all available."""
    names = sorted(per_complex)
    if not names:
        return f"  {label:34s} (no data)"
    sub = {n: (per_complex[n][:k] if k else per_complex[n]) for n in names}
    cells = []
    for lvl, thr in LEVELS:
        hits = sum(rate(v, thr) for v in sub.values())
        cells.append(f"{lvl[:4]} {100*hits/len(names):5.1f}%")
    best = [max(v) for v in sub.values() if v]
    return (f"  {label:34s} n={len(names):3d}  " + "  ".join(cells)
            + f"  medianDockQ {st.median(best):.3f}")


def main() -> None:
    ours = load(ROOT / "logs/dockq_recentset.jsonl")
    adcp = load(ROOT / "logs/dockq_adcp.jsonl")

    print("=" * 96)
    print("RAPiDock's own test set - RefPepDB-RecentSet, leak-free subset (345 of 523)")
    print("DockQ(capri_peptide=True); CAPRI: acceptable>=0.23 medium>=0.49 high>=0.80")
    print("=" * 96)

    print("\nUNRANKED sampling - ORACLE best-of-N (no pose selection; NOT a top-1):")
    for arm, lbl in (("rapidock_og", "RAPiDock original (best of 24)"),
                     ("hybridock_ft", "ours, fork tree  (best of 24)"),
                     ("hybridock_ft_shipped", "ours, SHIPPED path (best of 24)")):
        if arm in ours:
            print(summarise(lbl, ours[arm], None))

    # Same checkpoint, two code paths: is the shipped path's docking the same as the fork's?
    if "hybridock_ft" in ours and "hybridock_ft_shipped" in ours:
        f, s = ours["hybridock_ft"], ours["hybridock_ft_shipped"]
        both = sorted(set(f) & set(s))
        if both:
            df = [max(f[n]) for n in both]
            ds = [max(s[n]) for n in both]
            print(f"\nCODE-PATH CHECK, same checkpoint, {len(both)} complexes: fork vs shipped")
            print(f"  median best DockQ {st.median(df):.3f} vs {st.median(ds):.3f}   "
                  f"shipped better {sum(y > x + 0.02 for x, y in zip(df, ds))}  "
                  f"worse {sum(x > y + 0.02 for x, y in zip(df, ds))}")

    print("\nADCP - genuinely ranked by its own affinity:")
    if "adcp" in adcp:
        # Targets ADCP crashed on carry an empty pose list and count as failures, so its
        # denominator matches ours. The median is over the targets it did produce.
        n_empty = sum(1 for v in adcp["adcp"].values() if not v)
        if n_empty:
            print(f"  ({n_empty} of {len(adcp['adcp'])} produced no ranked output at all - "
                  f"ADCP's own clustering step crashed - and count as failures below;\n"
                  f"   the median is over the {len(adcp['adcp']) - n_empty} it did produce)")
        for k, lbl in ((1, "ADCP top-1"), (5, "ADCP top-5"), (None, "ADCP best of 10")):
            print(summarise(lbl, adcp["adcp"], k))

    # paired comparison, ours vs original, on complexes both arms finished
    if "rapidock_og" in ours and "hybridock_ft" in ours:
        a, b = ours["rapidock_og"], ours["hybridock_ft"]
        both = sorted(set(a) & set(b))
        if both:
            print(f"\nPAIRED, same {len(both)} complexes (best-of-24 DockQ):")
            da = [max(a[n]) for n in both]
            db = [max(b[n]) for n in both]
            better = sum(y > x + 0.02 for x, y in zip(da, db))
            worse = sum(x > y + 0.02 for x, y in zip(da, db))
            print(f"  median {st.median(da):.3f} -> {st.median(db):.3f}   "
                  f"better {better}  worse {worse}  tied {len(both)-better-worse}")
            for lvl, thr in LEVELS:
                ha = sum(x >= thr for x in da)
                hb = sum(x >= thr for x in db)
                print(f"  {lvl:11s} {ha:3d} -> {hb:3d}  ({100*ha/len(both):.1f}% -> {100*hb/len(both):.1f}%)")

    print("\nCONTEXT ONLY - the paper's published numbers (523 complexes, ranked top-N,")
    print("not directly comparable to the oracle rows above):")
    print("  RAPiDock   acceptable 81.3% top-1, 93.7% top-25 | medium 61.0/68.6/71.7% (top-1/5/25)")
    print("  AF2Multi_RS                                     | medium 56.4/58.7/61.8%")
    print("  ADCP                                            | medium 33.1/49.5/57.6%")


if __name__ == "__main__":
    main()
