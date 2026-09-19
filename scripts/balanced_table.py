#!/usr/bin/env python
"""Original RAPiDock vs our finetune on the length-balanced post-2020 benchmark.

240 complexes, 60 in each band (5-8 / 9-12 / 13-16 / 17-25 aa), deposited after RAPiDock's
training cutoff and sharing no PDB id with our finetune data. The point of this set is the
long bands: RAPiDock's own test set is almost entirely short and medium peptides, so it
cannot show whether the finetune's long-peptide gain holds on unseen structures.

Two metrics, both best-of-24 (our poses are UNRANKED, so only an oracle number is defined
-- never compare these to a published ranked top-1):
  direct RMSD   parsed from logs/balanced_<arm>.log, the same number the Sep-13 run reported
  DockQ         logs/dockq_balanced.jsonl, CAPRI-peptide, written by dockq_rs.py

Pairing is per complex, so the Wilcoxon test asks the only question that matters: on the
same structure, does the finetuned checkpoint beat the one it was finetuned from?

Usage: python scripts/balanced_table.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
BENCH = ROOT / "data/bench_balanced_post2020.csv"
ARMS = [("rapidock_og", "RAPiDock (original)"), ("hybridock_ft", "ours (finetuned)")]
BANDS = ["05-08", "09-12", "13-16", "17-25"]
SS = ["HELIX", "SHEET", "PPII", "UNUSUAL"]
RE_RES = re.compile(r"^\[(\S+)\] .*best_direct=([0-9.]+)A.*nearnative=(\d+)/(\d+)")


def read_rmsd(arm: str) -> dict[str, float]:
    """complex -> best-of-N direct RMSD, from the arm's run log."""
    log = ROOT / f"logs/balanced_{arm}.log"
    out: dict[str, float] = {}
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = RE_RES.match(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def read_dockq(arm: str) -> dict[str, float]:
    """complex -> best-of-N DockQ."""
    f = ROOT / "logs/dockq_balanced.jsonl"
    out: dict[str, float] = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["arm"] != arm:
                continue
            vals = [v for v in r["dockq"] if v == v]  # drop NaN
            if vals:
                out[r["name"]] = max(vals)
    return out


def wilcoxon(a: list[float], b: list[float]) -> str:
    """Two-sided paired test; falls back to a sign test if scipy is absent."""
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y and x != y]
    if len(pairs) < 6:
        return "n/a"
    try:
        from scipy.stats import wilcoxon as w

        return f"p={w([x for x, _ in pairs], [y for _, y in pairs]).pvalue:.4g}"
    except ImportError:
        from statistics import NormalDist

        wins = sum(1 for x, y in pairs if y < x)
        n = len(pairs)
        z = (wins - n / 2) / (n**0.5 / 2)
        return f"sign p={2 * (1 - NormalDist().cdf(abs(z))):.4g}"


def block(title: str, rows: list[tuple[str, list[str]]], hdr: list[str]) -> None:
    print(f"\n{title}")
    widths = [max(len(hdr[i]), *(len(r[1][i]) for r in rows)) for i in range(len(hdr))]
    label_w = max(len(r[0]) for r in rows + [("group", [])])
    print("  " + "group".ljust(label_w) + "  " + "  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for label, cells in rows:
        print("  " + label.ljust(label_w) + "  "
              + "  ".join(c.rjust(w) for c, w in zip(cells, widths)))


def summarize(names: list[str], rm: dict, dq: dict) -> list[str]:
    r = [rm[n] for n in names if n in rm]
    d = [dq[n] for n in names if n in dq]
    if not r:
        return ["-"] * 5
    return [
        f"{st.median(r):.2f}",
        f"{sum(1 for x in r if x <= 2)}/{len(r)}",
        f"{sum(1 for x in r if x <= 5)}/{len(r)}",
        f"{sum(1 for x in d if x >= 0.23)}/{len(d)}" if d else "-",
        f"{st.median(d):.3f}" if d else "-",
    ]


def main() -> None:
    meta = {r["name"]: r for r in csv.DictReader(open(BENCH))}
    rmsd = {a: read_rmsd(a) for a, _ in ARMS}
    dockq = {a: read_dockq(a) for a, _ in ARMS}
    scored = {a: sorted(set(rmsd[a]) & set(meta)) for a, _ in ARMS}

    print("=" * 78)
    print("LENGTH-BALANCED POST-2020 BENCHMARK  (240 complexes, 60 per band)")
    print("=" * 78)
    for arm, label in ARMS:
        print(f"  {label:22s} {len(scored[arm]):3d} complexes docked, "
              f"{len(dockq[arm]):3d} DockQ-scored")
    if not all(scored[a] for a, _ in ARMS):
        print("\nBoth arms must finish before the comparison is meaningful. Stopping here.")
        return

    common = sorted(set(scored["rapidock_og"]) & set(scored["hybridock_ft"]))
    print(f"  paired on {len(common)} complexes both arms completed")
    hdr = ["medRMSD", "<=2A", "<=5A", "DockQ>=.23", "medDockQ"]

    for group_name, key, keys in [("BY LENGTH BAND", "length_bucket", BANDS),
                                  ("BY SECONDARY STRUCTURE", "ss_class", SS)]:
        for arm, label in ARMS:
            rows = []
            for k in keys:
                names = [n for n in common if meta[n][key] == k]
                if names:
                    rows.append((f"{k} (n={len(names)})", summarize(names, rmsd[arm], dockq[arm])))
            if rows:
                block(f"{group_name} -- {label}", rows, hdr)

    print("\nPAIRED COMPARISON (same complex, both checkpoints; lower RMSD is better)")
    print(f"  {'group':18s} {'n':>4s}  {'og med':>7s}  {'ft med':>7s}  {'ft wins':>8s}  test")
    for key, keys in [("length_bucket", BANDS), ("ss_class", SS), ("", ["ALL"])]:
        for k in keys:
            names = [n for n in common if not key or meta[n][key] == k]
            a = [rmsd["rapidock_og"][n] for n in names]
            b = [rmsd["hybridock_ft"][n] for n in names]
            if len(names) < 3:
                continue
            wins = sum(1 for x, y in zip(a, b) if y < x)
            print(f"  {k:18s} {len(names):4d}  {st.median(a):7.2f}  {st.median(b):7.2f}  "
                  f"{wins:4d}/{len(names):<3d}  {wilcoxon(a, b)}")

    print("\nCaveats: poses are UNRANKED, so every number is oracle best-of-24 and is NOT")
    print("comparable to a published ranked top-1. Both arms run through the same")
    print("(Tanh-fixed) code path, so the checkpoint is the only variable.")


if __name__ == "__main__":
    main()
