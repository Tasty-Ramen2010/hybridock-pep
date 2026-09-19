#!/usr/bin/env python
"""Full pretrained-vs-finetuned table for the long-peptide Tanh finetune (longft_tanh).

Pairs every arm complex-by-complex against the pretrained model run through the SAME code
path (fork + Tanh), restricted to the <=25 aa scope, and reports by length band and SS.

Arms (logs/longft_bench_<label>.log):
  long:  pretrained_tanhfix (control) vs long_tanh_ep20 / long_tanh_mid / long_tanh_ep05
  short: short_pretrained_tanh (control) vs short_tanh_ep20
Missing arms are reported as pending, not skipped silently.
"""
from __future__ import annotations

import csv
import os
import re
import statistics as st
from math import comb

LOG = "logs/longft_bench_{}.log"
LINE = re.compile(r"^\[([^\]]+)\] partial=\S+ n=\d+ dt=[\d.]+s best_direct=([\d.]+)A "
                  r"spread=[\d.]+A nearnative=(\d+)/(\d+)")


def load(label: str) -> dict[str, tuple[float, float]] | None:
    path = LOG.format(label)
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)) / int(m.group(4)))
    return out or None


def meta() -> dict[str, dict[str, str]]:
    m = {}
    for f in ("data/bench_long.csv", "data/bench_short.csv"):
        for r in csv.DictReader(open(f)):
            m[r["name"]] = r
    return m


def band(n: int) -> str:
    return "05-08" if n <= 8 else "09-12" if n <= 12 else "13-16" if n <= 16 else "17-25"


def sign_p(better: int, worse: int) -> float:
    m = better + worse
    if m == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def row(label: str, ctrl: dict, arm: dict, keys: list[str]) -> str:
    k = [n for n in keys if n in ctrl and n in arm]
    if not k:
        return f"  {label:22s} (no paired complexes)"
    c = [ctrl[n][0] for n in k]
    a = [arm[n][0] for n in k]
    d = [arm[n][0] - ctrl[n][0] for n in k]
    b = sum(x < -0.5 for x in d)
    w = sum(x > 0.5 for x in d)
    p = sign_p(sum(x < 0 for x in d), sum(x > 0 for x in d))
    return (f"  {label:22s} n={len(k):3d} | median {st.median(c):5.2f} -> {st.median(a):5.2f}A "
            f"| <=2A {sum(x <= 2 for x in c):2d} -> {sum(x <= 2 for x in a):2d} "
            f"| <=5A {sum(x <= 5 for x in c):2d} -> {sum(x <= 5 for x in a):2d} "
            f"| nn {st.mean(ctrl[n][1] for n in k):.3f} -> {st.mean(arm[n][1] for n in k):.3f} "
            f"| better/worse {b:2d}/{w:2d} | p={p:.4f}")


def main() -> None:
    M = meta()
    L = lambda n: int(M[n]["pep_len"])  # noqa: E731
    inscope = lambda d: [n for n in d if n in M and L(n) <= 25]  # noqa: E731

    pairs = [
        ("LONG 13-25 aa", "longft_bench_long_pretrained_tanhfix".replace("longft_bench_", ""),
         ["long_tanh_ep20", "long_tanh_mid", "long_tanh_ep05"]),
        ("SHORT/MED <=12 aa", "short_pretrained_tanh", ["short_tanh_ep20"]),
    ]
    for title, ctrl_label, arms in pairs:
        ctrl = load(ctrl_label)
        print(f"\n{'=' * 118}\n{title}   control = pretrained, same fork+Tanh code "
              f"({ctrl_label}){'' if ctrl else '  -- PENDING'}\n{'=' * 118}")
        if not ctrl:
            continue
        for a_label in arms:
            arm = load(a_label)
            if not arm:
                print(f"  {a_label}: PENDING")
                continue
            keys = inscope(arm)
            print(f"--- {a_label} ---")
            print(row("ALL in scope", ctrl, arm, keys))
            for b in ("05-08", "09-12", "13-16", "17-25"):
                kb = [n for n in keys if band(L(n)) == b]
                if kb:
                    print(row(f"  {b} aa", ctrl, arm, kb))
            for s in ("HELIX", "SHEET", "UNUSUAL"):
                ks = [n for n in keys if M[n]["ss_class"] == s]
                if ks:
                    print(row(f"  {s}", ctrl, arm, ks))


if __name__ == "__main__":
    main()
