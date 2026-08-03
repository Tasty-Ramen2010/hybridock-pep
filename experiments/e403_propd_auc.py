"""E403 — turn E402's docking scores into specificity AUCs.

Reads `data/propd_dock_scores.jsonl` and reports how well the pipeline separates
ProP-PD binders from constructed non-binders.

Two discriminants, because the obvious one is not the fair one
-------------------------------------------------------------
**Raw ΔG.** The number the tool reports. But ΔG is an *absolute* affinity
estimate, and the benchmark spans 27 different receptors with interfaces from a
39-residue WW domain to a 476-residue TANK repeat. A bigger interface buries
more surface and scores more negative whether or not the peptide belongs there,
so a raw-ΔG ranking partly measures domain size. Reported first because it is
what a naive user would do.

**Per-domain z-score.** ΔG standardised within each domain, which is the
quantity the task actually asks about: *given this domain, is this peptide a
binder?* Every negative pair is a real peptide docked against a decoy domain, so
within a domain the comparison is like-for-like. This is the number to compare
against published specificity predictors — CliPepPI's cosine similarity behaves
the same way, since a domain embedding is fixed within its own column.

Both are reported. If they disagree, the gap is the size confound.

Reference (CliPepPI, Table 1, ProP-PD): CliPepPI AUC 0.72, AlphaFold2 actifpTM
AUC 0.90. Neither is a like-for-like comparison here — this is a reconstruction
of the benchmark from the same primary source, not their exact split — so treat
it as a scale reference, not a scoreboard.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "propd_dock_scores.jsonl"
OUT = ROOT / "data" / "propd_specificity_auc.json"


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged.

    Higher score must mean "more likely a binder", so callers pass -ΔG.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_ci(scores, labels, n=2000, seed=0):
    """Percentile CI on the AUC. n=1185 pairs is small enough that a point
    estimate alone would over-claim."""
    import random

    rng = random.Random(seed)
    idx = range(len(scores))
    vals = []
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        a = auc([scores[i] for i in sample], [labels[i] for i in sample])
        if a is not None:
            vals.append(a)
    if not vals:
        return None, None
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals))], 3)


def main() -> int:
    if not RESULTS.is_file():
        print(f"missing {RESULTS} — run e402 first")
        return 2

    rows = []
    seen = set()
    # Later rows win: E402 appends a retry rather than rewriting a failure.
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("delta_g") is None:
            continue
        seen.add((r["domain_id"], r["peptide"]))
        rows.append(r)
    dedup = {}
    for r in rows:
        dedup[(r["domain_id"], r["peptide"])] = r
    rows = list(dedup.values())

    n_pos = sum(r["label"] for r in rows)
    print("=== E403 — ProP-PD specificity AUC ===")
    print(f"  {len(rows)} scored pairs · {n_pos} positive · {len(rows) - n_pos} negative")
    if not rows or n_pos in (0, len(rows)):
        print("  not enough of both classes yet")
        return 1

    labels = [r["label"] for r in rows]

    # Discriminant 1 — raw dG (more negative = better binder, so negate).
    raw = [-r["delta_g"] for r in rows]
    a_raw = auc(raw, labels)
    lo_raw, hi_raw = bootstrap_ci(raw, labels)

    # Discriminant 2 — z-score within each domain.
    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain_id"]].append(r["delta_g"])
    stats = {}
    for d, vals in by_domain.items():
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        stats[d] = (mu, math.sqrt(var) or 1.0)
    z = [-((r["delta_g"] - stats[r["domain_id"]][0]) / stats[r["domain_id"]][1]) for r in rows]
    a_z = auc(z, labels)
    lo_z, hi_z = bootstrap_ci(z, labels)

    print()
    print("  discriminant                AUC     95% CI")
    print("  " + "-" * 46)
    print(f"  raw ΔG                     {a_raw:.3f}   [{lo_raw}, {hi_raw}]")
    print(f"  ΔG z-scored within domain  {a_z:.3f}   [{lo_z}, {hi_z}]   ← the fair one")
    print("  " + "-" * 46)
    print("  reference (CliPepPI paper, their ProP-PD split):")
    print("    CliPepPI 0.72 · AlphaFold2 actifpTM 0.90")

    # Per-domain AUC, so a single bad receptor is visible rather than averaged in.
    print("\n  per-domain (within-domain ranking):")
    per_domain = {}
    for d in sorted(by_domain):
        sub = [(i, r) for i, r in enumerate(rows) if r["domain_id"] == d]
        s = [-rows[i]["delta_g"] for i, _ in sub]
        l = [rows[i]["label"] for i, _ in sub]
        a = auc(s, l)
        per_domain[d] = {"auc": a, "n": len(sub), "n_pos": sum(l)}
        if a is not None:
            print(f"    {d[:44]:44s} n={len(sub):4d} pos={sum(l):3d}  AUC {a:.3f}")

    payload = {
        "n_pairs": len(rows),
        "n_positive": n_pos,
        "auc_raw_dg": a_raw,
        "auc_raw_dg_ci": [lo_raw, hi_raw],
        "auc_domain_zscore": a_z,
        "auc_domain_zscore_ci": [lo_z, hi_z],
        "per_domain": per_domain,
        "reference": {"clipeppi_propd": 0.72, "af2_actifptm_propd": 0.90},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
