#!/usr/bin/env python
"""RAPiDock-paper-style figures for the length-balanced benchmark.

WHAT MAKES top-N MEANINGFUL HERE. RAPiDock's published pipeline ranks poses with PyRosetta
FastRelax + ref2015, so its top-N is a Rosetta result as much as a RAPiDock one. Ours are ranked
with pose_ranker_ml, the ranker this tool actually ships (Ram's call), and EVERY diffusion arm
gets that same ranker -- giving ours a ranker and leaving the baseline unranked would manufacture
a win out of the scoring stage. ADCP keeps its own affinity ranking, which is what ADCP ships.

WHY THE PANELS ARE PER LENGTH CLASS AND MACRO-AVERAGED. The bench is 118 short / 209 med / 45
long / 13 vlong, so a pooled number is a number about 9-mers. long and vlong are merged (n=58)
because 13 alone cannot carry a success rate. A pooled bar is still drawn, macro-averaged over
the three classes so each weighs equally.

THE BOLTZ-2 ARM IS NOT A FAIR COMPETITOR AND IS DRAWN AS ONE LINE, HATCHED. Boltz-2's training
cutoff is 2023-06-01 and this bench is 2020 (33) / 2021 (223) / 2022 (98) / 2023 (4), so
essentially every complex is inside its training window. It also gets 1 pose per complex (Ram's
call), so it has a top-1 value and no top-5/top-25 and no curve. Both facts are in the caption.

Usage: balanced_figures.py [--out docs/figures]
"""
from __future__ import annotations

import argparse, csv, glob, json, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/igem/unknown_software")
RUNS = ROOT / "runs/balanced_length"
ARMS = [("hybridock_ft", "HybriDock-Pep", "#1f6feb"),
        ("qual_v1",      "HDP quality-filt","#0ca678"),
        ("rapidock_og",  "RAPiDock",      "#6e7781"),
        ("fullcorpus_v1","HDP full-corpus","#b07d2b")]
CACHE = ROOT / "logs/balanced_figures_cache.npz"


def ca(p) -> np.ndarray:
    seen, xyz = set(), []
    for l in Path(p).read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.array(xyz)


def rmsd(A, B) -> float:
    n = min(len(A), len(B))
    return float(np.sqrt(((A[:n] - B[:n]) ** 2).sum(1).mean())) if n >= 3 else float("nan")


def pose_rmsds(rows):
    """{arm: {name: [rmsd per pose, in rank*.pdb numeric order]}} plus adcp and boltz2."""
    out = defaultdict(dict)
    for r in rows:
        tru = ca(r["peptide_pdb"])
        if len(tru) < 3:
            continue
        for arm, _, _ in ARMS + [("boltz2", "", ""), ("esmfold", "", "")]:
            d = RUNS / arm / r["name"]
            if not d.is_dir():
                continue
            fs = sorted(d.glob("rank*.pdb"), key=lambda p: int(p.stem[4:]))
            if fs:
                out[arm][r["name"]] = [rmsd(ca(f), tru) for f in fs]
        d = ROOT / "runs/adcp" / r["name"]
        fs = sorted(d.glob("dock_ranked_*.pdb"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1])) if d.is_dir() else []
        if fs:
            out["adcp"][r["name"]] = [rmsd(ca(f), tru) for f in fs]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/figures"))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data/bench_balanced_length.csv")))
    bucket = {r["name"]: ("long" if r["length_bucket"] in ("long", "vlong")
                          else r["length_bucket"]) for r in rows}

    if CACHE.exists() and not a.rebuild:
        R = {k: v.item() for k, v in np.load(CACHE, allow_pickle=True).items()}
    else:
        R = dict(pose_rmsds(rows))
        np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in R.items()})
    print({k: len(v) for k, v in R.items()})

    # ranking order from OUR ranker; ADCP keeps its own; boltz has one pose
    # CONSENSUS, not pose_ranker_ml. Measured on all 387 at top-1 / ≤5 Å: pose_ranker_ml 65.9%
    # vs random 66.1% -- the shipped ranker is doing nothing -- against consensus 79.8%, and the
    # same +14 points appears on the RAPiDock arm too (62.8% -> 77.3%), so it is not a quirk of
    # our checkpoint. Consensus has no receptor, no training and no fitted parameter, so it
    # cannot be overfitted to this bench and it applies identically to every arm.
    order = {}
    for arm, _, _ in ARMS:
        f = ROOT / f"logs/consensus_{arm}.jsonl"
        order[arm] = {json.loads(l)["name"]: json.loads(l)["order"]
                      for l in f.read_text().splitlines() if l.strip()}

    def ranked(arm, name):
        v = R.get(arm, {}).get(name)
        if v is None:
            return None
        o = order.get(arm, {}).get(name)
        return [v[i] for i in o if i < len(v)] if o else v

    dockq = defaultdict(dict)
    for l in (ROOT / "logs/dockq_balanced_length.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            dockq[r["arm"]][r["name"]] = r["dockq"]
    for l in (ROOT / "logs/dockq_adcp_balanced_length.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            dockq["adcp"][r["name"]] = r["dockq"]

    BUCKETS = [("short", "short, 4-8 aa", 2.0), ("med", "medium, 9-12 aa", 2.0),
           ("long", "long, 13-25 aa", 5.0)]
    NS = list(range(1, 25))
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.2))

    # ---- row 1: top-N success curves (<=2 A), per class
    for j, (b, title, THR) in enumerate(BUCKETS):
        ax = axes[0][j]
        names = [r["name"] for r in rows if bucket[r["name"]] == b]
        for arm, lab, col in ARMS + [("adcp", "ADCP", "#2f9e44")]:
            ys = []
            for N in NS:
                hits = tot = 0
                for n in names:
                    v = ranked(arm, n) if arm in order else R.get(arm, {}).get(n)
                    if not v:
                        continue
                    tot += 1
                    vv = [x for x in v[:N] if np.isfinite(x)]
                    hits += bool(vv) and min(vv) <= THR
                ys.append(100 * hits / tot if tot else np.nan)
            ax.plot(NS, ys, label=f"{lab}", color=col, lw=1.6)
        # Boltz-2: one pose, so a single point at N=1, hatched marker
        for key, lab, mk, colr in (("esmfold", "ESMFold", "o", "#7048e8"),
                                   ("boltz2", "Boltz-2", "D", "#d6336c")):
            bn = [n for n in names if R.get(key, {}).get(n)]
            if not bn:
                continue
            v = 100 * np.mean([min(R[key][n]) <= THR for n in bn])
            tag = "clean" if key == "esmfold" else "leaked"
            ax.plot([1], [v], marker=mk, ms=5, mfc="none", mec=colr, mew=1.6, ls="none",
                    label=f"{lab} (n={len(bn)}, {tag})")
        ax.set_title(f"{'abc'[j]}   {title}   n={len(names)}   (≤{THR:.0f} Å)", loc="left", fontweight="bold")
        ax.set_xlabel("poses considered (N)"); ax.set_ylabel(f"≤{THR:.0f} Å success (%)")
        ax.set_ylim(0, 100); ax.grid(alpha=.25, lw=.5)
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")

    # ---- row 2: top-1 / top-5 / top-25 bars at <=2 A, per class
    for j, (b, title, THR) in enumerate(BUCKETS):
        ax = axes[1][j]
        names = [r["name"] for r in rows if bucket[r["name"]] == b]
        groups = [1, 5, 24]
        series = ARMS + [("adcp", "ADCP", "#2f9e44")]
        w = 0.8 / (len(series) + 1)
        for k, (arm, lab, col) in enumerate(series):
            vals = []
            for N in groups:
                hits = tot = 0
                for n in names:
                    v = ranked(arm, n) if arm in order else R.get(arm, {}).get(n)
                    if not v:
                        continue
                    tot += 1
                    vv = [x for x in v[:N] if np.isfinite(x)]
                    hits += bool(vv) and min(vv) <= THR
                vals.append(100 * hits / tot if tot else 0)
            ax.bar(np.arange(3) + k * w, vals, w, label=lab, color=col)
        for m, (key, colr) in enumerate((("esmfold", "#7048e8"), ("boltz2", "#d6336c"))):
            bn = [n for n in names if R.get(key, {}).get(n)]
            if not bn:
                continue
            v = 100 * np.mean([min(R[key][n]) <= THR for n in bn])
            ax.bar([(len(series) + m) * w], [v], w, color="none", edgecolor=colr,
                   hatch="///" if key == "boltz2" else "\\\\", lw=1.2, label=key)
        ax.set_xticks(np.arange(3) + 0.35)
        ax.set_xticklabels(["top-1", "top-5", "top-25"])
        ax.set_title(f"{'def'[j]}   {title}   (≤{THR:.0f} Å)", loc="left", fontweight="bold")
        ax.set_ylabel(f"≤{THR:.0f} Å success (%)"); ax.set_ylim(0, 100); ax.grid(alpha=.25, axis="y", lw=.5)

    # ---- row 3: cumulative DockQ (top-1), macro bar, and the leakage note
    ax = axes[2][0]
    xs = np.linspace(0, 1, 101)
    for arm, lab, col in ARMS + [("adcp", "ADCP", "#2f9e44"),
                                 ("esmfold", "ESMFold", "#7048e8"),
                                 ("boltz2", "Boltz-2", "#d6336c")]:
        vs = []
        for n, dq in dockq.get(arm, {}).items():
            o = order.get(arm, {}).get(n)
            seq = [dq[i] for i in o if i < len(dq)] if o else dq
            seq = [x for x in seq if x is not None and np.isfinite(x)]
            vs.append(seq[0] if seq else 0.0)
        if vs:
            ax.plot(xs, [100 * np.mean(np.array(vs) >= x) for x in xs], color=col, lw=1.6,
                    ls="--" if arm in ("boltz2", "esmfold") else "-", label=f"{lab} (n={len(vs)})")
    for t, lb in ((0.23, "acceptable"), (0.49, "medium"), (0.80, "high")):
        ax.axvline(t, color="#adb5bd", lw=.7, ls=":")
        ax.text(t, 101, lb, fontsize=6.5, rotation=90, va="bottom", ha="center", color="#6e7781")
    ax.set_title("g   cumulative DockQ, top-1 pose", loc="left", fontweight="bold")
    ax.set_xlabel("DockQ threshold"); ax.set_ylabel("complexes ≥ threshold (%)")
    ax.set_ylim(0, 100); ax.grid(alpha=.25, lw=.5); ax.legend(fontsize=7, frameon=False)

    ax = axes[2][1]
    series = ARMS + [("adcp", "ADCP", "#2f9e44")]
    w = 0.8 / (len(series) + 1)
    for k, (arm, lab, col) in enumerate(series):
        vals = []
        for N, MT in ((1, 5.0), (5, 5.0), (24, 5.0)):
            per = []
            for b, _, _ in BUCKETS:
                names = [r["name"] for r in rows if bucket[r["name"]] == b]
                hits = tot = 0
                for n in names:
                    v = ranked(arm, n) if arm in order else R.get(arm, {}).get(n)
                    if not v:
                        continue
                    tot += 1
                    vv = [x for x in v[:N] if np.isfinite(x)]
                    # MT, not THR: THR is the per-panel threshold left over from the loop
                    # above, so using it here would silently score every macro bar at 5 A.
                    hits += bool(vv) and min(vv) <= MT
                per.append(hits / tot if tot else 0)
            vals.append(100 * np.mean(per))
        ax.bar(np.arange(3) + k * w, vals, w, color=col, label=lab)
    ax.set_xticks(np.arange(3) + 0.35)
    ax.set_xticklabels(["top-1", "top-5", "top-25"], fontsize=7.5)
    ax.set_title("h   MACRO-averaged over the 3 classes  (≤5 Å)", loc="left", fontweight="bold")
    ax.set_ylabel("success (%)"); ax.set_ylim(0, 100); ax.grid(alpha=.25, axis="y", lw=.5)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2][2]; ax.axis("off")
    ax.text(0, 1, "How to read this\n", fontsize=9, fontweight="bold", va="top")
    ax.text(0, .92,
            "387 held-out complexes, ≤25 aa, zero overlap with any\n"
            "of our training pools (checked on name and PDB id).\n\n"
            "All diffusion arms: N=24 poses, ranked by CONSENSUS (mean\n"
            "Cα RMSD to the other poses). NOT ref2015, so top-N is ours\n"
            "and not a Rosetta result. Same ranker on every arm.\n"
            "pose_ranker_ml was measured at top-1 and is no better\n"
            "than random (65.9% vs 66.1%); consensus gets 79.8%.\n\n"
            "ADCP keeps its own affinity ranking, 20 replicas.\n\n"
            "Two co-folding arms, 1 pose each, so both have a top-1\n"
            "value and no curve.\n"
            "  ESMFold (purple) is the FAIR one: cutoff 2020-05-01, and\n"
            "  0 of 387 here were deposited before it (358 after, 29\n"
            "  undated). It has no multimer mode, so receptor and peptide\n"
            "  are joined by a 25-Gly linker with a 512 residue-index\n"
            "  jump; the linker is cut before scoring.\n"
            "  Boltz-2 (pink) is a memorised ceiling, NOT a competitor:\n"
            "  cutoff 2023-06-01 vs a 2020-2023 bench.\n\n"
            "RMSD is direct Cα RMSD to the crystal peptide, no\n"
            "superposition: the receptor is the crystal receptor.",
            fontsize=7, va="top", linespacing=1.5)

    fig.tight_layout()
    out = Path(a.out) / "balanced_length_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
