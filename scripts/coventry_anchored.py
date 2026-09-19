#!/usr/bin/env python
"""Our own scorer, in the regime we actually validated it for: known-target anchoring.

THE REGIME, AND WHY IT IS OURS.  E221/E222 found the hidden variable behind our absolute-Kd
ceiling: ~75% of affinity variance is a per-RECEPTOR baseline -- how tightly a pocket binds
peptides in general, peptide-blind.  Receptor-mean alone predicts dG at r=0.578, better than our
full 262-feature model at 0.292, and no static representation recovers it (pocket composition
0.049, ProtDCal 0.149, ESM-2 0.154 -- a hard wall around 0.15).

That is why absolute cross-receptor Kd is bounded for everyone, us and PPI-Affinity alike.  But
it also says exactly when the wall disappears: **give the receptor one measured binder and the
baseline cancels.**  In the known-target regime E222 measured r=0.686 against PPI's 0.55 -- the
model learns each receptor's baseline from its other peptides.  Anchoring (e260) is the same
statement from the other side: cold cross-receptor r=-0.07 becomes +0.71 once per-receptor
offsets are measured rather than predicted, and a shuffle control with anchors from the WRONG
receptor collapses to -0.05, so it is genuine offset cancellation and not leakage.

THE COVENTRY GRID HANDS US EXACTLY ONE ANCHOR PER BINDER.  Wu et al. report a Kd for all 18
cognate pairs.  So for binder B we can measure its offset instead of predicting it:

    b(B) = S(cognate_B, B) - dG_exp(cognate_B, B)
    anchored(P, B) = S(P, B) - b(B)

and then ask the question the paper is actually about: **given each binder's known cognate,
which OTHER peptides also bind it?**  That is 12 measured cross-reactivities hiding among 294
non-binders.

THE DIAGONAL IS THE ANCHOR, SO THE DIAGONAL IS EXCLUDED FROM EVERY ANCHORED METRIC.  Scoring the
cell you calibrated on is circular. Every number below is computed on the 306 off-diagonal cells
only, and the unanchored baselines are recomputed on the same 306 so the comparison is fair.

Usage: coventry_anchored.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
RT = 0.0019872041 * 298.0


def load(path: str, field: str = "best_iface") -> dict:
    M = {}
    for line in (ROOT / path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "name" in r:
                p, b = r["name"].split("__")
                b = b[:-1] if b.endswith("B") else b
            else:
                p, b = r["peptide"], r["binder"]
            v = r.get(field)
            if v is not None:
                M[(p, b)] = v
    return M


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(1.0 if x < y else 0.5 if x == y else 0.0
               for x in pos for y in neg) / (len(pos) * len(neg))


def centre(M: dict) -> dict:
    peps = [p for p in ORDER if all((p, b) in M for b in ORDER)]
    rm = {p: st.mean([M[(p, b)] for b in ORDER]) for p in peps}
    cm = {b: st.mean([M[(p, b)] for p in peps]) for b in ORDER}
    g = st.mean([M[(p, b)] for p in peps for b in ORDER])
    return {(p, b): M[(p, b)] - rm[p] - cm[b] + g for p in peps for b in ORDER}


def main() -> None:
    truth = {(r["peptide"], r["binder_label"][:-1]): r
             for r in csv.DictReader(open(ROOT / "data/coventry_pairs.csv"))}
    dG_exp = {}
    for k, r in truth.items():
        if r["measured"] == "1" and r.get("kd_M"):
            try:
                dG_exp[k] = RT * math.log(float(r["kd_M"]))
            except (TypeError, ValueError):
                pass

    dg = load("logs/coventry_dg.jsonl")
    ref = load("logs/coventry_refine_fixed.jsonl")
    # our scorer on poses that came through Stage 1.4 co-folding -- same scorer, same repack,
    # same anchoring; the only difference is that the peptide is in the right place
    cof = {}
    for line in (ROOT / "logs/coventry_boltz_refine.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            v = r.get("best_iface")
            cof[(r["peptide"], r["binder"])] = 50.0 if (v is None or v > 50) else v

    # ---- 1. MAE on the diagonal, our calibrated dG against experiment ---------
    print("=" * 78)
    print("OUR CALIBRATED dG ON THE 18 COGNATE CELLS (absolute accuracy)")
    print("=" * 78)
    rows = [(p, dg[(p, p)], dG_exp[(p, p)]) for p in ORDER
            if (p, p) in dg and (p, p) in dG_exp]
    print(f"{'pair':>8}{'ours':>10}{'experiment':>12}{'error':>9}   Kd")
    for p, pred, obs in rows:
        print(f"{p:>8}{pred:10.2f}{obs:12.2f}{pred - obs:+9.2f}   "
              f"{truth[(p, p)]['kd_M']}")
    err = [a - b for _, a, b in rows]
    mae = st.mean([abs(e) for e in err])
    rmse = math.sqrt(st.mean([e * e for e in err]))
    print(f"\n  n = {len(rows)}   MAE {mae:.2f} kcal/mol   RMSE {rmse:.2f}   "
          f"bias {st.mean(err):+.2f}")
    print(f"  experiment spans {min(b for _, _, b in rows):.2f} to "
          f"{max(b for _, _, b in rows):.2f} kcal/mol (sd {st.pstdev([b for _, _, b in rows]):.2f})")
    print(f"  ours spans       {min(a for _, a, _ in rows):.2f} to "
          f"{max(a for _, a, _ in rows):.2f} kcal/mol (sd {st.pstdev([a for _, a, _ in rows]):.2f})")
    ma = st.mean([b for _, _, b in rows])
    print(f"  MAE of just predicting the mean ({ma:.2f}): "
          f"{st.mean([abs(b - ma) for _, _, b in rows]):.2f} kcal/mol")

    # ---- 2. anchored cross-reactivity, diagonal excluded ---------------------
    print("\n" + "=" * 78)
    print("ANCHORED CROSS-REACTIVITY — each binder's cognate Kd is the anchor")
    print("the 18 anchor cells are EXCLUDED from every number below")
    print("=" * 78)
    off = [(p, b) for p in ORDER for b in ORDER if p != b]
    pos = [k for k in off if truth.get(k, {}).get("measured") == "1"]
    neg = [k for k in off if truth.get(k, {}).get("measured") == "0"]
    print(f"{len(pos)} measured cross-reactivities among {len(neg)} non-binders\n")
    print(f"{'estimator':<40}{'AUC':>8}{'top-12 hits':>13}{'best rank':>11}")

    for label, M in (("our calibrated dG", dg), ("ref2015 interface", ref),
                     ("our scorer on CO-FOLDED poses", cof)):
        if not M:
            continue
        variants = {
            "raw": M,
            "double-centred (unsupervised)": centre(M),
        }
        anchors = {b: M[(b, b)] - dG_exp[(b, b)] for b in ORDER
                   if (b, b) in M and (b, b) in dG_exp}
        if len(anchors) >= 15:
            variants["ANCHORED on cognate Kd"] = {
                k: M[k] - anchors[k[1]] for k in M if k[1] in anchors}
        for vname, V in variants.items():
            have_pos = [V[k] for k in pos if k in V]
            have_neg = [V[k] for k in neg if k in V]
            if len(have_pos) < 8:
                continue
            a = auc(have_pos, have_neg)
            ranked = sorted([k for k in off if k in V], key=lambda k: V[k])
            hits = sum(1 for k in ranked[:12] if k in pos)
            best = min((i + 1 for i, k in enumerate(ranked) if k in pos), default=0)
            print(f"  {label} · {vname:<24}{a:>8.3f}{hits:>10d}/12{best:>11d}")

    # ---- 3. shuffle control: anchors from the WRONG binder -------------------
    import random
    rng = random.Random(0)
    print("\nSHUFFLE CONTROL (anchor taken from a different binder):")
    for label, M in (("our calibrated dG", dg), ("ref2015 interface", ref),
                     ("our scorer on CO-FOLDED poses", cof)):
        anchors = {b: M[(b, b)] - dG_exp[(b, b)] for b in ORDER
                   if (b, b) in M and (b, b) in dG_exp}
        if len(anchors) < 15:
            continue
        aucs = []
        keys = list(anchors)
        for _ in range(20):
            perm = keys[:]
            rng.shuffle(perm)
            sw = {k: anchors[v] for k, v in zip(keys, perm)}
            V = {k: M[k] - sw[k[1]] for k in M if k[1] in sw}
            aucs.append(auc([V[k] for k in pos if k in V], [V[k] for k in neg if k in V]))
        print(f"  {label:<24} AUC {st.mean(aucs):.3f} ± {st.pstdev(aucs):.3f}")


if __name__ == "__main__":
    main()
