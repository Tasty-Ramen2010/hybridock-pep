#!/usr/bin/env python3
"""
held_out_rmsd_eval.py — Honest held-out RMSD evaluation via 5-fold CV.

Same fold splits as burial_analysis.py (RandomState(7), 5 folds).
For each fold: train encoder head on train complexes, evaluate on held-out val.
Reports top-1 RMSD, Hit@2Å, Hit@5Å for four strategies:

    rapd_top1   — RAPiDock's own top prediction (pose_0, no reranking)
    ref2015     — naive total_score ranking (lower = better)
    rankerv2    — burial z-blend + encoder head (no consensus: adds <0.5% τ)
    oracle      — always picks the lowest-RMSD pose

All numbers are averaged over held-out complexes only.
"""
from __future__ import annotations

import json
import math
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
D    = REPO / "logs" / "diagnosis"
OUT  = REPO / "logs" / "training_campaign"

BJSON = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))
BURIAL_IDX  = [0, 2, 1]
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])
ENC_DIM = 96
EPOCHS  = 50
SEEDS   = (0, 1, 2)
FOLDS   = 5

torch.set_num_threads(4)


def _per_cx_z(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0)
    return (M - mu) / np.where(sd < 1e-9, 1.0, sd)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


# ── load pool ────────────────────────────────────────────────────────────────

def load_pool() -> dict:
    phys = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    enc  = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    pool: dict = {}
    for k, pv in phys.items():
        cn, mk, pi = k
        if k not in enc:
            continue
        rr = BJSON.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rr):
            continue
        pool.setdefault(cn, []).append({
            "phys": np.asarray(pv, np.float64),
            "enc":  np.asarray(enc[k], np.float64),
            "rmsd": float(rr[pi]),
            "pi":   pi,
        })
    return {c: v for c, v in pool.items() if len(v) >= 3}


# ── encoder head ─────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, d: int = ENC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


def _train_head(pool: dict, cxs: list, seed: int) -> Head:
    pairs = []
    for c in cxs:
        rows = pool[c]
        E  = _per_cx_z(np.array([r["enc"] for r in rows]))
        rr = np.array([r["rmsd"] for r in rows])
        for i, j in combinations(range(len(rr)), 2):
            if abs(rr[i] - rr[j]) > 1e-6:
                pairs.append((E[i], E[j], 1.0 if rr[i] < rr[j] else 0.0))
    torch.manual_seed(seed)
    h = Head()
    opt = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
    fi = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lb = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    n  = len(pairs)
    for _ in range(EPOCHS):
        h.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b + 512]
            loss = -F.logsigmoid(
                (h(fi[idx]).squeeze(-1) - h(fj[idx]).squeeze(-1))
                * (lb[idx] * 2 - 1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return h.eval()


def _score_enc(h: Head, enc_normed: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return h(torch.tensor(enc_normed, dtype=torch.float32)).squeeze(-1).numpy()


# ── per-complex ranking ───────────────────────────────────────────────────────

def rank_complex(rows: list, head: Head) -> dict[str, float]:
    """Return top-1 RMSD for each strategy on one complex."""
    rmsd = np.array([r["rmsd"] for r in rows])

    # oracle
    oracle_rmsd = float(rmsd.min())

    # RAPiDock top-1 = pose_0
    pose0_idx = next((i for i, r in enumerate(rows) if r["pi"] == 0), 0)
    rapd_rmsd = float(rmsd[pose0_idx])

    # naive ref2015: total_score (idx 13), lower = better
    total = np.array([r["phys"][13] for r in rows])
    ref_rmsd = float(rmsd[np.argmin(total)])

    # RankerV2: burial z-blend + encoder head
    phys_arr = np.array([r["phys"] for r in rows])
    enc_arr  = np.array([r["enc"]  for r in rows])
    burial   = _per_cx_z(phys_arr[:, BURIAL_IDX]) @ BURIAL_SIGN
    enc_s    = _score_enc(head, _per_cx_z(enc_arr))
    score    = _z(-burial) + _z(enc_s)
    rv2_rmsd = float(rmsd[np.argmax(score)])

    # best-of-top-5 for ref2015 and rankerv2
    ref_top5  = float(rmsd[np.argsort(total)[:5]].min())
    rv2_top5  = float(rmsd[np.argsort(-score)[:5]].min())
    oracle_5  = float(np.sort(rmsd)[:5].min())  # = oracle since poses >= 5

    return {
        "rapd_top1": rapd_rmsd,
        "ref2015":   ref_rmsd,
        "rankerv2":  rv2_rmsd,
        "oracle":    oracle_rmsd,
        "ref2015_top5": ref_top5,
        "rankerv2_top5": rv2_top5,
        "oracle_top5": oracle_5,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading pool...")
    pool = load_pool()
    cxs  = sorted(pool)
    print(f"  {len(pool)} complexes, {sum(len(v) for v in pool.values())} poses")

    rng  = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    folds = [[cxs[i] for i in perm[f::FOLDS]] for f in range(FOLDS)]

    # cx_results[cx][strategy] = mean RMSD over seeds (one entry per complex)
    cx_results: dict[str, dict[str, float]] = {}
    fold_taus: dict[str, list[float]] = {"ref2015": [], "rankerv2": []}

    for fi in range(FOLDS):
        val_cxs = folds[fi]
        tr_cxs  = [c for c in cxs if c not in set(val_cxs)]
        print(f"\nFold {fi+1}/5  (train={len(tr_cxs)} val={len(val_cxs)})", flush=True)

        # accumulate per-complex per-seed results
        cx_seed: dict[str, dict[str, list]] = {c: {} for c in val_cxs}
        seed_tau_ref, seed_tau_rv2 = [], []

        for sd in SEEDS:
            head = _train_head(pool, tr_cxs, sd)
            tau_ref_vals, tau_rv2_vals = [], []

            for c in val_cxs:
                rows = pool[c]
                r = rank_complex(rows, head)
                for k, v in r.items():
                    cx_seed[c].setdefault(k, []).append(v)

                rmsd = np.array([row["rmsd"] for row in rows])
                total = np.array([row["phys"][13] for row in rows])
                t, _ = sp.kendalltau(total, rmsd)
                if not math.isnan(t): tau_ref_vals.append(t)

                phys_arr = np.array([row["phys"] for row in rows])
                enc_arr  = np.array([row["enc"]  for row in rows])
                burial = _per_cx_z(phys_arr[:, BURIAL_IDX]) @ BURIAL_SIGN
                enc_s  = _score_enc(head, _per_cx_z(enc_arr))
                score  = _z(-burial) + _z(enc_s)
                t, _   = sp.kendalltau(-score, rmsd)
                if not math.isnan(t): tau_rv2_vals.append(t)

            seed_tau_ref.append(float(np.mean(tau_ref_vals)))
            seed_tau_rv2.append(float(np.mean(tau_rv2_vals)))

        # average over seeds → one value per complex
        for c in val_cxs:
            cx_results[c] = {k: float(np.mean(vs)) for k, vs in cx_seed[c].items()}

        fold_taus["ref2015"].append(float(np.mean(seed_tau_ref)))
        fold_taus["rankerv2"].append(float(np.mean(seed_tau_rv2)))
        print(f"  ref2015 τ={fold_taus['ref2015'][-1]:+.4f}  rankerv2 τ={fold_taus['rankerv2'][-1]:+.4f}")

    # flatten: each complex appears exactly once (from its held-out fold)
    all_results: dict[str, list[float]] = {}
    for rd in cx_results.values():
        for k, v in rd.items():
            all_results.setdefault(k, []).append(v)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"HELD-OUT EVALUATION  (5-fold CV, {len(pool)} complexes, {len(SEEDS)} seeds/fold)")
    print(f"{'='*70}")

    strategies_top1 = ["rapd_top1", "ref2015", "rankerv2", "oracle"]
    print(f"\nTop-1 pose selection:")
    print(f"  {'Strategy':<20} {'Mean RMSD':>10} {'Median':>8} {'Hit@2Å':>8} {'Hit@5Å':>8}")
    print(f"  {'-'*56}")

    final: dict = {}
    for s in strategies_top1:
        vals = np.array(all_results[s])
        m    = float(vals.mean())
        med  = float(np.median(vals))
        h2   = float(np.mean(vals <= 2.0)) * 100
        h5   = float(np.mean(vals <= 5.0)) * 100
        final[s] = {"mean": m, "median": med, "hit2": h2, "hit5": h5}
        print(f"  {s:<20} {m:>8.2f} Å   {med:>6.2f}   {h2:>6.1f}%   {h5:>6.1f}%")

    print(f"\nBest-of-top-5 (send top 5 to experiment / next stage):")
    print(f"  {'Strategy':<20} {'Mean RMSD':>10} {'Median':>8} {'Hit@2Å':>8}")
    print(f"  {'-'*48}")
    for s in ["ref2015_top5", "rankerv2_top5", "oracle_top5"]:
        vals = np.array(all_results[s])
        m    = float(vals.mean())
        med  = float(np.median(vals))
        h2   = float(np.mean(vals <= 2.0)) * 100
        final[s] = {"mean": m, "median": med, "hit2": h2}
        label = s.replace("_top5", " top-5").replace("oracle", "oracle   ")
        print(f"  {label:<20} {m:>8.2f} Å   {med:>6.2f}   {h2:>6.1f}%")

    print(f"\nKendall τ (mean ± std across folds):")
    for s, tvals in fold_taus.items():
        print(f"  {s:<20} τ = {np.mean(tvals):+.4f} ± {np.std(tvals):.4f}")

    final["_meta"] = {
        "n_complexes": len(pool), "folds": FOLDS, "seeds": list(SEEDS),
        "epochs": EPOCHS, "note": "fully held-out: encoder head trained on fold train only",
    }
    (OUT / "held_out_rmsd_eval.json").write_text(json.dumps(final, indent=2))
    print(f"\nSaved → {OUT}/held_out_rmsd_eval.json")


if __name__ == "__main__":
    main()
