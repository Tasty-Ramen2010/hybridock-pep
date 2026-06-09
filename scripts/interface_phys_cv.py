#!/usr/bin/env python3
"""
interface_phys_cv.py — CV comparison of interface-specific physics features.

Tests Route A (interface-residue summed energies) and Route B (cross-chain
pairwise interaction energies) against the global burial baseline.
Also tests combinations with the encoder stream.

Conditions evaluated:
  global_burial      — existing physslim (burial axis), z-blend with encoder
  route_a_slim       — burial axis from Route A features only
  route_b_slim       — burial axis from Route B features only
  route_a_enc        — Route A burial + encoder z-blend
  route_b_enc        — Route B burial + encoder z-blend
  both_enc           — Route A burial + Route B burial + encoder z-blend

All use 5-fold CV, 3 seeds, 50 epochs BPR, held-out complexes.
Restricted to the 115-cx physics set (complexes with both phys + interface_phys features).

Output: logs/training_campaign/interface_phys_cv.json
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
D = REPO / "logs" / "diagnosis"
OUT = REPO / "logs" / "training_campaign"

BURIAL_IDX = [0, 2, 1]      # fa_atr, fa_sol, fa_rep
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])
ENC_DIM = 96

torch.set_num_threads(4)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _per_cx_z(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0)
    return (M - mu) / np.where(sd < 1e-9, 1.0, sd)


def _burial(phys_3col: np.ndarray) -> np.ndarray:
    """Signed burial coordinate from 3-column [fa_atr, fa_sol, fa_rep] matrix."""
    return _per_cx_z(phys_3col) @ BURIAL_SIGN


# ── load pool ────────────────────────────────────────────────────────────────

def load_pool() -> dict:
    phys = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    enc = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    iface = pickle.load(open(D / "feats_bench300_interface_phys.pkl", "rb"))
    bjson = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))

    pool: dict = {}
    n_miss = 0
    for k, pv in phys.items():
        cn, mk, pi = k
        if k not in enc:
            n_miss += 1; continue
        if iface.get(k) is None:
            n_miss += 1; continue
        rmsds = bjson.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rmsds):
            n_miss += 1; continue

        ientry = iface[k]
        pool.setdefault(cn, []).append({
            "phys":    np.asarray(pv, np.float64),
            "enc":     np.asarray(enc[k], np.float64),
            "route_a": np.asarray(ientry["route_a"], np.float64),
            "route_b": np.asarray(ientry["route_b"], np.float64),
            "rmsd":    float(rmsds[pi]),
        })

    pool = {c: v for c, v in pool.items() if len(v) >= 3}
    print(f"  {len(pool)} complexes, {n_miss} entries skipped")
    return pool


# ── MLP head ─────────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


def _train(pairs, d, seed, epochs=50):
    torch.manual_seed(seed)
    h = Head(d)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
    fi = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lb = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    n = len(pairs)
    for _ in range(epochs):
        h.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b + 512]
            loss = -F.logsigmoid(
                (h(fi[idx]).squeeze(-1) - h(fj[idx]).squeeze(-1))
                * (lb[idx] * 2 - 1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return h.eval()


def _score_h(h, F):
    with torch.no_grad():
        return h(torch.tensor(F, dtype=torch.float32)).squeeze(-1).numpy()


# ── CV engine ────────────────────────────────────────────────────────────────

def _make_pairs_and_feats(pool, tr_cxs, stream_fn):
    """Build BPR pairs for a single stream, return (pairs, val_feats_fn)."""
    pairs = []
    for c in tr_cxs:
        rows = pool[c]
        F = _per_cx_z(stream_fn(rows))
        rr = np.array([r["rmsd"] for r in rows])
        for i, j in combinations(range(len(rr)), 2):
            if abs(rr[i] - rr[j]) > 1e-6:
                pairs.append((F[i], F[j], 1.0 if rr[i] < rr[j] else 0.0))
    return pairs


def cv_streams(pool, stream_specs, seeds=(0, 1, 2), folds=5, epochs=50):
    """
    5-fold CV for a z-blend of independent streams.

    stream_specs: list of (name, feat_fn)
        feat_fn(rows) -> (n, d) raw feature matrix for one complex's rows.
        Each stream gets its own Head; outputs are z-scored and summed.
    """
    cxs = sorted(pool)
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    fold_groups = [[cxs[i] for i in perm[f::folds]] for f in range(folds)]

    fold_taus = []
    for fi in range(folds):
        val_cxs = fold_groups[fi]
        tr_cxs = [c for c in cxs if c not in set(val_cxs)]

        seed_taus = []
        for sd in seeds:
            heads = {}
            for sname, sfn in stream_specs:
                pairs = _make_pairs_and_feats(pool, tr_cxs, sfn)
                d = sfn(pool[tr_cxs[0]]).shape[1]
                heads[sname] = _train(pairs, d, sd, epochs)

            ts = []
            for c in val_cxs:
                rows = pool[c]
                rr = np.array([r["rmsd"] for r in rows])
                score = np.zeros(len(rows))
                for sname, sfn in stream_specs:
                    F = _per_cx_z(sfn(rows))
                    s = _score_h(heads[sname], F)
                    score = score + _z(s)
                t, _ = sp.kendalltau(-score, rr)
                if not math.isnan(t):
                    ts.append(t)
            seed_taus.append(float(np.mean(ts)) if ts else float("nan"))

        fold_taus.append(float(np.nanmean(seed_taus)))

    return float(np.mean(fold_taus)), float(np.std(fold_taus))


# ── feature functions ────────────────────────────────────────────────────────

def global_burial(rows):
    return np.array([r["phys"][BURIAL_IDX] * BURIAL_SIGN for r in rows])


def route_a_burial(rows):
    # same burial axis applied to Route A features [fa_atr, fa_sol, fa_rep]
    return np.array([r["route_a"][BURIAL_IDX] * BURIAL_SIGN for r in rows])


def route_b_burial(rows):
    # burial axis from cross-chain interactions
    return np.array([r["route_b"][BURIAL_IDX] * BURIAL_SIGN for r in rows])


def route_a_full(rows):
    return np.array([r["route_a"] for r in rows])


def route_b_full(rows):
    return np.array([r["route_b"] for r in rows])


def encoder(rows):
    return np.array([r["enc"] for r in rows])


def both_burial(rows):
    # concatenate Route A and B burial coordinates as a 2-col input
    a = np.array([r["route_a"][BURIAL_IDX] * BURIAL_SIGN for r in rows])   # (n,3)
    b = np.array([r["route_b"][BURIAL_IDX] * BURIAL_SIGN for r in rows])   # (n,3)
    return np.concatenate([a, b], axis=1)                                    # (n,6)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading pools...")
    pool = load_pool()

    configs = [
        ("global_burial + enc (baseline)",    [("burial", global_burial),  ("enc", encoder)]),
        ("route_A_burial + enc",              [("burial", route_a_burial), ("enc", encoder)]),
        ("route_B_burial + enc",              [("burial", route_b_burial), ("enc", encoder)]),
        ("route_A_full + enc",                [("phys",   route_a_full),   ("enc", encoder)]),
        ("route_B_full + enc",                [("phys",   route_b_full),   ("enc", encoder)]),
        ("A_burial + B_burial + enc",         [("a",      route_a_burial), ("b", route_b_burial), ("enc", encoder)]),
        ("A_full + B_full + enc",             [("a",      route_a_full),  ("b", route_b_full),  ("enc", encoder)]),
        ("global + A_burial + B_burial + enc",[("g",      global_burial),  ("a", route_a_burial), ("b", route_b_burial), ("enc", encoder)]),
    ]

    print(f"\n5-FOLD CV — z-blend streams, 3 seeds, 50 epochs  ({len(pool)} complexes)")
    print("=" * 75)
    print(f"  {'Config':<42}  {'τ mean':>8}  {'τ std':>7}  {'Δ vs base':>10}")
    print("  " + "-" * 70)

    results = {}
    baseline = None

    for name, streams in configs:
        tau_m, tau_s = cv_streams(pool, streams)
        results[name] = {"tau_mean": tau_m, "tau_std": tau_s}
        delta = f"{tau_m - baseline:+.4f}" if baseline is not None else "(baseline)"
        if baseline is None:
            baseline = tau_m
        print(f"  {name:<42}  {tau_m:+.4f}    {tau_s:.4f}    {delta:>10}")

    print()
    best = max(results, key=lambda k: results[k]["tau_mean"])
    print(f"Best: {best}")
    print(f"  τ = {results[best]['tau_mean']:+.4f}")
    print(f"\nComparison: RankerV2 burial+enc+cons = 0.2420")

    results["_meta"] = {
        "rankerv2_burial_enc": 0.2375,
        "rankerv2_3stream": 0.2420,
        "n_complexes": len(pool),
        "seeds": 3, "folds": 5, "epochs": 50,
    }
    (OUT / "interface_phys_cv.json").write_text(json.dumps(results, indent=2))
    print(f"Saved → {OUT}/interface_phys_cv.json")


if __name__ == "__main__":
    main()
