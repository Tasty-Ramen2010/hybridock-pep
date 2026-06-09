#!/usr/bin/env python3
"""
interface_rmsd_cv.py — Test whether interface-RMSD labels improve ranker τ.

Compares 5-fold CV τ under three label conditions:
    (A) global_rmsd  — existing ref_rmsds (Kabsch on all Cα)
    (B) iface_rmsd   — Kabsch RMSD on interface residues only  [Route 1 fix]
    (C) weighted     — 3× weight on interface residues in RMSD computation

Each condition runs the same 3-stream z-blend ranker (burial+encoder+consensus)
on the 115-cx physics set, 5-fold CV, 3 seeds.

Output: prints table + saves logs/training_campaign/interface_rmsd_cv.json
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

BJSON = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))
IFACE = json.load(open(REPO / "logs" / "analysis_bench300" / "interface_rmsd_labels.json"))

BURIAL_IDX = [0, 2, 1]
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])
CONS_WEIGHT = 0.5
ENC_DIM = 96

torch.set_num_threads(4)


# ── helpers ──────────────────────────────────────────────────────────────────

def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _per_cx_z(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0)
    return (M - mu) / np.where(sd < 1e-9, 1.0, sd)


# ── load pool ────────────────────────────────────────────────────────────────

def load_pool(label: str) -> dict:
    """Load the 115-cx physics pool with the requested RMSD label.

    label: "global" | "iface" | "weighted"
    """
    phys = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    enc = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    pool: dict = {}

    for k, pv in phys.items():
        cn, mk, pi = k
        if k not in enc:
            continue

        # global RMSD (always available as baseline)
        global_r = BJSON.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(global_r):
            continue
        g_rmsd = float(global_r[pi])

        if label == "global":
            rmsd = g_rmsd

        elif label == "iface":
            iface_entry = IFACE.get(cn, {}).get(mk, {})
            iface_list = iface_entry.get("interface_rmsds", [])
            if pi >= len(iface_list) or iface_list[pi] is None:
                rmsd = g_rmsd  # fallback
            else:
                rmsd = float(iface_list[pi])

        elif label == "weighted":
            # 3× weight on interface residues: computed from stored iface RMSD +
            # residue counts rather than re-reading PDBs. Approximation:
            #   weighted_rmsd ≈ sqrt( (3*n_i*iface_sq + n_t*global_sq) / (3*n_i + n_t) )
            # where n_i = n_interface, n_t = n_total
            iface_entry = IFACE.get(cn, {}).get(mk, {})
            iface_list = iface_entry.get("interface_rmsds", [])
            n_i = iface_entry.get("n_interface", 0)
            n_t = iface_entry.get("n_total", 0)
            if (pi >= len(iface_list) or iface_list[pi] is None
                    or n_i == 0 or n_t == 0):
                rmsd = g_rmsd
            else:
                i_rmsd = float(iface_list[pi])
                tail_n = max(n_t - n_i, 0)
                # iface_sq ≈ iface_rmsd^2, global_sq ≈ global_rmsd^2
                # Reconstruct tail RMSD: global^2 * n_t = iface^2 * n_i + tail^2 * tail_n
                if tail_n > 0:
                    tail_sq = max(0.0, (g_rmsd**2 * n_t - i_rmsd**2 * n_i) / tail_n)
                    w_sq = (3 * n_i * i_rmsd**2 + tail_n * tail_sq) / (3 * n_i + tail_n)
                    rmsd = math.sqrt(w_sq)
                else:
                    rmsd = i_rmsd
        else:
            raise ValueError(f"Unknown label: {label}")

        pool.setdefault(cn, []).append({
            "phys": np.asarray(pv, np.float64),
            "enc": np.asarray(enc[k], np.float64),
            "rmsd": rmsd,
            "key": k,
        })

    return {c: v for c, v in pool.items() if len(v) >= 3}


# ── consensus (Cα, from stored pairwise RMSD matrix) ─────────────────────────

def add_consensus(pool: dict):
    """Add Cα consensus using stored pairwise RMSD data in benchmark JSON."""
    for cn, rows in pool.items():
        # pairwise_rmsd is stored as dict with mean/median/std, not per-pose
        # Fall back to building from pose PDBs via stored poses_dir
        # Use the same mk for all rows (mixed model pool — just skip consensus here;
        # burial_analysis.py already showed consensus adds <+0.005 τ in the CV)
        for r in rows:
            r.setdefault("consensus", np.nan)


# ── MLP head ─────────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def _score(h, F):
    with torch.no_grad():
        return h(torch.tensor(F, dtype=torch.float32)).squeeze(-1).numpy()


# ── 3-stream z-blend CV ───────────────────────────────────────────────────────

def cv_3stream(pool, seeds=(0, 1, 2), folds=5, epochs=50):
    """5-fold CV of the 3-stream z-blend (burial + encoder head, no ca_consensus
    since we don't have per-pose coords here — adds <0.005 anyway)."""
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
            # build pairwise training data for encoder head
            pairs = []
            for c in tr_cxs:
                rows = pool[c]
                E = _per_cx_z(np.array([r["enc"] for r in rows]))
                rr = np.array([r["rmsd"] for r in rows])
                for i, j in combinations(range(len(rr)), 2):
                    if abs(rr[i] - rr[j]) > 1e-6:
                        pairs.append((E[i], E[j], 1.0 if rr[i] < rr[j] else 0.0))

            enc_head = _train(pairs, ENC_DIM, sd, epochs)

            ts = []
            for c in val_cxs:
                rows = pool[c]
                n = len(rows)
                rr = np.array([r["rmsd"] for r in rows])

                phys = np.array([r["phys"] for r in rows])
                burial = _per_cx_z(phys[:, BURIAL_IDX]) @ BURIAL_SIGN

                enc_f = _per_cx_z(np.array([r["enc"] for r in rows]))
                enc_s = _score(enc_head, enc_f)

                score = _z(-burial) + _z(enc_s)

                t, _ = sp.kendalltau(-score, rr)
                if not math.isnan(t):
                    ts.append(t)
            seed_taus.append(float(np.mean(ts)) if ts else float("nan"))

        fold_taus.append(float(np.nanmean(seed_taus)))

    return float(np.mean(fold_taus)), float(np.std(fold_taus))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading pools...")
    pools = {
        "global_rmsd": load_pool("global"),
        "iface_rmsd":  load_pool("iface"),
        "weighted_rmsd": load_pool("weighted"),
    }
    print(f"  {len(pools['global_rmsd'])} complexes (physics set)")

    print("\n5-FOLD CV — 3-stream z-blend (burial + encoder), 3 seeds, 50 epochs")
    print("=" * 65)
    print(f"  {'Label':<20}  {'τ mean':>8}  {'τ std':>7}  {'Δ vs global':>12}")
    print("  " + "-" * 55)

    results = {}
    baseline_tau = None

    for name, pool in pools.items():
        tau_m, tau_s = cv_3stream(pool)
        results[name] = {"tau_mean": tau_m, "tau_std": tau_s}
        delta = ""
        if baseline_tau is not None:
            delta = f"{tau_m - baseline_tau:+.4f}"
        else:
            baseline_tau = tau_m
        print(f"  {name:<20}  {tau_m:+.4f}    {tau_s:.4f}    {delta or '(baseline)':>12}")

    print()
    best = max(results, key=lambda k: results[k]["tau_mean"])
    print(f"Best: {best}  τ = {results[best]['tau_mean']:+.4f}")
    print(f"(RankerV2 burial+enc CV τ = 0.2375 for comparison)")

    results["_note"] = {
        "rankerv2_burial_enc_no_cons": 0.2375,
        "rankerv2_3stream_with_cons": 0.2420,
    }
    (OUT / "interface_rmsd_cv.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}/interface_rmsd_cv.json")


if __name__ == "__main__":
    main()
