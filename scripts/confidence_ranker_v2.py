#!/usr/bin/env python3
"""
confidence_ranker_v2.py — Production pose ranker (3-stream z-blend).

Derived from the burial-axis investigation (2026-06-09). Supersedes the
combined14_CLIP joint head. Key findings that motivate this design:

  • ref2015 carries ONE collinear ranking axis ("burial": fa_atr/fa_rep/fa_sol
    PC1 = 92.6% var, τ≈0.20). Reweighting to the signed burial coordinate beats
    ref2015's own total_score (CV 0.191 vs 0.170) and the full 14-term head.
  • A jointly-trained head on concatenated features OVERFITS the modality
    weighting (CV 0.205). An equal-weight per-complex z-blend of independently
    scored streams REGULARIZES and wins (CV 0.238–0.242).
  • Geometric consensus (mean pairwise Cα-RMSD of a pose to the rest of the
    ensemble) is a genuinely orthogonal signal (ρ=0.33 to burial, τ≈0.125).
    Weak, but free — production clustering already builds the RMSD matrix.

Three streams, each reduced to a per-complex z-scored scalar, summed:
    score = z(−burial)  +  z(encoder_head)  +  w_cons · z(−consensus)
(higher = better pose). burial and consensus are lower=better, so negated.

CV (5-fold, held-out complexes, 115-cx physics set):
    burial + enc            τ = 0.2375
    burial + enc + cons     τ = 0.2420   ← this config
    production combined14   τ ≈ 0.20

NOTE: τ≈0.24 is the practical ceiling for these signals. The global-peptide-RMSD
target is only loosely coupled to interface energy/geometry, which caps achievable
τ regardless of feature engineering. Breaking past ~0.25 needs a new signal source
(interface-RMSD label, per-pose MM-GBSA, or a larger learned pose-quality model) —
out of ranker scope.

Usage:
    # train + persist the encoder head and config
    PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python \\
        scripts/confidence_ranker_v2.py --train

    # at inference (in your pipeline):
    from scripts.confidence_ranker_v2 import RankerV2
    r = RankerV2.load()
    scores = r.rank(phys=phys_16d, enc=enc_96d, rmsd_matrix=clustering_dist)
"""
from __future__ import annotations

import argparse
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
OUT = REPO / "logs" / "confidence"
OUT.mkdir(parents=True, exist_ok=True)

# burial axis: fa_atr (anti-corr → flip), fa_sol, fa_rep
BURIAL_IDX = [0, 2, 1]
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])
CONS_WEIGHT = 0.5          # 1:1:0.5 ≡ 1:1:1 in CV (0.2419 vs 0.2420); 0.5 is gentler
ENC_DIM = 96

torch.set_num_threads(4)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _per_cx_z(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (M - mu) / sd


class EncoderHead(nn.Module):
    """Maps a 96-dim per-complex-normalized encoder feature to a scalar."""

    def __init__(self, d: int = ENC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RankerV2:
    """3-stream pose ranker. Stateless except for the encoder head."""

    def __init__(self, enc_head: EncoderHead, cons_weight: float = CONS_WEIGHT):
        self.enc_head = enc_head.eval()
        self.cons_weight = cons_weight

    # ── inference ──────────────────────────────────────────────────────────
    def rank(self, phys: np.ndarray, enc: np.ndarray,
             rmsd_matrix: np.ndarray | None = None,
             ca_coords: list[np.ndarray] | None = None) -> np.ndarray:
        """Score N poses of ONE complex. Higher = better (lower predicted RMSD).

        Args:
            phys: (N, 16) ref2015 physics features (raw, score-only).
            enc:  (N, 96) encoder features (pretrained RAPiDock encoder).
            rmsd_matrix: (N, N) pairwise Cα-RMSD — pass the clustering distance
                matrix to get consensus for free. If None, computed from ca_coords.
            ca_coords: list of (L,3) peptide Cα arrays (shared receptor frame),
                used only if rmsd_matrix is None.

        Returns:
            (N,) score array; argmax = best pose.
        """
        n = len(phys)
        burial = _per_cx_z(phys[:, BURIAL_IDX]) @ BURIAL_SIGN     # lower = native
        with torch.no_grad():
            enc_score = self.enc_head(
                torch.tensor(_per_cx_z(enc), dtype=torch.float32)
            ).squeeze(-1).numpy()                                 # higher = native

        score = _z(-burial) + _z(enc_score)

        cons = self._consensus(n, rmsd_matrix, ca_coords)
        if cons is not None:
            score = score + self.cons_weight * _z(-cons)          # lower cons = native

        return score

    @staticmethod
    def _consensus(n: int, rmsd_matrix: np.ndarray | None,
                   ca_coords: list[np.ndarray] | None) -> np.ndarray | None:
        if rmsd_matrix is not None:
            m = np.array(rmsd_matrix, dtype=float)
            np.fill_diagonal(m, np.nan)
            cons = np.nanmean(m, axis=1)
            return cons if np.std(cons) > 1e-9 else None
        if ca_coords is not None:
            valid = [c for c in ca_coords if c is not None]
            if len(valid) < 3:
                return None
            L = min(c.shape[0] for c in valid)
            co = [c[:L] if c is not None else None for c in ca_coords]
            cons = np.full(n, np.nan)
            for i in range(n):
                if co[i] is None:
                    continue
                ds = [np.sqrt(np.mean(np.sum((co[i] - co[j]) ** 2, 1)))
                      for j in range(n) if i != j and co[j] is not None]
                if ds:
                    cons[i] = np.mean(ds)
            cons = np.nan_to_num(cons, nan=np.nanmean(cons[~np.isnan(cons)]))
            return cons if np.std(cons) > 1e-9 else None
        return None

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, tag: str = "v2") -> None:
        torch.save(self.enc_head.state_dict(), OUT / f"ranker_{tag}_enc_head.pt")
        (OUT / f"ranker_{tag}_config.json").write_text(json.dumps({
            "streams": ["burial", "encoder", "consensus"],
            "burial_idx": BURIAL_IDX, "burial_sign": BURIAL_SIGN.tolist(),
            "cons_weight": self.cons_weight, "enc_dim": ENC_DIM,
            "blend": "per-complex z-score, equal weight (cons scaled)",
            "cv_tau": 0.2420, "note": "see module docstring; ceiling ~0.24",
        }, indent=2))

    @classmethod
    def load(cls, tag: str = "v2") -> "RankerV2":
        cfg = json.loads((OUT / f"ranker_{tag}_config.json").read_text())
        head = EncoderHead(cfg["enc_dim"])
        head.load_state_dict(torch.load(OUT / f"ranker_{tag}_enc_head.pt"))
        return cls(head, cfg["cons_weight"])


# ── training ─────────────────────────────────────────────────────────────────
def _load_training_pool() -> dict:
    bjson = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))
    phys = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    enc = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    pool: dict = {}
    for k, pv in phys.items():
        cn, mk, pi = k
        r = bjson.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(r) or k not in enc:
            continue
        pool.setdefault(cn, []).append(
            {"enc": np.asarray(enc[k], np.float64), "rmsd": float(r[pi])})
    return {c: v for c, v in pool.items() if len(v) >= 3}


def train_encoder_head(pool: dict, epochs: int = 50, seed: int = 0) -> EncoderHead:
    """Train the encoder stream head with BPR on per-complex-normalized features."""
    pairs = []
    for c, rows in pool.items():
        E = _per_cx_z(np.array([r["enc"] for r in rows]))
        rr = np.array([r["rmsd"] for r in rows])
        for i, j in combinations(range(len(rr)), 2):
            if abs(rr[i] - rr[j]) > 1e-6:
                pairs.append((E[i], E[j], 1.0 if rr[i] < rr[j] else 0.0))
    torch.manual_seed(seed)
    head = EncoderHead()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    fi = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lb = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    n = len(pairs)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b + 512]
            loss = -F.logsigmoid(
                (head(fi[idx]).squeeze(-1) - head(fj[idx]).squeeze(-1))
                * (lb[idx] * 2 - 1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train + persist the head")
    a = ap.parse_args()

    pool = _load_training_pool()
    print(f"Training pool: {len(pool)} complexes")
    head = train_encoder_head(pool)
    ranker = RankerV2(head)

    # sanity: in-sample τ of the full 3-stream ranker (NOT a CV number; the
    # honest held-out τ is 0.242 from burial_analysis.py)
    phys_all = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    bjson = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))
    enc_all = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    taus = []
    for c, rows in pool.items():
        keys = [k for k in phys_all
                if k[0] == c and k in enc_all
                and k[2] < len(bjson.get(k[0], {}).get(k[1], {}).get("ref_rmsds", []))]
        if len(keys) < 3:
            continue
        phys = np.array([phys_all[k] for k in keys])
        enc = np.array([enc_all[k] for k in keys])
        rmsd = np.array([bjson[k[0]][k[1]]["ref_rmsds"][k[2]] for k in keys])
        s = ranker.rank(phys, enc, ca_coords=None)  # no consensus in this sanity pass
        t, _ = sp.kendalltau(-s, rmsd)
        if not math.isnan(t):
            taus.append(t)
    print(f"In-sample τ (burial+enc, no consensus): {np.mean(taus):+.4f}")
    print("Held-out CV τ (3-stream w/ consensus): 0.2420  [see burial_analysis.py]")

    if a.train:
        ranker.save("v2")
        print(f"Saved → {OUT}/ranker_v2_enc_head.pt + ranker_v2_config.json")


if __name__ == "__main__":
    main()
