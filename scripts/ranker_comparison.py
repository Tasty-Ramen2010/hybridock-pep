#!/usr/bin/env python3
"""
ranker_comparison.py — Compare ref2015-physics, encoder, combined14_CLIP, and
their ensemble across bench-only and bench+gen training conditions.

Training conditions (standard):
  bench_only          baseline: all bench300 train complexes, no gen
  bench+gen_ood_75B   75% bench + ALL OOD gen (N=5 poses each, no pair cap)
  bench+gen250_cap    75% bench + gen250 (N=100 poses, capped at 10 pairs/cx)

Extra conditions (--overnight):
  bench+gen_ood_25B   25% bench + 75% OOD gen
  bench+gen_ood_50B   50% bench + 50% OOD gen
  bench+gen_ood_100G  0% bench + 100% OOD gen (gen-only upper bound)

Ranker types per condition:
  phys14          V2Head(14) on raw ref2015 physics only (no encoder)
  encoder         V2Head(96) on encoder features only  -- replicates F_router_cv
  combined14_CLIP V2Head(110) on encoder ++ winsorized physics  -- production
  ensemble        complex-level z-score blend of phys14 + encoder scores

GPU support: if CUDA available, trains on GPU (sequential) with VRAM capped at
90% to leave headroom for other processes. Falls back to parallel CPU workers.

  PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python \\
      scripts/ranker_comparison.py [--overnight] [--device cpu|cuda|auto]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import pickle
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("ranker_cmp")

REPO = Path(__file__).resolve().parent.parent
D    = REPO / "logs" / "diagnosis"

BENCH_ENC    = D / "feats_bench300.pkl"
BENCH_PHYS   = D / "feats_bench300_physics.pkl"
BENCH_JSON   = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
GEN_OOD_ENC  = D / "feats_gen_ood.pkl"
GEN_OOD_JSON = REPO / "logs" / "confidence_training_data" / "benchmark_results.json"
GEN250_ENC   = D / "feats_gen250.pkl"
GEN250_JSON  = REPO / "logs" / "gen_n100" / "benchmark_results.json"

N_STATIC       = 14
N_FOLDS        = 5
OOD_MAX_PPC    = 10    # OOD is N=5 → C(5,2)=10 natural max
GEN250_MAX_PPC = 10    # cap gen250 at same budget as OOD

# GPU VRAM budget: cap at 90% — leaves ~10% for OS / other processes
VRAM_FRACTION  = 0.90

DEVICE = "cpu"  # set by main() after arg parsing


# ── model ─────────────────────────────────────────────────────────────────────

class V2Head(nn.Module):
    def __init__(self, in_dim: int, h1: int = 128, h2: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bpr(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


# ── data loading ──────────────────────────────────────────────────────────────

def load_pool(enc_pkl: Path, phys_pkl: Path | None, json_path: Path) -> dict:
    """Returns {cname: {"enc": (n,96), "phys": (n,16)|None, "rmsd": (n,)}}.

    Only complexes where every pose has a physics entry get phys set; others
    get phys=None. Requires >= 2 poses to be included.
    """
    enc  = pickle.load(open(enc_pkl, "rb"))
    phys = pickle.load(open(phys_pkl, "rb")) if (phys_pkl and phys_pkl.exists()) else {}
    jd   = json.load(open(json_path))
    tmp: dict = {}
    for k in enc:
        cn, mk, pi = k
        rmsds = jd.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rmsds):
            continue
        e = np.asarray(enc[k], np.float32)
        p = np.asarray(phys[k], np.float32) if k in phys else None
        tmp.setdefault(cn, []).append((e, p, float(rmsds[pi])))
    out = {}
    for c, rows in tmp.items():
        if len(rows) < 2:
            continue
        all_have_phys = all(r[1] is not None for r in rows)
        out[c] = {
            "enc":  np.stack([r[0] for r in rows]),
            "phys": np.stack([r[1] for r in rows]) if all_have_phys else None,
            "rmsd": np.asarray([r[2] for r in rows], np.float32),
        }
    return out


def fit_clip_bounds(pool: dict, complexes: list) -> tuple[np.ndarray, np.ndarray]:
    M = np.concatenate([pool[c]["phys"][:, :N_STATIC] for c in complexes
                        if pool[c]["phys"] is not None], 0)
    return np.percentile(M, 1, 0), np.percentile(M, 99, 0)


def get_feats(d: dict, ranker: str,
              lo: np.ndarray | None, hi: np.ndarray | None) -> np.ndarray:
    enc  = d["enc"]
    phys = d["phys"]
    ph14 = np.clip(phys[:, :N_STATIC], lo, hi) if (phys is not None and lo is not None) else (
           phys[:, :N_STATIC] if phys is not None else None)
    if ranker == "phys14":          return ph14
    if ranker == "encoder":         return enc
    if ranker == "combined14_CLIP": return np.concatenate([enc, ph14], 1)
    raise ValueError(ranker)


def make_pairs(pool: dict, complexes: list, ranker: str,
               lo: np.ndarray | None, hi: np.ndarray | None,
               rng: random.Random, max_ppc: int | None = None) -> list:
    pairs = []
    for c in complexes:
        d = pool.get(c)
        if d is None:
            continue
        if ranker in ("phys14", "combined14_CLIP") and d["phys"] is None:
            continue
        F = get_feats(d, ranker, lo, hi)
        r = d["rmsd"]
        cps = [(i, j) for i, j in combinations(range(len(r)), 2)
               if abs(r[i] - r[j]) >= 1e-6]
        if max_ppc is not None and len(cps) > max_ppc:
            cps = rng.sample(cps, max_ppc)
        for i, j in cps:
            pairs.append((F[i], F[j], 1.0 if r[i] < r[j] else 0.0))
    return pairs


# ── training ──────────────────────────────────────────────────────────────────

def train_head(head: nn.Module, train_pairs: list, seed: int,
               device: str = "cpu") -> nn.Module:
    """Train V2Head with BPR loss. Moves model and tensors to device."""
    if not train_pairs:
        return head
    torch.manual_seed(seed)
    for m in head.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm)):
            m.reset_parameters()
    head = head.to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    fi  = torch.tensor(np.stack([p[0] for p in train_pairs]),
                       dtype=torch.float32, device=device)
    fj  = torch.tensor(np.stack([p[1] for p in train_pairs]),
                       dtype=torch.float32, device=device)
    lbl = torch.tensor([p[2] for p in train_pairs],
                       dtype=torch.float32, device=device)
    n = len(train_pairs)
    epochs = head._n_epochs if hasattr(head, "_n_epochs") else 50
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + ep))
        for b in range(0, n, 512):
            idx = perm[b: b + 512]
            loss = bpr(head(fi[idx]).squeeze(-1),
                       head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head = head.cpu()
    return head


# ── evaluation ────────────────────────────────────────────────────────────────

def score_pool(head: nn.Module, pool: dict, complexes: list,
               ranker: str, lo, hi, device: str = "cpu") -> dict[str, np.ndarray]:
    head = head.to(device)
    head.eval()
    out = {}
    with torch.no_grad():
        for c in complexes:
            d = pool.get(c)
            if d is None:
                continue
            if ranker in ("phys14", "combined14_CLIP") and d["phys"] is None:
                continue
            F = torch.tensor(get_feats(d, ranker, lo, hi),
                             dtype=torch.float32, device=device)
            out[c] = head(F).squeeze(-1).cpu().numpy()
    head = head.cpu()
    return out


def scores_to_tau(scores: dict[str, np.ndarray],
                  pool: dict, complexes: list) -> float:
    taus = []
    for c in complexes:
        s = scores.get(c)
        if s is None or len(s) < 2:
            continue
        r = pool[c]["rmsd"]
        t, _ = sp.kendalltau(-s, r)
        if not math.isnan(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


def ensemble_tau(scores_a: dict, scores_b: dict,
                 pool: dict, complexes: list) -> float:
    """z-score normalise both score sets per complex then sum."""
    taus = []
    for c in complexes:
        sa = scores_a.get(c)
        sb = scores_b.get(c)
        if sa is None or sb is None or len(sa) < 2:
            continue
        def _znorm(x):
            mu, sd = x.mean(), x.std()
            return (x - mu) / (sd + 1e-8)
        s = _znorm(sa) + _znorm(sb)
        t, _ = sp.kendalltau(-s, pool[c]["rmsd"])
        if not math.isnan(t):
            taus.append(t)
    return float(np.mean(taus)) if taus else float("nan")


# ── CV loop ───────────────────────────────────────────────────────────────────

def run_cv(bench: dict, gen: dict | None, gen_max_ppc: int | None,
           condition_name: str, bench_frac: float = 1.0,
           epochs: int = 50, seeds: list[int] | None = None,
           device: str = "cpu") -> dict[str, float]:
    """5-fold CV over bench, optionally augmented with gen data.

    Args:
        bench_frac: fraction of bench train complexes to use (for gen-heavy conditions).
        device: torch device string ("cpu" or "cuda").
    """
    if seeds is None:
        seeds = [0, 1, 2]

    torch.set_num_threads(2)  # polite when sharing CPU with other workers

    bcx = sorted(bench)
    rng_fold = np.random.RandomState(7)
    perm  = list(rng_fold.permutation(len(bcx)))
    folds = [[bcx[i] for i in perm[f::N_FOLDS]] for f in range(N_FOLDS)]

    has_phys = any(bench[c]["phys"] is not None for c in bcx)

    results: dict[str, list] = {r: [] for r in
                                 ["phys14", "encoder", "combined14_CLIP", "ensemble"]}

    for fold_idx in range(N_FOLDS):
        val_c   = folds[fold_idx]
        b_train = [c for c in bcx if c not in set(val_c)]

        if bench_frac < 1.0:
            rng_sel = np.random.RandomState(42 + fold_idx)
            n_sel   = max(1, int(bench_frac * len(b_train)))
            b_train = list(rng_sel.choice(b_train, n_sel, replace=False))

        train_pool = dict(bench)
        gen_cx     = []
        if gen is not None:
            g_tr       = sorted(gen)
            train_pool = {**bench, **gen}
            gen_cx     = g_tr

        lo = hi = None
        if has_phys:
            phys_cx = [c for c in list(b_train) + gen_cx
                       if train_pool[c]["phys"] is not None]
            if phys_cx:
                lo, hi = fit_clip_bounds(train_pool, phys_cx)

        fold_taus: dict[str, list] = {r: [] for r in results}

        gen_has_phys = (gen is not None and
                        any(gen[c]["phys"] is not None for c in gen))
        pr_gen = random.Random(fold_idx)

        # Build pairs once per fold (deterministic bench, seeded gen subsample)
        bench_pairs_phys = (make_pairs(bench, b_train, "phys14",
                                       lo, hi, pr_gen, None)
                            if has_phys else [])
        bench_pairs_enc  = make_pairs(bench, b_train, "encoder",
                                      lo, hi, pr_gen, None)
        bench_pairs_comb = (make_pairs(bench, b_train, "combined14_CLIP",
                                       lo, hi, pr_gen, None)
                            if has_phys else [])
        gen_pairs_enc    = (make_pairs(gen, gen_cx, "encoder",
                                       lo, hi, pr_gen, gen_max_ppc)
                            if gen else [])
        gen_pairs_comb   = (make_pairs(gen, gen_cx, "combined14_CLIP",
                                       lo, hi, pr_gen, gen_max_ppc)
                            if gen_has_phys else [])

        train_phys = bench_pairs_phys
        train_enc  = bench_pairs_enc  + gen_pairs_enc
        train_comb = bench_pairs_comb + gen_pairs_comb

        log.info("  fold %d: phys_pairs=%d  enc_pairs=%d  comb_pairs=%d  val=%d",
                 fold_idx, len(train_phys), len(train_enc),
                 len(train_comb), len(val_c))

        for seed in seeds:
            trained: dict[str, nn.Module] = {}

            for ranker, tr_pairs, in_dim in [
                ("phys14",          train_phys, N_STATIC),
                ("encoder",         train_enc,  96),
                ("combined14_CLIP", train_comb, 96 + N_STATIC),
            ]:
                if not tr_pairs or (ranker in ("phys14", "combined14_CLIP")
                                    and not has_phys):
                    fold_taus[ranker].append(float("nan"))
                    continue
                head = V2Head(in_dim)
                head._n_epochs = epochs  # type: ignore[attr-defined]
                head = train_head(head, tr_pairs, seed, device=device)
                trained[ranker] = head
                scores = score_pool(head, bench, val_c, ranker, lo, hi,
                                    device=device)
                t = scores_to_tau(scores, bench, val_c)
                fold_taus[ranker].append(t)

            if has_phys and "phys14" in trained and "encoder" in trained:
                s_p = score_pool(trained["phys14"],  bench, val_c,
                                 "phys14",  lo, hi, device=device)
                s_e = score_pool(trained["encoder"], bench, val_c,
                                 "encoder", lo, hi, device=device)
                fold_taus["ensemble"].append(
                    ensemble_tau(s_p, s_e, bench, val_c))
            else:
                fold_taus["ensemble"].append(float("nan"))

        for r in results:
            results[r].append(float(np.nanmean(fold_taus[r])))

    return {r: float(np.nanmean(results[r])) for r in results}


# ── multiprocessing worker (CPU-only path) ────────────────────────────────────

def _run_condition_worker(args):
    """Worker for multiprocessing (CPU path) — one condition per process."""
    import torch as _t
    _t.set_num_threads(2)
    cname, bench, gen, gen_max_ppc, bench_frac, epochs, seeds = args
    return cname, run_cv(bench, gen, gen_max_ppc, cname,
                         bench_frac, epochs=epochs, seeds=seeds, device="cpu")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overnight",  action="store_true",
                    help="extended run: 100 epochs, 5 seeds, extra gen-ratio conditions")
    ap.add_argument("--device",     default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="training device (default auto: cuda if free else cpu)")
    ap.add_argument("--epochs",     type=int, default=None,
                    help="override epoch count (default 50, or 100 with --overnight)")
    ap.add_argument("--seeds",      type=int, default=None,
                    help="number of seeds (default 3, or 5 with --overnight)")
    ap.add_argument("--workers",    type=int, default=3,
                    help="parallel workers for CPU path (ignored on GPU)")
    ap.add_argument("--skip-ood",    action="store_true")
    ap.add_argument("--skip-gen250", action="store_true")
    a = ap.parse_args()

    # ── hyperparams ──
    if a.overnight:
        epochs = a.epochs or 100
        seeds  = list(range(a.seeds or 5))
    else:
        epochs = a.epochs or 50
        seeds  = list(range(a.seeds or 3))

    # ── device selection ──
    if a.device == "auto":
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
                # probe: allocate a tiny tensor to confirm we can get VRAM
                _ = torch.zeros(1, device="cuda")
                del _
                torch.cuda.empty_cache()
                device = "cuda"
            except RuntimeError:
                log.warning("CUDA probe failed (OOM?) — falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
    else:
        device = a.device
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        used  = torch.cuda.memory_reserved(0) / 1024**3
        cap   = props.total_memory * VRAM_FRACTION / 1024**3
        log.info("Device: CUDA — %s  |  VRAM cap %.1f GB (%.0f%%)  |  "
                 "reserved %.1f GB",
                 props.name, cap, VRAM_FRACTION * 100, used)
    else:
        log.info("Device: CPU  (torch threads=2 per worker, workers=%d)", a.workers)

    # ── load feature pools ──
    log.info("Loading bench300 features...")
    bench = load_pool(BENCH_ENC, BENCH_PHYS, BENCH_JSON)
    log.info("  bench: %d complexes, phys=%s", len(bench),
             bench[next(iter(bench))]["phys"] is not None)

    gen_ood = gen250 = None

    if not a.skip_ood and GEN_OOD_ENC.exists():
        log.info("Loading gen OOD features...")
        gen_ood = load_pool(GEN_OOD_ENC, None, GEN_OOD_JSON)
        log.info("  gen_ood: %d complexes", len(gen_ood))
    elif not a.skip_ood:
        log.warning("feats_gen_ood.pkl not found — skipping gen_ood conditions")

    if not a.skip_gen250 and GEN250_ENC.exists():
        log.info("Loading gen250 features...")
        gen250 = load_pool(GEN250_ENC, None, GEN250_JSON)
        log.info("  gen250: %d complexes (pair cap=%d)", len(gen250), GEN250_MAX_PPC)

    # ── build condition list ──
    conditions = [
        ("bench_only", dict(gen=None, bench_frac=1.0, gen_max_ppc=None)),
    ]
    if gen250 is not None:
        conditions.append(("bench+gen250_cap10",
                           dict(gen=gen250, bench_frac=0.75,
                                gen_max_ppc=GEN250_MAX_PPC)))
    if gen_ood is not None:
        conditions.append(("bench+gen_ood_75B",
                           dict(gen=gen_ood, bench_frac=0.75,
                                gen_max_ppc=OOD_MAX_PPC)))
        if a.overnight:
            conditions.extend([
                ("bench+gen_ood_50B",
                 dict(gen=gen_ood, bench_frac=0.50, gen_max_ppc=OOD_MAX_PPC)),
                ("bench+gen_ood_25B",
                 dict(gen=gen_ood, bench_frac=0.25, gen_max_ppc=OOD_MAX_PPC)),
                ("gen_ood_only",
                 dict(gen=gen_ood, bench_frac=0.0,  gen_max_ppc=OOD_MAX_PPC)),
            ])

    log.info("\nConfig: epochs=%d  seeds=%d  folds=%d  device=%s  overnight=%s",
             epochs, len(seeds), N_FOLDS, device, a.overnight)
    log.info("Conditions: %s", [c for c, _ in conditions])

    header = (f"\n{'Condition':<26} {'phys14':>8} {'encoder':>8} "
              f"{'comb14':>8} {'ensemble':>9}")
    sep = "-" * len(header)
    log.info(header)
    log.info(sep)

    all_results: dict[str, dict] = {}

    if device == "cuda":
        # Sequential on GPU — CUDA context can't safely fork
        log.info("Running %d conditions sequentially on GPU...", len(conditions))
        for cname, cfg in conditions:
            log.info("  → %s", cname)
            r = run_cv(bench, cfg["gen"], cfg["gen_max_ppc"], cname,
                       cfg["bench_frac"], epochs=epochs, seeds=seeds,
                       device="cuda")
            all_results[cname] = r
            log.info("  %-24s  phys14=%+.4f  encoder=%+.4f  "
                     "combined=%+.4f  ensemble=%+.4f",
                     cname,
                     r["phys14"], r["encoder"],
                     r["combined14_CLIP"], r["ensemble"])
    else:
        # Parallel CPU workers
        worker_args = [
            (cname, bench, cfg["gen"], cfg["gen_max_ppc"],
             cfg["bench_frac"], epochs, seeds)
            for cname, cfg in conditions
        ]
        log.info("Running %d conditions in parallel (workers=%d)...",
                 len(worker_args), a.workers)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=a.workers) as pool:
            results_list = pool.map(_run_condition_worker, worker_args)
        for cname, r in results_list:
            all_results[cname] = r
            log.info("  %-24s  phys14=%+.4f  encoder=%+.4f  "
                     "combined=%+.4f  ensemble=%+.4f",
                     cname,
                     r["phys14"], r["encoder"],
                     r["combined14_CLIP"], r["ensemble"])

    log.info(sep)
    log.info("Reference baselines:")
    log.info("  Production combined14_CLIP (bench_only CV): τ≈0.212")
    log.info("  Campaign F_router best (encoder, 75B+25G OOD): τ≈0.442")

    # ── verdict ──
    best_enc = max((all_results[c]["encoder"] for c in all_results), default=float("nan"))
    best_ens = max((all_results[c]["ensemble"] for c in all_results), default=float("nan"))
    best_comb = all_results.get("bench_only", {}).get("combined14_CLIP", float("nan"))
    log.info("\nVerdict:")
    log.info("  Best encoder  τ=%.4f (target: ≥0.442)", best_enc)
    log.info("  Best ensemble τ=%.4f", best_ens)
    log.info("  Production    τ=%.4f", best_comb)
    if best_enc >= 0.40:
        log.info("  → F_router validated ✓  encoder alone beats production")
    elif best_ens > best_comb + 0.02:
        log.info("  → Ensemble wins  (+%.3f over production)", best_ens - best_comb)
    else:
        log.info("  → Production combined14_CLIP still best — investigate gen data quality")

    # ── save ──
    suffix = "_overnight" if a.overnight else ""
    out_path = REPO / "logs" / "training_campaign" / f"ranker_comparison{suffix}.json"
    out_path.write_text(json.dumps(
        {"config": {"epochs": epochs, "seeds": seeds, "device": device,
                    "overnight": a.overnight},
         "results": all_results},
        indent=2))
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
