#!/usr/bin/env python3
"""
confidence_exp_e.py — Definitive encoder information + fine-tuning experiment.

Three conditions:
  E0       Cached 96-dim features + E0 batch training (control, τ≈0.28)
  E0_live  Live encoder features (extracted once) + same E0 batch training
           → answers: do live features support τ=0.28 given proper training?
  E1_v3    E0_live head warm-start + encoder fine-tuned at enc_lr=1e-5
           → answers: does encoder FT beat frozen encoder?

Key lessons from failed runs:
  - Live encoder features have 70× smaller absolute variance than cached.
    Fixed by LayerNorm(96) input normalisation on V2Head.
  - Training volume was the real bottleneck: E0 gets 24× more pair-evaluations
    than the per-complex E1 runs. E0_live uses identical procedure to E0.
  - Encoder LR 5e-7 = effectively frozen (weights moved <0.1%). E1_v3 uses 1e-5.

Usage:
  /home/igem/miniconda3/envs/rapidock/bin/python3 -u scripts/confidence_exp_e.py
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
torch.set_num_threads(4)

REPO       = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))

FEAT_BENCH  = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
BENCH_JSON  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH_CSV   = REPO / "data" / "benchmark300.csv"
PARAMS_YML  = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
PRETRAINED  = REPO / "third_party" / "RAPiDock" / "train_models" / \
              "CGTensorProductEquivariantModel" / "rapidock_global.pt"
LIVE_FEATS  = REPO / "logs" / "diagnosis" / "feats_bench300_live.pkl"
OUT         = REPO / "logs" / "training_campaign"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
for _h in logging.root.handlers:
    if hasattr(_h, "stream"):
        _h.stream = os.fdopen(os.dup(_h.stream.fileno()), "w", buffering=1)
log = logging.getLogger("exp_e")


# ── head + training utilities ─────────────────────────────────────────────────

class V2Head(nn.Module):
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),                        # critical: normalise input scale
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(),           nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def bpr_loss(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


def split_complexes(complexes, train_frac=0.85, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(complexes))
    n   = max(1, int(len(complexes) * train_frac))
    return [complexes[i] for i in idx[:n]], [complexes[i] for i in idx[n:]]


def build_dataset(feat_map, json_data):
    ds = {}
    for (cname, mkey, pose_idx), feat in feat_map.items():
        rmsds = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pose_idx >= len(rmsds): continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pose_idx])))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def build_pairs(ds, complexes):
    pairs = []
    for c in complexes:
        for (fi, ri), (fj, rj) in combinations(ds.get(c, []), 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))
    return pairs


def train_head_cached(head, train_pairs, val_pairs, epochs=50, lr=1e-3, seed=0):
    """Batch head training on pre-cached features (E0 / E0_live procedure)."""
    torch.manual_seed(seed)
    for m in head.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm)): m.reset_parameters()

    fi  = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in train_pairs],           dtype=torch.float32)
    vfi = torch.tensor(np.stack([p[0] for p in val_pairs]),   dtype=torch.float32)
    vfj = torch.tensor(np.stack([p[1] for p in val_pairs]),   dtype=torch.float32)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    best_tau, best_state = -1.0, None
    n = len(train_pairs)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b+512]
            loss = bpr_loss(head(fi[idx]).squeeze(-1),
                            head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            vs = head(vfi).squeeze(-1); vt = head(vfj).squeeze(-1)
            correct = ((vs > vt).float()).mean().item()
        if correct > best_tau:
            best_tau = correct; best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)


def eval_tau_cached(head, ds, complexes):
    from scipy import stats as sp
    head.eval(); taus, tops = [], []
    with torch.no_grad():
        for c in complexes:
            poses = ds.get(c, [])
            if len(poses) < 2: continue
            feats  = torch.tensor(np.array([p[0] for p in poses], dtype=np.float32))
            rmsds  = np.array([p[1] for p in poses])
            scores = head(feats).squeeze(-1).numpy()
            tau, _ = sp.kendalltau(-scores, rmsds)
            if math.isnan(tau): continue
            taus.append(tau); tops.append(float(rmsds[np.argmax(scores)]))
    return (float(np.mean(taus)) if taus else float("nan"),
            float(np.mean(tops))  if tops else float("nan"))


# ── encoder + graph utilities ─────────────────────────────────────────────────

class _FeatureRecorder(nn.Module):
    """Drop-in replacement for confidence_predictor; records 96-dim inputs."""
    def __init__(self):
        super().__init__()
        self.captured: list[torch.Tensor] = []

    def reset(self): self.captured.clear()

    def forward(self, x):
        self.captured.append(x.detach().cpu())
        return torch.zeros(x.shape[0], 1, device=x.device)


def load_model(n_unfreeze: int, device: str, new_head: nn.Module):
    import yaml
    from models.model import ConfidenceModel
    from argparse import Namespace
    with open(PARAMS_YML) as f: params = yaml.safe_load(f)
    params["confidence_mode"] = True
    model = ConfidenceModel(Namespace(**params))
    ckpt  = torch.load(PRETRAINED, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    for p in model.parameters(): p.requires_grad_(False)
    enc = model.encoder
    for blk in list(enc.cross_convs)[-n_unfreeze:]:
        for p in blk.parameters(): p.requires_grad_(True)
    for p in new_head.parameters(): p.requires_grad_(True)
    enc.confidence_predictor = new_head
    model.to(device)
    return model, enc


def load_frozen_encoder_with_recorder(device: str):
    """Frozen encoder with FeatureRecorder; used for live feature extraction."""
    import yaml
    from models.model import ConfidenceModel
    from argparse import Namespace
    with open(PARAMS_YML) as f: params = yaml.safe_load(f)
    params["confidence_mode"] = True
    model = ConfidenceModel(Namespace(**params))
    ckpt  = torch.load(PRETRAINED, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    for p in model.parameters(): p.requires_grad_(False)
    rec = _FeatureRecorder()
    model.encoder.confidence_predictor = rec
    model.eval().to(device)
    return model, rec


def build_base_graphs(csv_path, label=""):
    from utils.inference_utils import InferenceDataset
    df     = pd.read_csv(csv_path)
    names, recs, peps = [], [], []
    for _, row in df.iterrows():
        if Path(str(row.get("receptor",""))).exists() and \
           Path(str(row.get("peptide_pdb",""))).exists():
            names.append(row["name"]); recs.append(str(row["receptor"]))
            peps.append(str(row["peptide_pdb"]))
    tmp = f"/tmp/exp_e_base_{label}"
    os.makedirs(tmp, exist_ok=True)
    ds = InferenceDataset(
        output_dir=tmp, complex_name_list=names,
        protein_description_list=recs, peptide_description_list=peps,
        lm_embeddings=True, lm_embeddings_pep=False,
        conformation_type=None, conformation_partial="1:1:1",
    )
    base = {}
    for i, n in enumerate(names):
        try:
            g = ds.get(i)
            if g is not None: base[n] = g
        except Exception: pass
    log.info("Built %d base graphs (%s)", len(base), label)
    return base


def load_pose_positions(pdb: str, exclude_oxt: bool = False):
    import MDAnalysis as mda
    try:
        u = mda.Universe(pdb); pos = []
        for res in u.residues:
            sel   = "not type H" + (" and not name OXT" if exclude_oxt else "")
            heavy = res.atoms.select_atoms(sel)
            if not heavy.select_atoms("name CA").n_atoms or not len(heavy): continue
            pos.append(heavy.positions.astype(np.float32))
        return torch.tensor(np.concatenate(pos)) if pos else None
    except Exception: return None


def inject_pose(bg, pos):
    from utils.diffusion_utils import set_time
    g = copy.deepcopy(bg); center = pos.mean(0)
    g["pep_a"].pos    = pos - center
    g["pep_a"].x      = g["pep_a"].x.float()
    g["receptor"].pos = g["receptor"].pos - center
    if hasattr(g["pep_a"], "node_sigma_emb"): del g["pep_a"].node_sigma_emb
    set_time(g, 0., 0., 0., 0., 1, device="cpu")
    return g


def cache_pose_positions(json_data, base_graphs):
    cache: dict[str, list] = {}
    n_ok = n_skip = 0
    for cname, model_results in json_data.items():
        bg = base_graphs.get(cname)
        if bg is None: continue
        n_g = bg["pep_a"].pos.shape[0]; poses = []
        for mkey, res in model_results.items():
            pdir  = Path(res["poses_dir"]); rmsds = res.get("ref_rmsds", [])
            for i, rmsd in enumerate(rmsds):
                pdb = pdir / f"pose_{i}.pdb"
                if not pdb.exists(): continue
                pos = load_pose_positions(str(pdb))
                if pos is None: n_skip += 1; continue
                if pos.shape[0] != n_g:
                    pos = load_pose_positions(str(pdb), exclude_oxt=True)
                    if pos is None or pos.shape[0] != n_g: n_skip += 1; continue
                poses.append((pos, float(rmsd))); n_ok += 1
        if len(poses) >= 2: cache[cname] = poses
    log.info("Pose cache: %d poses across %d complexes (%d skipped)", n_ok, len(cache), n_skip)
    return cache


def build_cx_graphs_from_cache(cname, pos_cache, base_graphs):
    bg = base_graphs.get(cname)
    if bg is None or cname not in pos_cache: return []
    out = []
    for pos, rmsd in pos_cache[cname]:
        try: out.append((inject_pose(bg, pos), rmsd))
        except Exception: pass
    return out


# ── live feature extraction ───────────────────────────────────────────────────

def extract_live_feats_to_pkl(bench_json, bench_base, device):
    """
    Run all bench300 poses through the frozen pretrained encoder.
    Save 96-dim features as {(cname, mkey, pose_idx): feat}.
    Same format as feats_bench300.pkl — can be used directly with build_dataset.
    """
    from torch_geometric.data import Batch

    model, rec = load_frozen_encoder_with_recorder(device)
    feat_map   = {}
    n_cx = 0

    for cname, model_results in bench_json.items():
        bg = bench_base.get(cname)
        if bg is None: continue
        n_g = bg["pep_a"].pos.shape[0]

        for mkey, res in model_results.items():
            pdir  = Path(res["poses_dir"])
            rmsds = res.get("ref_rmsds", [])

            # Build pose graphs, keeping (mkey, pose_idx) provenance
            entries, graphs = [], []
            for i, rmsd in enumerate(rmsds):
                pdb = pdir / f"pose_{i}.pdb"
                if not pdb.exists(): continue
                pos = load_pose_positions(str(pdb))
                if pos is None: continue
                if pos.shape[0] != n_g:
                    pos = load_pose_positions(str(pdb), exclude_oxt=True)
                    if pos is None or pos.shape[0] != n_g: continue
                try:
                    graphs.append(inject_pose(bg, pos))
                    entries.append((mkey, i))
                except Exception: pass

            if len(graphs) < 2: continue

            rec.reset()
            try:
                batch = Batch.from_data_list(graphs).to(device)
                with torch.no_grad(): model(batch)
                del batch; torch.cuda.empty_cache()
            except Exception:
                torch.cuda.empty_cache(); continue

            if not rec.captured: continue
            feats = torch.cat(rec.captured, dim=0).numpy()
            if feats.shape[0] != len(entries): continue

            for (mk, pi), feat in zip(entries, feats):
                feat_map[(cname, mk, pi)] = feat.astype(np.float32)

        n_cx += 1
        if n_cx % 20 == 0:
            log.info("  Extracted %d / %d complexes", n_cx, len(bench_json))

    log.info("Live feature extraction complete: %d features, %d complexes",
             len(feat_map), len(set(k[0] for k in feat_map)))
    return feat_map


# ── E1_v3: on-the-fly encoder fine-tuning ─────────────────────────────────────

def run_encoder_finetune(label, device, train_pos_cache, train_base_graphs,
                         val_pos_cache, val_base_graphs, train_c, val_c,
                         head_init_state, n_epochs=50,
                         enc_lr=1e-5, head_lr=1e-4,
                         pairs_per_cx=200):
    """
    True encoder fine-tuning:
    - Warm-starts head from E0_live (already calibrated for live features)
    - Unfreezes last 1 cross_conv block
    - Uses grad_accum = len(train_c): ONE optimizer step per epoch,
      gradient averaged over ALL train complexes
      → matches E0's gradient diversity while allowing encoder to adapt
    - 50 epochs = 50 clean encoder gradient steps from full training data
    """
    from torch_geometric.data import Batch
    from scipy import stats as sp

    log.info("=== %s ===", label)
    new_head = V2Head().to(device)
    new_head.load_state_dict(head_init_state)
    log.info("  Head warm-started from E0_live weights")

    model, enc = load_model(n_unfreeze=1, device=device, new_head=new_head)
    enc_params  = [p for blk in list(enc.cross_convs)[-1:] for p in blk.parameters()]
    head_params = list(new_head.parameters())
    trainable   = sum(p.numel() for p in enc_params + head_params)
    log.info("  Trainable params: %d  (enc_lr=%.0e  head_lr=%.0e)",
             trainable, enc_lr, head_lr)

    opt = torch.optim.Adam([
        {"params": enc_params,  "lr": enc_lr,  "weight_decay": 1e-4},
        {"params": head_params, "lr": head_lr, "weight_decay": 1e-4},
    ])

    rng = np.random.RandomState(0)
    best_tau, best_head_state = -1.0, None

    for ep in range(n_epochs):
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.eval()

        cx_order = list(train_c); rng.shuffle(cx_order)
        opt.zero_grad()
        ep_loss, ep_pairs, n_accum = 0.0, 0, 0

        for cname in cx_order:
            cx_data = build_cx_graphs_from_cache(cname, train_pos_cache, train_base_graphs)
            if len(cx_data) < 2: continue

            batch  = Batch.from_data_list([g for g, _ in cx_data]).to(device)
            rmsds  = [r for _, r in cx_data]
            try: scores = model(batch)
            except Exception:
                del batch; torch.cuda.empty_cache(); continue

            pair_idx = list(combinations(range(len(cx_data)), 2))
            if len(pair_idx) > pairs_per_cx:
                sel      = rng.choice(len(pair_idx), pairs_per_cx, replace=False)
                pair_idx = [pair_idx[k] for k in sel]

            cx_loss = torch.tensor(0.0, device=device); cx_n = 0
            for i, j in pair_idx:
                ri, rj = rmsds[i], rmsds[j]
                if abs(ri - rj) < 1e-6: continue
                lbl     = torch.tensor(1.0 if ri < rj else 0.0, device=device)
                cx_loss = cx_loss + bpr_loss(scores[i:i+1], scores[j:j+1], lbl.unsqueeze(0))
                cx_n += 1

            if cx_n > 0:
                # Divide by len(train_c) — normalise so accumulated gradient
                # is the mean over all complexes, not the sum.
                (cx_loss / len(train_c)).backward()
                ep_loss  += cx_loss.item()
                ep_pairs += cx_n
                n_accum  += 1

            del batch, scores, cx_loss; torch.cuda.empty_cache()

        # ONE step per epoch: gradient is averaged over all complexes
        torch.nn.utils.clip_grad_norm_(enc_params,  0.1)
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step(); opt.zero_grad()

        # Eval
        model.eval(); taus, tops = [], []
        with torch.no_grad():
            for cname in val_c:
                cx_data = build_cx_graphs_from_cache(cname, val_pos_cache, val_base_graphs)
                if len(cx_data) < 2: continue
                batch  = Batch.from_data_list([g for g, _ in cx_data]).to(device)
                rmsds  = np.array([r for _, r in cx_data])
                try: sv = model(batch).detach().cpu().numpy()
                except Exception:
                    del batch; torch.cuda.empty_cache(); continue
                tau, _ = sp.kendalltau(-sv, rmsds)
                if not math.isnan(tau):
                    taus.append(tau); tops.append(float(rmsds[np.argmax(sv)]))
                del batch; torch.cuda.empty_cache()

        ep_tau  = float(np.mean(taus)) if taus else float("nan")
        ep_top1 = float(np.mean(tops)) if tops else float("nan")
        mean_loss = ep_loss / max(ep_pairs, 1)
        log.info("  ep=%2d  loss=%.4f  val_τ=%.4f  top1=%.3f  n_cx=%d",
                 ep, mean_loss, ep_tau, ep_top1, n_accum)

        if not math.isnan(ep_tau) and ep_tau > best_tau:
            best_tau = ep_tau
            best_head_state = copy.deepcopy(new_head.state_dict())

    if best_head_state is not None: new_head.load_state_dict(best_head_state)
    log.info("  %s best_τ=%.4f", label, best_tau)
    return best_tau


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--e0-epochs",   type=int, default=50)
    ap.add_argument("--e1v3-epochs", type=int, default=50)
    ap.add_argument("--skip-extract", action="store_true",
                    help="Reuse existing feats_bench300_live.pkl if present")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    bench_json = json.load(open(BENCH_JSON))

    # ── E0: cached features, E0 batch training (control) ─────────────────────
    log.info("=== E0: cached features (control) ===")
    with open(FEAT_BENCH, "rb") as f: bench_feats = pickle.load(f)
    bench_ds   = build_dataset(bench_feats, bench_json)
    bench_all  = sorted(bench_ds.keys())
    train_c, val_c = split_complexes(bench_all, 0.85, seed=42)
    tr_pairs  = build_pairs(bench_ds, train_c)
    va_pairs  = build_pairs(bench_ds, val_c)
    head_e0   = V2Head()
    train_head_cached(head_e0, tr_pairs, va_pairs, epochs=args.e0_epochs, seed=0)
    tau_e0, top1_e0 = eval_tau_cached(head_e0, bench_ds, val_c)
    log.info("  E0 τ=%.4f  top1=%.3f  (pairs: train=%d val=%d)",
             tau_e0, top1_e0, len(tr_pairs), len(va_pairs))

    rows = [{"label": "E0_cached", "tau": tau_e0, "top1": top1_e0,
             "n_train_pairs": len(tr_pairs), "note": "control"}]

    if device == "cpu":
        log.warning("No GPU — skipping E0_live and E1_v3"); return

    # ── Build base graphs (one-time) ──────────────────────────────────────────
    log.info("Building bench base graphs (~3-5 min)...")
    bench_base = build_base_graphs(BENCH_CSV, label="bench")

    # ── E0_live: extract live encoder features, same E0 training procedure ───
    # Critical experiment: tests whether live encoder features support τ=0.28
    # when given the same training volume as E0 (batch training, all pairs).
    # If yes → the per-complex training volume was the bottleneck all along.
    # If no  → live encoder features genuinely lack discriminative signal.
    log.info("=== E0_live: live features + E0 batch training ===")

    if args.skip_extract and LIVE_FEATS.exists():
        log.info("  Loading cached live features from %s", LIVE_FEATS)
        with open(LIVE_FEATS, "rb") as f: live_feat_map = pickle.load(f)
    else:
        log.info("  Extracting live encoder features for all bench300 complexes (~25 min)...")
        live_feat_map = extract_live_feats_to_pkl(bench_json, bench_base, device)
        with open(LIVE_FEATS, "wb") as f: pickle.dump(live_feat_map, f)
        log.info("  Saved to %s", LIVE_FEATS)

    live_ds   = build_dataset(live_feat_map, bench_json)
    live_all  = sorted(live_ds.keys())
    # Use same train/val split as E0 — compare on identical validation set
    live_tr_c, live_va_c = split_complexes(
        [c for c in train_c if c in live_ds], 1.0, seed=42)  # all that survived extraction
    live_tr_c = [c for c in train_c if c in live_ds]
    live_va_c = [c for c in val_c   if c in live_ds]
    live_tr_pairs = build_pairs(live_ds, live_tr_c)
    live_va_pairs = build_pairs(live_ds, live_va_c)
    log.info("  Live DS: %d train complexes (%d pairs), %d val complexes (%d pairs)",
             len(live_tr_c), len(live_tr_pairs), len(live_va_c), len(live_va_pairs))

    head_e0live = V2Head()
    train_head_cached(head_e0live, live_tr_pairs, live_va_pairs,
                      epochs=args.e0_epochs, seed=0)
    tau_e0live, top1_e0live = eval_tau_cached(head_e0live, live_ds, live_va_c)
    log.info("  E0_live τ=%.4f  top1=%.3f", tau_e0live, top1_e0live)
    log.info("  Gap vs E0: %.4f  (%.1f%%)",
             tau_e0live - tau_e0, 100 * (tau_e0live - tau_e0) / max(tau_e0, 1e-9))

    rows.append({"label": "E0_live", "tau": tau_e0live, "top1": top1_e0live,
                 "n_train_pairs": len(live_tr_pairs),
                 "note": "live features, E0 batch training"})

    e0_live_head_state = copy.deepcopy(head_e0live.state_dict())

    # ── E1_v3: encoder fine-tuning on top of E0_live ─────────────────────────
    # Design:
    #   - Head warm-starts from E0_live (already calibrated for live features)
    #   - Encoder unfrozen at enc_lr=1e-5 (real fine-tuning, not pseudo-frozen)
    #   - grad_accum = ALL complexes → one clean step per epoch
    #     (mean gradient over full training set — maximum stability)
    #   - 50 epochs = 50 encoder gradient steps
    # Question: does the encoder learn to output more pose-discriminative
    # features when given a well-calibrated head as a training signal?
    log.info("=== E1_v3: encoder fine-tuning (warm-started from E0_live) ===")

    log.info("  Caching bench pose positions...")
    bench_pos = cache_pose_positions(bench_json, bench_base)

    tau_e1v3 = run_encoder_finetune(
        label      = "E1_v3_encoder_FT",
        device     = device,
        train_pos_cache   = bench_pos,
        train_base_graphs = bench_base,
        val_pos_cache     = bench_pos,
        val_base_graphs   = bench_base,
        train_c    = live_tr_c,
        val_c      = live_va_c,
        head_init_state = e0_live_head_state,
        n_epochs   = args.e1v3_epochs,
        enc_lr     = 1e-5,
        head_lr    = 1e-4,
        pairs_per_cx = 200,
    )

    rows.append({"label": "E1_v3_encoder_FT", "tau": tau_e1v3, "top1": float("nan"),
                 "n_train_pairs": -1, "note": "live features, encoder FT, 1 step/epoch"})

    # ── summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp_e_results.csv", index=False)

    # Merge into all_results.csv
    existing = OUT / "all_results.csv"
    if existing.exists():
        df_old = pd.read_csv(existing)
        df_old = df_old[~df_old.get("label", df_old.get("exp", "")).isin(
            ["E0_cached", "E0_live", "E1_v3_encoder_FT"])]
        df_out = pd.concat([df_old, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(existing, index=False)

    log.info("\n=== Exp E Final Summary ===")
    log.info("  E0 (cached, control):  τ=%.4f", tau_e0)
    log.info("  E0_live (live feats):  τ=%.4f  (Δ=%.4f)", tau_e0live, tau_e0live - tau_e0)
    log.info("  E1_v3 (encoder FT):    τ=%.4f  (Δ=%.4f vs E0_live)", tau_e1v3, tau_e1v3 - tau_e0live)
    log.info("  Results: %s", OUT / "exp_e_results.csv")


if __name__ == "__main__":
    main()
