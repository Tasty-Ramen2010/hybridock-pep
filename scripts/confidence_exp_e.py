#!/usr/bin/env python3
"""
confidence_exp_e.py — Frozen vs partial encoder finetuning (standalone).

Three conditions:
  E0  Frozen encoder        Use cached 96-dim features, head-only training
  E1  Unfreeze last block   cross_convs[-1] + head (1.05M trainable params)
  E2  Unfreeze last 2       cross_convs[-2:] + head (1.70M trainable params)

Memory design:
  - base_graphs cached once (~500 MB for 240 receptors)
  - Pose graphs built ON-THE-FLY inside training loop, discarded after each complex
  - Peak GPU: ~1-2 GB (one complex batch at a time)
  - Peak RAM: ~1-2 GB (base_graphs + one complex in flight)

Usage:
  conda run -n rapidock python3 -u scripts/confidence_exp_e.py --device cuda
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
from collections import defaultdict
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

FEAT_BENCH = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH_CSV  = REPO / "data" / "benchmark300.csv"
PARAMS_YML = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
PRETRAINED = REPO / "third_party" / "RAPiDock" / "train_models" / \
             "CGTensorProductEquivariantModel" / "rapidock_global.pt"
OUT        = REPO / "logs" / "training_campaign"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("exp_e")


# ── shared head / training utilities (same as campaign script) ────────────────

class V2Head(nn.Module):
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(),          nn.Dropout(0.2),
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
        if pose_idx >= len(rmsds):
            continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pose_idx])))
    return {k: v for k, v in ds.items() if len(v) >= 2}


def build_pairs(ds, complexes):
    pairs = []
    for c in complexes:
        for (fi, ri), (fj, rj) in combinations(ds.get(c, []), 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))
    return pairs


def train_head_cached(head, train_pairs, val_pairs, epochs=50, seed=0):
    """Standard cached-feature head training (E0)."""
    torch.manual_seed(seed)
    for m in head.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm)): m.reset_parameters()

    fi  = torch.tensor(np.stack([p[0] for p in train_pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in train_pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in train_pairs],           dtype=torch.float32)
    vfi = torch.tensor(np.stack([p[0] for p in val_pairs]),   dtype=torch.float32)
    vfj = torch.tensor(np.stack([p[1] for p in val_pairs]),   dtype=torch.float32)
    vlb = torch.tensor([p[2] for p in val_pairs],             dtype=torch.float32)

    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state = -1.0, None
    n = len(train_pairs)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b: b+512]
            loss = bpr_loss(head(fi[idx]).squeeze(-1), head(fj[idx]).squeeze(-1), lbl[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = ((head(vfi).squeeze(-1) > head(vfj).squeeze(-1)).float() - vlb).abs().lt(0.5).float().mean().item()
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(head.state_dict())

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
    return float(np.mean(taus)) if taus else float("nan"), \
           float(np.mean(tops))  if tops else float("nan")


# ── encoder loading + graph utilities ────────────────────────────────────────

def load_model(n_unfreeze: int, device: str, new_head: nn.Module):
    """Load pretrained model, freeze all, unfreeze last n cross_conv blocks."""
    import yaml
    from models.model import ConfidenceModel
    from argparse import Namespace

    with open(PARAMS_YML) as f:
        params = yaml.safe_load(f)
    params["confidence_mode"] = True
    model = ConfidenceModel(Namespace(**params))
    ckpt  = torch.load(PRETRAINED, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    enc = model.encoder
    for blk in list(enc.cross_convs)[-n_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for p in new_head.parameters():
        p.requires_grad_(True)
    enc.confidence_predictor = new_head
    model.to(device)
    return model, enc


def build_base_graphs(bench_csv, bench_json_data):
    """Build per-complex base graphs (receptor + peptide topology, no pose)."""
    from utils.inference_utils import InferenceDataset

    df     = pd.read_csv(bench_csv)
    names, recs, peps = [], [], []
    for _, row in df.iterrows():
        if Path(str(row.get("receptor", ""))).exists() and \
           Path(str(row.get("peptide_pdb", ""))).exists():
            names.append(row["name"]); recs.append(str(row["receptor"]))
            peps.append(str(row["peptide_pdb"]))

    tmp = "/tmp/exp_e_base"
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
    log.info("Built %d base graphs", len(base))
    return base


def load_pose_positions(pdb: str, exclude_oxt: bool = False):
    import MDAnalysis as mda
    try:
        u   = mda.Universe(pdb)
        pos = []
        for res in u.residues:
            sel   = "not type H" + (" and not name OXT" if exclude_oxt else "")
            heavy = res.atoms.select_atoms(sel)
            if not heavy.select_atoms("name CA").n_atoms or not len(heavy): continue
            pos.append(heavy.positions.astype(np.float32))
        return torch.tensor(np.concatenate(pos)) if pos else None
    except Exception: return None


def inject_pose(bg, pos):
    from utils.diffusion_utils import set_time
    g      = copy.deepcopy(bg)
    center = pos.mean(0)
    g["pep_a"].pos    = pos - center
    if g["pep_a"].x is not None:
        g["pep_a"].x  = g["pep_a"].x.float()
    g["receptor"].pos = g["receptor"].pos - center
    if hasattr(g["pep_a"], "node_sigma_emb"):
        del g["pep_a"].node_sigma_emb
    set_time(g, 0.0, 0.0, 0.0, 0.0, 1, device="cpu")
    return g


def build_cx_graphs(cname, bench_json_data, base_graphs):
    """Build (graph, rmsd) list for one complex. Called per-epoch on-the-fly."""
    bg  = base_graphs.get(cname)
    if bg is None: return []
    n_g = bg["pep_a"].pos.shape[0]
    out = []
    for mkey, res in bench_json_data.get(cname, {}).items():
        pdir  = Path(res["poses_dir"])
        rmsds = res.get("ref_rmsds", [])
        for i, rmsd in enumerate(rmsds):
            pdb = pdir / f"pose_{i}.pdb"
            if not pdb.exists(): continue
            pos = load_pose_positions(str(pdb))
            if pos is None: continue
            if pos.shape[0] != n_g:
                pos = load_pose_positions(str(pdb), exclude_oxt=True)
                if pos is None or pos.shape[0] != n_g: continue
            try:
                out.append((inject_pose(bg, pos), float(rmsd)))
            except Exception: pass
    return out


# ── Exp E1 / E2: on-the-fly GPU finetuning ───────────────────────────────────

def run_finetune(label: str, n_unfreeze: int, device: str,
                 bench_json: dict, base_graphs: dict,
                 bench_ds: dict, train_c: list, val_c: list,
                 n_epochs: int = 20, pairs_per_cx: int = 10) -> dict:
    from torch_geometric.data import Batch
    from scipy import stats as sp

    log.info("=== %s: unfreezing %d cross_conv block(s) ===", label, n_unfreeze)
    new_head  = V2Head().to(device)
    model, enc = load_model(n_unfreeze, device, new_head)
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable params: %d", trainable)

    enc_block_params = [p for blk in list(enc.cross_convs)[-n_unfreeze:]
                        for p in blk.parameters()]
    opt = torch.optim.Adam([
        {"params": enc_block_params,          "lr": 5e-6, "weight_decay": 1e-4},
        {"params": list(new_head.parameters()), "lr": 1e-3, "weight_decay": 1e-4},
    ])

    rng       = np.random.RandomState(0)
    best_tau  = -1.0
    best_head = None

    for ep in range(n_epochs):
        # ── training ──────────────────────────────────────────────────────────
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.eval()

        ep_loss, ep_pairs = 0.0, 0
        cx_order = list(train_c); rng.shuffle(cx_order)

        for cname in cx_order:
            cx_data = build_cx_graphs(cname, bench_json, base_graphs)
            if len(cx_data) < 2: continue

            batch  = Batch.from_data_list([g for g, _ in cx_data]).to(device)
            rmsds  = [r for _, r in cx_data]
            try:
                scores = model(batch)          # [n_poses] on device
            except Exception:
                del batch; torch.cuda.empty_cache(); continue

            pair_idx = list(combinations(range(len(cx_data)), 2))
            if len(pair_idx) > pairs_per_cx:
                sel = rng.choice(len(pair_idx), pairs_per_cx, replace=False)
                pair_idx = [pair_idx[k] for k in sel]

            loss = torch.tensor(0.0, device=device)
            for i, j in pair_idx:
                ri, rj = rmsds[i], rmsds[j]
                if abs(ri - rj) < 1e-6: continue
                lbl  = torch.tensor(1.0 if ri < rj else 0.0, device=device)
                loss = loss + bpr_loss(scores[i:i+1], scores[j:j+1], lbl.unsqueeze(0))
                ep_pairs += 1

            if ep_pairs > 0 and loss.item() > 0:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_block_params + list(new_head.parameters()), 1.0)
                opt.step()
                ep_loss += loss.item()

            # Free immediately — key to keeping RAM/VRAM in check
            del batch, scores, loss
            torch.cuda.empty_cache()

        # ── eval ──────────────────────────────────────────────────────────────
        model.eval()
        taus, tops = [], []
        with torch.no_grad():
            for cname in val_c:
                cx_data = build_cx_graphs(cname, bench_json, base_graphs)
                if len(cx_data) < 2: continue

                batch  = Batch.from_data_list([g for g, _ in cx_data]).to(device)
                rmsds  = np.array([r for _, r in cx_data])
                try:
                    scores = model(batch).detach().cpu().numpy()
                except Exception:
                    del batch; torch.cuda.empty_cache(); continue

                tau, _ = sp.kendalltau(-scores, rmsds)
                if not math.isnan(tau):
                    taus.append(tau); tops.append(float(rmsds[np.argmax(scores)]))

                del batch; torch.cuda.empty_cache()

        ep_tau  = float(np.mean(taus)) if taus else float("nan")
        ep_top1 = float(np.mean(tops)) if tops else float("nan")
        log.info("  ep=%2d  loss=%.4f  val_τ=%.4f  top1=%.3f",
                 ep, ep_loss / max(ep_pairs, 1), ep_tau, ep_top1)

        if not math.isnan(ep_tau) and ep_tau > best_tau:
            best_tau  = ep_tau
            best_head = copy.deepcopy(new_head.state_dict())

    log.info("  %s best_τ=%.4f", label, best_tau)
    return {
        "exp": "E_finetune", "label": label, "seed": 0,
        "n_train_cx": len(train_c), "n_train_pairs": n_epochs * len(train_c) * pairs_per_cx,
        "tau": best_tau, "top1": ep_top1,
        "train_acc": float("nan"), "val_acc": float("nan"),
        "best_val_acc": best_tau, "best_epoch": -1, "overfit_gap": float("nan"),
        "n_unfreeze_blocks": n_unfreeze, "trainable_params": trainable,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--epochs",  type=int, default=20)
    ap.add_argument("--pairs-per-cx", type=int, default=10)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    bench_json = json.load(open(BENCH_JSON))

    # ── E0: frozen encoder, cached features ───────────────────────────────────
    log.info("=== E0: frozen encoder (cached features) ===")
    with open(FEAT_BENCH, "rb") as f:
        bench_feats = pickle.load(f)
    bench_ds  = build_dataset(bench_feats, bench_json)
    bench_all = sorted(bench_ds.keys())
    train_c, val_c = split_complexes(bench_all, 0.85, seed=42)

    head_e0   = V2Head()
    tr_pairs  = build_pairs(bench_ds, train_c)
    va_pairs  = build_pairs(bench_ds, val_c)
    train_head_cached(head_e0, tr_pairs, va_pairs, epochs=50, seed=0)
    tau_e0, top1_e0 = eval_tau_cached(head_e0, bench_ds, val_c)
    log.info("  E0 τ=%.4f  top1=%.3f", tau_e0, top1_e0)

    rows = [{
        "exp": "E_finetune", "label": "E0_frozen", "seed": 0,
        "n_train_cx": len(train_c), "n_train_pairs": len(tr_pairs),
        "tau": tau_e0, "top1": top1_e0,
        "train_acc": float("nan"), "val_acc": float("nan"),
        "best_val_acc": float("nan"), "best_epoch": -1, "overfit_gap": float("nan"),
        "n_unfreeze_blocks": 0, "trainable_params": sum(p.numel() for p in head_e0.parameters()),
    }]

    if device == "cpu":
        log.warning("No GPU — skipping E1/E2")
    else:
        # ── build base graphs once ─────────────────────────────────────────────
        log.info("Building base graphs (one-time, ~3-5 min)...")
        base_graphs = build_base_graphs(BENCH_CSV, bench_json)

        # ── E1: unfreeze last 1 cross_conv ─────────────────────────────────────
        rows.append(run_finetune(
            "E1_unfreeze_last1", n_unfreeze=1, device=device,
            bench_json=bench_json, base_graphs=base_graphs,
            bench_ds=bench_ds, train_c=train_c, val_c=val_c,
            n_epochs=args.epochs, pairs_per_cx=args.pairs_per_cx,
        ))

        # ── E2: unfreeze last 2 cross_convs ────────────────────────────────────
        rows.append(run_finetune(
            "E2_unfreeze_last2", n_unfreeze=2, device=device,
            bench_json=bench_json, base_graphs=base_graphs,
            bench_ds=bench_ds, train_c=train_c, val_c=val_c,
            n_epochs=args.epochs, pairs_per_cx=args.pairs_per_cx,
        ))

    # ── merge with A-F results and save ───────────────────────────────────────
    df_new = pd.DataFrame(rows)
    existing = OUT / "all_results.csv"
    if existing.exists():
        df_old = pd.read_csv(existing)
        # Drop any prior E_finetune rows and replace
        df_old = df_old[df_old["exp"] != "E_finetune"]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(existing, index=False)
    df_new.to_csv(OUT / "exp_e_results.csv", index=False)
    log.info("Saved results to %s", existing)

    log.info("\n=== Exp E Summary ===")
    for r in rows:
        log.info("  %-25s  τ=%.4f  top1=%.3f  trainable_params=%s",
                 r["label"], r["tau"], r["top1"],
                 r.get("trainable_params", "n/a"))


if __name__ == "__main__":
    main()
