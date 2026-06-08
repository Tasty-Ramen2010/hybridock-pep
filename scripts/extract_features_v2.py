#!/usr/bin/env python3
"""
extract_features_v2.py — Extract 115-dim V2 encoder features from bench300 poses.

Uses V2ConfidenceModel (interface-aware pooling + sidechain proxy + SS proxy)
on existing bench300 pose PDBs. Backbone weights loaded from pretrained
RAPiDock-Reloaded checkpoint. New v2 layers are NOT trained yet — features
capture the richer pooling and geometry without learned projection weights
mattering for the architecture test; the head training handles that.

Output: logs/diagnosis/feats_bench300_v2.pkl
  dict[(cx_name, model_key, pose_idx)] -> np.array[115]

Usage (rapidock env):
  PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python3 \
      scripts/extract_features_v2.py [--device cuda] [--batch-size 8]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_v2")

warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock_v2" / "models"))

PARAMS_YML  = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
PRETRAINED  = (
    REPO / "third_party" / "RAPiDock" / "train_models"
    / "CGTensorProductEquivariantModel" / "rapidock_global.pt"
)
BENCH_JSON  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH_CSV   = REPO / "data" / "benchmark300.csv"
TMP_BENCH   = "/tmp/v2_bench_graphs"
OUT_PATH    = REPO / "logs" / "diagnosis" / "feats_bench300_v2.pkl"
PARTIAL_OUT = REPO / "logs" / "diagnosis" / "feats_bench300_v2_partial.pkl"


def load_v2_model(device: str):
    from diffusion_v2 import V2ConfidenceModel
    from argparse import Namespace

    with open(PARAMS_YML) as f:
        params = yaml.safe_load(f)
    params["confidence_mode"] = True

    model = V2ConfidenceModel(Namespace(**params))
    ckpt  = torch.load(PRETRAINED, map_location="cpu")
    # ckpt['model'] has 'encoder.*' keys (from BaseModel wrapper).
    # V2ConfidenceModel directly subclasses CGTensorProductEquivariantModel,
    # so strip the 'encoder.' prefix to match its flat parameter names.
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    if isinstance(state, dict):
        stripped = {
            (k[len("encoder."):] if k.startswith("encoder.") else k): v
            for k, v in state.items()
        }
    else:
        stripped = state
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    log.info(
        "Pretrained weights loaded. Missing: %d  Unexpected: %d",
        len(missing), len(unexpected),
    )
    # Only log actually important missing keys (v2-specific are expected missing)
    backbone_missing = [k for k in missing if "chi1_proj" not in k
                        and "confidence_predictor_v2" not in k
                        and "ss_proj" not in k]
    if backbone_missing:
        log.warning("Unexpected backbone missing keys: %s", backbone_missing[:5])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def _build_base_graphs(json_data: dict, csv_path: Path, tmp_dir: str) -> dict:
    """Reuse the same graph-building logic as confidence_diagnosis.py."""
    import pandas as pd
    from utils.inference_utils import InferenceDataset

    df = pd.read_csv(csv_path)
    names, recs, peps = [], [], []
    for _, row in df.iterrows():
        cname = row["name"]
        if cname not in json_data:
            continue
        rec = row.get("receptor", "")
        pep = row.get("peptide_pdb", "")
        if not Path(str(rec)).exists() or not Path(str(pep)).exists():
            continue
        names.append(cname)
        recs.append(str(rec))
        peps.append(str(pep))

    os.makedirs(tmp_dir, exist_ok=True)
    ds = InferenceDataset(
        output_dir=tmp_dir,
        complex_name_list=names,
        protein_description_list=recs,
        peptide_description_list=peps,
        lm_embeddings=True,
        lm_embeddings_pep=False,
        conformation_type=None,
        conformation_partial="1:1:1",
    )
    out = {}
    for i, n in enumerate(names):
        try:
            g = ds.get(i)
            if g is not None:
                out[n] = g
        except Exception:
            pass
    log.info("Built %d base graphs", len(out))
    return out


def _load_pose_positions(pdb_path: str, exclude_oxt: bool = False):
    try:
        import MDAnalysis
        u = MDAnalysis.Universe(pdb_path)
        positions = []
        for res in u.residues:
            sel = "not type H" + (" and not name OXT" if exclude_oxt else "")
            heavy = res.atoms.select_atoms(sel)
            ca    = heavy.select_atoms("name CA")
            if len(ca) == 0 or len(heavy) == 0:
                continue
            positions.append(heavy.positions.astype(np.float32))
        if not positions:
            return None
        return torch.tensor(np.concatenate(positions))
    except Exception:
        return None


def _inject_pose(bg, pos):
    from utils.diffusion_utils import set_time
    g      = copy.deepcopy(bg)
    center = pos.mean(0)
    g["pep_a"].pos     = pos - center
    if hasattr(g["pep_a"], "x") and g["pep_a"].x is not None:
        g["pep_a"].x = g["pep_a"].x.float()
    g["receptor"].pos  = g["receptor"].pos - center
    if hasattr(g["pep_a"], "node_sigma_emb"):
        del g["pep_a"].node_sigma_emb
    return g


@torch.no_grad()
def extract_features_v2(
    model,
    json_data: dict,
    base_graphs: dict,
    device: str,
    batch_size: int = 8,
) -> dict:
    """Extract 115-dim V2 features. Hooks the first Linear in confidence_predictor_v2."""
    from utils.diffusion_utils import set_time
    from torch_geometric.data import Batch

    entries = []
    for cname, model_results in json_data.items():
        bg = base_graphs.get(cname)
        if bg is None:
            continue
        n_graph = bg["pep_a"].pos.shape[0]
        for mkey, res in model_results.items():
            poses_dir = Path(res["poses_dir"])
            rmsds     = res.get("ref_rmsds", [])
            for i, rmsd in enumerate(rmsds):
                entries.append((cname, mkey, poses_dir, i, float(rmsd), n_graph))

    # Resume from partial checkpoint
    feat_map: dict = {}
    if PARTIAL_OUT.exists():
        with open(PARTIAL_OUT, "rb") as f:
            feat_map = pickle.load(f)
        log.info("Resumed %d features from partial checkpoint", len(feat_map))

    total  = len(entries)
    done   = 0

    for b_start in range(0, total, batch_size):
        batch_entries = entries[b_start : b_start + batch_size]
        graphs, keys  = [], []

        for cname, mkey, poses_dir, i, rmsd, n_graph in batch_entries:
            key = (cname, mkey, i)
            if key in feat_map:
                continue
            pdb = poses_dir / f"pose_{i}.pdb"
            if not pdb.exists():
                continue
            bg  = base_graphs[cname]
            pos = _load_pose_positions(str(pdb))
            if pos is None:
                continue
            if pos.shape[0] != n_graph:
                pos2 = _load_pose_positions(str(pdb), exclude_oxt=True)
                if pos2 is not None and pos2.shape[0] == n_graph:
                    pos = pos2
                else:
                    continue
            try:
                g = _inject_pose(bg, pos)
                set_time(g, 0.0, 0.0, 0.0, 0.0, 1, device="cpu")
                graphs.append(g)
                keys.append(key)
            except Exception:
                continue

        if not graphs:
            done += len(batch_entries)
            continue

        try:
            gbatch = Batch.from_data_list(graphs).to(device)
        except Exception as e:
            log.warning("Batch transfer failed at b_start=%d: %s", b_start, e)
            if device != "cpu":
                torch.cuda.empty_cache()
            done += len(batch_entries)
            continue

        # Hook first Linear in confidence_predictor_v2 to capture 115-dim input.
        # V2ConfidenceModel is a direct CGTensorProduct subclass — no .encoder wrapper.
        feats_captured = []
        def _hook(mod, inp, out):
            feats_captured.append(inp[0].detach().cpu())

        # V2ConfidenceModel directly subclasses CGTensorProductEquivariantModel,
        # so confidence_predictor_v2 is a top-level attribute (no .encoder wrapper).
        cp_v2 = model.confidence_predictor_v2
        first_linear = next(m for m in cp_v2.modules() if isinstance(m, nn.Linear))
        handle = first_linear.register_forward_hook(_hook)
        try:
            model(gbatch)
        except Exception as e:
            log.warning("Forward failed at b_start=%d: %s", b_start, e)
            handle.remove()
            del gbatch
            if device != "cpu":
                torch.cuda.empty_cache()
            done += len(batch_entries)
            continue
        handle.remove()

        if feats_captured:
            feats = feats_captured[0].numpy()  # [batch, 115]
            for k_idx, key in enumerate(keys):
                if k_idx < feats.shape[0]:
                    feat_map[key] = feats[k_idx]

        del gbatch
        if device != "cpu":
            torch.cuda.empty_cache()

        done += len(batch_entries)

        if done % 500 < batch_size:
            log.info("  %d / %d entries processed, %d features extracted",
                     done, total, len(feat_map))
            with open(PARTIAL_OUT, "wb") as f:
                pickle.dump(feat_map, f)

    # Final save
    with open(OUT_PATH, "wb") as f:
        pickle.dump(feat_map, f)
    log.info("Saved %d V2 features → %s", len(feat_map), OUT_PATH)

    if PARTIAL_OUT.exists():
        PARTIAL_OUT.unlink()

    return feat_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    log.info("Loading V2 model...")
    model = load_v2_model(device)

    log.info("Loading bench300 JSON...")
    with open(BENCH_JSON) as f:
        bench_json = json.load(f)

    log.info("Building base graphs (ESM run ~2 min)...")
    base_graphs = _build_base_graphs(bench_json, BENCH_CSV, TMP_BENCH)

    feat_map = extract_features_v2(
        model, bench_json, base_graphs, device, batch_size=args.batch_size
    )

    # Quick sanity check
    sample_feat = next(iter(feat_map.values()))
    log.info("Feature dim: %d  (expected 115)", sample_feat.shape[0])


if __name__ == "__main__":
    main()
