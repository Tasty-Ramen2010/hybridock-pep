#!/usr/bin/env python3
"""Train RAPiDock confidence model from bench300 pose PDB files.

Strategy
--------
- Architecture : ConfidenceModel (same backbone as diffusion model, confidence_mode=True)
- Init         : Load pretrained rapidock_global.pt encoder weights (strict=False)
- Training     : ONLY the confidence_predictor MLP head is trained (encoder frozen)
- Data         : 4,800 pose PDBs from logs/analysis_bench300/ with RMSD labels
- Objective    : Pairwise ranking (BPR / margin ranking loss)
                 For each complex × model: C(5,2)=10 pairs → ~96,000 pairs total
- Epochs       : 60

Usage (rapidock env):
    conda run -n rapidock python3 scripts/train_confidence_model.py \
        --epochs 60 --lr 1e-3 --device cuda \
        --out-dir train_models/confidence_model

Outputs:
    train_models/confidence_model/confidence_model.pt   (best checkpoint)
    train_models/confidence_model/model_parameters.yml  (copy of diffusion params)
    logs/confidence_training/training_log.csv
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))

# Suppress noisy warnings from the RAPiDock data pipeline
warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from dataset.peptide_feature import get_ori_peptide_feature_mda
from models.model import ConfidenceModel
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from utils.diffusion_utils import set_time
from utils.inference_utils import InferenceDataset

import MDAnalysis

log = logging.getLogger("conf_train")


# ── constants ─────────────────────────────────────────────────────────────────

BENCH300_JSON    = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH300_CSV     = REPO / "data" / "benchmark300.csv"
MODEL_DIR        = REPO / "third_party" / "RAPiDock" / "train_models" / "CGTensorProductEquivariantModel"
PRETRAINED_CKPT  = MODEL_DIR / "rapidock_global.pt"
MODELS_IN_BENCH  = ["pretrained", "v3c", "v4c", "v5c"]


# ── helpers ───────────────────────────────────────────────────────────────────

def build_all_base_graphs(bench300_df: pd.DataFrame, bench300_data: dict,
                           tmp_dir: str = "/tmp/conf_train") -> dict[str, HeteroData]:
    """Build base graphs for ALL complexes in one InferenceDataset call.

    This batches ESM generation over all 240 receptors (runs ESM only once).
    Returns {cname: HeteroData}.
    """
    names, receptors, peptides = [], [], []
    for _, row in bench300_df.iterrows():
        cname = row["name"]
        if cname not in bench300_data:
            continue
        rec = row["receptor"]
        pep = row["peptide_pdb"]
        if not Path(rec).exists() or not Path(pep).exists():
            log.warning("Missing files for %s — skip", cname)
            continue
        names.append(cname)
        receptors.append(rec)
        peptides.append(pep)

    # Pre-create output dirs (InferenceDataset requires them)
    for n in names:
        os.makedirs(os.path.join(tmp_dir, n), exist_ok=True)

    log.info("Running InferenceDataset for %d complexes (ESM runs once)...", len(names))
    ds = InferenceDataset(
        output_dir=tmp_dir,
        complex_name_list=names,
        protein_description_list=receptors,
        peptide_description_list=peptides,
        lm_embeddings=True,
        lm_embeddings_pep=False,
        conformation_type=None,
        conformation_partial="1:1:1",
    )

    base_graphs = {}
    for i, cname in enumerate(names):
        try:
            g = ds.get(i)
            base_graphs[cname] = g
            if (i + 1) % 20 == 0:
                log.info("  Built graph %d/%d", i + 1, len(names))
        except Exception as e:
            log.warning("Graph build failed for %s (idx %d): %s", cname, i, e)
    log.info("Built %d / %d base graphs", len(base_graphs), len(names))
    return base_graphs


def _load_pose_positions(pose_pdb: str) -> torch.Tensor:
    """Load heavy-atom positions from pose PDB in residue-then-atom order (no H)."""
    u = MDAnalysis.Universe(pose_pdb)
    positions = []
    for res in u.residues:
        heavy = res.atoms.select_atoms("not type H")
        if len(heavy) == 0:
            continue
        # Filter out residues without backbone (should never happen in pose PDBs)
        ca = heavy.select_atoms("name CA")
        if len(ca) == 0:
            continue
        positions.append(heavy.positions.astype(np.float32))
    return torch.tensor(np.concatenate(positions, axis=0))


def _inject_pose_into_graph(graph: HeteroData, pose_positions: torch.Tensor) -> HeteroData:
    """Replace pep_a.pos with pose positions (deep-copy the graph first)."""
    g = copy.deepcopy(graph)
    n_graph = g["pep_a"].pos.shape[0]
    n_pose  = pose_positions.shape[0]
    if n_graph != n_pose:
        raise ValueError(f"Atom count mismatch: graph={n_graph}, pose={n_pose}")
    g["pep_a"].pos = pose_positions.to(dtype=torch.float)
    return g


def _center_graph(graph: HeteroData) -> HeteroData:
    """Center graph on receptor Cα mean (CPU, in-place).

    pep.pos is intentionally NOT set here — the encoder sets it internally
    via get_updated_peptide_feature in the forward pass.
    """
    center = graph["receptor"].pos.mean(dim=0, keepdim=True)
    graph["receptor"].pos    = graph["receptor"].pos    - center
    graph["pep_a"].pos       = graph["pep_a"].pos       - center
    graph["pep_a"].orig_pos  = graph["pep_a"].orig_pos  - center
    return graph


# ── dataset ───────────────────────────────────────────────────────────────────

class PairwiseConfidenceDataset(torch.utils.data.Dataset):
    """Dataset of (graph_i, graph_j, label) pairs for pairwise ranking.

    label=1  ⟹  pose_i has lower RMSD (better) than pose_j
    label=0  ⟹  pose_j is better
    """

    def __init__(self, pairs):
        self.pairs = pairs  # list of (graph_i, graph_j, label)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_training_pairs(bench300_data: dict, bench300_df: pd.DataFrame,
                          base_graphs: dict,
                          device, max_complexes: int = -1) -> list:
    """Build pairwise training data from bench300 poses + pre-built base graphs.

    Returns a flat list of (graph_i, graph_j, label) tuples.
    """
    pairs = []
    n_ok = 0
    n_fail = 0

    complexes = list(bench300_data.items())
    if max_complexes > 0:
        complexes = complexes[:max_complexes]

    for ci, (cname, model_results) in enumerate(complexes):
        base_graph = base_graphs.get(cname)
        if base_graph is None:
            n_fail += 1
            continue

        for mname in MODELS_IN_BENCH:
            if mname not in model_results:
                continue
            result = model_results[mname]
            poses_dir = Path(result["poses_dir"])
            ref_rmsds = result["ref_rmsds"]

            pose_graphs = []
            pose_rmsds  = []
            for i, rmsd in enumerate(ref_rmsds):
                pose_pdb = poses_dir / f"pose_{i}.pdb"
                if not pose_pdb.exists():
                    continue
                try:
                    pose_pos = _load_pose_positions(str(pose_pdb))
                    g = _inject_pose_into_graph(base_graph, pose_pos)
                    g = _center_graph(g)          # CPU, no model call
                    set_time(g, 0.0, 0.0, 0.0, 0.0, 1, device="cpu")
                    pose_graphs.append(g)
                    pose_rmsds.append(float(rmsd))
                except Exception as e:
                    log.debug("Pose %d for %s/%s failed: %s", i, cname, mname, e)
                    continue

            # Generate all C(n,2) pairs
            for i in range(len(pose_graphs)):
                for j in range(i + 1, len(pose_graphs)):
                    ri, rj = pose_rmsds[i], pose_rmsds[j]
                    if abs(ri - rj) < 0.01:
                        continue  # Skip ties
                    label = 1.0 if ri < rj else 0.0
                    pairs.append((pose_graphs[i], pose_graphs[j], label))

        n_ok += 1
        if (ci + 1) % 20 == 0:
            log.info("  Pairs built for %d/%d complexes, %d pairs so far",
                     ci + 1, len(complexes), len(pairs))

    log.info("Built %d pairs from %d complexes (%d failed)", len(pairs), n_ok, n_fail)
    return pairs


# ── model setup ───────────────────────────────────────────────────────────────

def build_pairs_from_corpus(corpus_jsonl: Path, pool_csv: Path, base_graphs: dict,
                            max_complexes: int = -1) -> list:
    """Build pairwise ranking data from a gen_confidence_corpus.py JSONL corpus.

    Corpus line: {"name": str, "poses": [{"pose": path, "rmsd": float}, ...]}

    Only pairs whose RMSDs differ by a real margin are kept: two poses that are both
    ~7A apart from native teach the ranker nothing except to fit noise, and the
    project's ranker work already showed noise-fitting is how these models stall.
    """
    import itertools
    pairs = []
    n_cx = 0
    with open(corpus_jsonl) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cname = rec["name"]
            base = base_graphs.get(cname)
            if base is None:
                continue
            poses = [p for p in rec.get("poses", []) if "rmsd" in p]
            if len(poses) < 2:
                continue
            built = []
            for p in poses:
                try:
                    pos = _load_pose_positions(p["pose"])
                    g = _center_graph(_inject_pose_into_graph(base, pos))
                    built.append((g, float(p["rmsd"])))
                except Exception:
                    continue
            if len(built) < 2:
                continue
            for (gi, ri), (gj, rj) in itertools.combinations(built, 2):
                if abs(ri - rj) < 0.5:      # uninformative pair
                    continue
                pairs.append((gi, gj, 1.0 if ri < rj else 0.0))
            n_cx += 1
            if max_complexes > 0 and n_cx >= max_complexes:
                break
    log.info("Corpus pairs: %d from %d complexes", len(pairs), n_cx)
    return pairs


def load_confidence_model(model_dir: Path, pretrained_ckpt: Path, device,
                          unfreeze_encoder: int = 0) -> ConfidenceModel:
    with open(model_dir / "model_parameters.yml") as f:
        args = Namespace(**yaml.full_load(f))

    # Do NOT set rmsd_classification_cutoff — we want num_confidence_outputs=1
    # (single scalar score for pairwise ranking, not multi-threshold classification)
    if hasattr(args, "rmsd_classification_cutoff"):
        delattr(args, "rmsd_classification_cutoff")

    model = ConfidenceModel(args)  # num_confidence_outputs=1 → scalar output

    # Load pretrained encoder weights (strict=False to skip confidence_predictor)
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info("Loaded pretrained weights — missing: %d, unexpected: %d",
             len(missing), len(unexpected))
    log.info("Missing keys: %s", missing[:5])

    # Encoder freezing policy.
    #
    # Default (unfreeze_encoder=0) freezes the whole encoder and trains only the
    # confidence_predictor head. That is *frozen-embedding* rescoring, and
    # project_ranker_ceiling_jun10 measured it hitting a wall at tau~0.14:
    # "post-hoc rescoring with frozen embeddings hits a wall because embeddings
    # weren't trained for ranking". The memory's prescribed fix is DFMDock-style
    # co-training, i.e. letting the representation itself learn pose quality --
    # which requires unfreezing encoder layers. That is what >0 enables.
    #
    # Unfreezing is done from the TOP down (last cross/intra conv layers first),
    # because the deepest interaction layers carry the pose-discriminative signal
    # while early embedding layers are generic; this also limits how many params
    # move, which matters because the corpus is finite.
    if unfreeze_encoder <= 0:
        for name, param in model.encoder.named_parameters():
            if "confidence_predictor" not in name:
                param.requires_grad = False
        log.info("Encoder FROZEN (head-only training)")
    else:
        conv_ids = sorted({
            int(m.group(1))
            for name, _ in model.encoder.named_parameters()
            for m in [re.search(r"(?:cross_convs|intra_convs)\.(\d+)\.", name)]
            if m
        })
        keep = set(conv_ids[-unfreeze_encoder:]) if conv_ids else set()
        for name, param in model.encoder.named_parameters():
            if "confidence_predictor" in name:
                continue
            m = re.search(r"(?:cross_convs|intra_convs)\.(\d+)\.", name)
            param.requires_grad = bool(m and int(m.group(1)) in keep)
        log.info("Encoder CO-TRAINING: unfroze top %d conv layer(s) %s of %s",
                 unfreeze_encoder, sorted(keep), conv_ids)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %d / %d (%.2f%%)", n_trainable, n_total,
             100 * n_trainable / n_total)
    return model.to(device)


# ── training ──────────────────────────────────────────────────────────────────

def bpr_loss(score_i: torch.Tensor, score_j: torch.Tensor,
             label: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss.

    label=1 means i is better (lower RMSD), so score_i > score_j is desired.
    """
    diff = score_i - score_j
    sign = label * 2 - 1.0  # +1 if i better, -1 if j better
    return -F.logsigmoid(diff * sign).mean()


def margin_loss(score_i, score_j, label, margin=0.5):
    """Margin ranking loss: push winner score ≥ loser score + margin."""
    target = label * 2 - 1.0  # +1 if i wins
    return F.margin_ranking_loss(score_i, score_j, target, margin=margin)


def collate_pairs(batch):
    """Custom collate: return two batched graphs + labels."""
    from torch_geometric.data import Batch
    graphs_i = Batch.from_data_list([b[0] for b in batch])
    graphs_j = Batch.from_data_list([b[1] for b in batch])
    labels   = torch.tensor([b[2] for b in batch], dtype=torch.float)
    return graphs_i, graphs_j, labels


def train_epoch(model, loader, optimizer, device):
    """Train one epoch in float32 (no AMP).

    AMP is disabled because the frozen SO(3) equivariant encoder produces
    NaN activations in float16 (spherical-harmonic values overflow).  The
    confidence predictor head has only 7 k params so fp32 is cheap.
    """
    model.train()
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0
    n_nan      = 0
    for graphs_i, graphs_j, labels in loader:
        graphs_i = graphs_i.to(device)
        graphs_j = graphs_j.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        # Full float32 forward — no autocast
        si = model(graphs_i).squeeze(-1)   # [B]
        sj = model(graphs_j).squeeze(-1)   # [B]
        loss = bpr_loss(si, sj, labels)

        # Guard: skip NaN batches (can happen on degenerate graphs)
        if not torch.isfinite(loss):
            n_nan += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        # Accuracy: correct if winner has higher score
        with torch.no_grad():
            pred_winner = (si > sj).float()  # 1 if i wins by score
            n_correct += (pred_winner == labels).float().sum().item()
            n_total   += len(labels)

    if n_nan > 0:
        log.warning("  Skipped %d NaN batches this epoch", n_nan)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_correct  = 0
    n_total    = 0
    for graphs_i, graphs_j, labels in loader:
        graphs_i = graphs_i.to(device)
        graphs_j = graphs_j.to(device)
        labels   = labels.to(device)
        si = model(graphs_i).squeeze(-1)
        sj = model(graphs_j).squeeze(-1)
        loss = bpr_loss(si, sj, labels)
        total_loss += loss.item() * len(labels)
        pred_winner = (si > sj).float()
        n_correct += (pred_winner == labels).float().sum().item()
        n_total   += len(labels)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",         type=int,   default=60)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--batch-size",     type=int,   default=16)
    ap.add_argument("--val-split",      type=float, default=0.15)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--max-complexes",  type=int,   default=-1,
                    help="Limit complexes for debugging (-1 = all)")
    ap.add_argument("--tmp-dir",        default="/tmp/conf_train")
    ap.add_argument("--out-dir",        default="train_models/confidence_model")
    ap.add_argument("--bench300-json",  default=str(BENCH300_JSON))
    ap.add_argument("--bench300-csv",   default=str(BENCH300_CSV))
    ap.add_argument("--model-dir",      default=str(MODEL_DIR))
    ap.add_argument("--pretrained",     default=str(PRETRAINED_CKPT))
    ap.add_argument("--log-dir",        default="logs/confidence_training")
    ap.add_argument("--unfreeze-encoder", type=int, default=0, metavar="N",
                    help="Co-train the top N encoder conv layers along with the head. "
                         "0 (default) = original behaviour: encoder fully frozen, i.e. "
                         "frozen-embedding rescoring, which project_ranker_ceiling_jun10 "
                         "measured hitting a wall at tau~0.14. >0 enables the DFMDock-style "
                         "co-training that memory prescribes but was never runnable here.")
    ap.add_argument("--encoder-lr",     type=float, default=1e-5,
                    help="LR for co-trained encoder layers (head keeps --lr). Much lower "
                         "than the head LR to avoid destroying the pretrained representation.")
    ap.add_argument("--pool-csv",       default=str(REPO/"data"/"fullparam_train_pool.csv"),
                    help="Pool CSV used to resolve corpus complex -> receptor/peptide paths.")
    ap.add_argument("--corpus-jsonl",   default=None,
                    help="Large labelled pose corpus from gen_confidence_corpus.py "
                         "(JSONL: {name, poses:[{pose, rmsd}]}). Needed for co-training; "
                         "bench300 alone (~1200 poses) overfits an unfrozen encoder.")
    args = ap.parse_args()

    # ── setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log"),
        ],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── load data ──────────────────────────────────────────────────────────
    t0 = time.time()
    if args.corpus_jsonl:
        # Large-corpus path (co-training). bench300 is NOT usable for this: it has
        # only ~1200 labelled poses, and 213 of its 240 complexes are inside
        # fullparam_train_pool, so it is contaminated as a held-out set anyway.
        log.info("Loading corpus %s (pool=%s)", args.corpus_jsonl, args.pool_csv)
        names = []
        with open(args.corpus_jsonl) as fh:
            for line in fh:
                try:
                    names.append(json.loads(line)["name"])
                except Exception:
                    continue
        pool = pd.read_csv(args.pool_csv)
        pool = pool[pool["complex_name"].isin(set(names))]
        corpus_df = pd.DataFrame({
            "name": pool["complex_name"],
            "receptor": pool["protein_description"],
            "peptide_pdb": pool["peptide_description"],
        })
        log.info("Corpus complexes resolvable against pool: %d / %d",
                 len(corpus_df), len(set(names)))
        base_graphs = build_all_base_graphs(
            corpus_df, {n: {} for n in corpus_df["name"]}, tmp_dir=args.tmp_dir)
        log.info("Base graphs done in %.1f min", (time.time() - t0) / 60)
        t0 = time.time()
        pairs = build_pairs_from_corpus(
            Path(args.corpus_jsonl), Path(args.pool_csv), base_graphs,
            max_complexes=args.max_complexes,
        )
    else:
        log.info("Loading bench300 data...")
        with open(args.bench300_json) as f:
            bench300_data = json.load(f)
        bench300_df = pd.read_csv(args.bench300_csv)

        log.info("Building base graphs (one-shot ESM for all %d complexes)...", len(bench300_df))
        base_graphs = build_all_base_graphs(bench300_df, bench300_data, tmp_dir=args.tmp_dir)
        log.info("Base graphs done in %.1f min", (time.time() - t0) / 60)

        log.info("Injecting pose positions and building pairs...")
        t0 = time.time()
        pairs = build_training_pairs(
            bench300_data, bench300_df, base_graphs, device,
            max_complexes=args.max_complexes,
        )
    log.info("Data build: %.1f min, %d total pairs", (time.time() - t0) / 60, len(pairs))

    if len(pairs) == 0:
        log.error("No pairs built — check data paths")
        sys.exit(1)

    # ── split ──────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(pairs))
    n_val   = max(1, int(len(pairs) * args.val_split))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    train_ds = PairwiseConfidenceDataset([pairs[i] for i in trn_idx])
    val_ds   = PairwiseConfidenceDataset([pairs[i] for i in val_idx])
    log.info("Train pairs: %d  |  Val pairs: %d", len(train_ds), len(val_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pairs, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pairs, num_workers=0,
    )

    # ── model ──────────────────────────────────────────────────────────────
    model = load_confidence_model(Path(args.model_dir), Path(args.pretrained), device,
                                  unfreeze_encoder=args.unfreeze_encoder)

    # Separate LR for co-trained encoder layers: the head is randomly initialised and
    # needs a large LR, while the encoder is pretrained and must move slowly or it
    # forgets the representation that makes it useful at all (the same catastrophic-
    # forgetting dynamic that destroyed every generator finetune in this project).
    _head, _enc = [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (_head if "confidence_predictor" in n_ else _enc).append(p_)
    groups = [{"params": _head, "lr": args.lr}]
    if _enc:
        groups.append({"params": _enc, "lr": args.encoder_lr})
        log.info("Param groups: head=%d tensors @lr=%g | encoder=%d tensors @lr=%g",
                 len(_head), args.lr, len(_enc), args.encoder_lr)
    optimizer = Adam(groups, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    # scaler intentionally not used — AMP disabled (fp16 causes NaN in SO(3) encoder)

    # Copy model parameters YAML to out_dir
    shutil.copy2(Path(args.model_dir) / "model_parameters.yml",
                 out_dir / "model_parameters.yml")

    # ── training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_rows = []
    log.info("Starting training: %d epochs, lr=%.1e, batch=%d",
             args.epochs, args.lr, args.batch_size)

    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        trn_loss, trn_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t_ep
        log.info("ep %3d  trn_loss=%.4f acc=%.3f  val_loss=%.4f acc=%.3f  "
                 "lr=%.2e  t=%.0fs",
                 ep, trn_loss, trn_acc, val_loss, val_acc,
                 optimizer.param_groups[0]["lr"], elapsed)

        log_rows.append({
            "epoch": ep, "trn_loss": trn_loss, "trn_acc": trn_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        pd.DataFrame(log_rows).to_csv(log_dir / "training_log.csv", index=False)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(ckpt, out_dir / "confidence_model.pt")
            log.info("  ↳ Saved best checkpoint (val_loss=%.4f, acc=%.3f)",
                     val_loss, val_acc)

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)
    log.info("Checkpoint: %s", out_dir / "confidence_model.pt")


if __name__ == "__main__":
    main()
