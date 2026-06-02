#!/usr/bin/env python3
"""
confidence_diagnosis.py — Systematic 7-experiment diagnosis of Confidence v2 failure.

E1: gen_ood → gen_ood       within-distribution generalization
E2: bench300 → bench300     in-distribution baseline (clean complex split)
E3: gen_ood → bench300      current failure mode
E4: bench300 → gen_ood      reverse transfer
E5: mixed → bench300/ood    25/50/75% bench300 mixing
E6: memorization test        random-label permutation
E7: feature ceiling          linear probe vs v2 head on both datasets

Feature extraction:
  - Pretrained encoder only, model.eval() throughout (zero BN drift)
  - Features cached to logs/diagnosis/feats_*.pkl
  - Head training is pure CPU on cached 96-dim vectors

Usage:
  conda run -n rapidock python3 scripts/confidence_diagnosis.py --device cuda
"""
from __future__ import annotations
import argparse, copy, json, math, os, pickle, sys, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))
warnings.filterwarnings("ignore")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from utils.diffusion_utils import set_time
from utils.inference_utils import InferenceDataset
from torch_geometric.data import Batch
import MDAnalysis

OUT        = REPO / "logs" / "diagnosis"
PARAMS_YML = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
PRETRAINED = REPO / "third_party" / "RAPiDock" / "train_models" / \
             "CGTensorProductEquivariantModel" / "rapidock_global.pt"
BENCH_JSON = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
BENCH_CSV  = REPO / "data" / "benchmark300.csv"
GEN_JSON   = REPO / "logs" / "confidence_training_data" / "benchmark_results.json"
GEN_CSV    = REPO / "data" / "confidence_training_500.csv"
TMP_BENCH  = "/tmp/diag_bench"
TMP_GEN    = "/tmp/diag_gen"

OUT.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("diagnosis")


# ── head architectures ────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, in_dim=96):
        super().__init__()
        self.w = nn.Linear(in_dim, 1)
    def forward(self, x): return self.w(x)

class SmallHead(nn.Module):
    def __init__(self, in_dim=96, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x)

class V2Head(nn.Module):
    def __init__(self, in_dim=96, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x): return self.net(x)

def make_head(arch: str) -> nn.Module:
    if arch == "linear": return LinearHead()
    if arch == "32":     return SmallHead()
    if arch == "v2":     return V2Head()
    raise ValueError(arch)

ARCHS = ["linear", "32", "v2"]


# ── encoder + feature extraction ─────────────────────────────────────────────

def load_encoder(device: str):
    from models.model import ConfidenceModel
    from argparse import Namespace
    with open(PARAMS_YML) as f:
        params = yaml.safe_load(f)
    params["confidence_mode"] = True
    model = ConfidenceModel(Namespace(**params))
    ckpt = torch.load(PRETRAINED, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    # CRITICAL: eval mode freezes BN running stats — never call model.train()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def _build_base_graphs(json_data: dict, csv_path: Path, tmp_dir: str) -> dict:
    df = pd.read_csv(csv_path)
    names, recs, peps = [], [], []
    for _, row in df.iterrows():
        cname = row["name"]
        if cname not in json_data: continue
        rec = row.get("receptor", "")
        pep = row.get("peptide_pdb", "")
        if not Path(str(rec)).exists() or not Path(str(pep)).exists(): continue
        names.append(cname); recs.append(str(rec)); peps.append(str(pep))
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
            if g is not None: out[n] = g
        except Exception:
            pass
    log.info("Built %d base graphs for %s", len(out), tmp_dir)
    return out


def _load_pose_positions(pdb_path: str, exclude_oxt: bool = False):
    try:
        u = MDAnalysis.Universe(pdb_path)
        positions = []
        for res in u.residues:
            sel = "not type H" + (" and not name OXT" if exclude_oxt else "")
            heavy = res.atoms.select_atoms(sel)
            ca = heavy.select_atoms("name CA")
            if len(ca) == 0 or len(heavy) == 0: continue
            positions.append(heavy.positions.astype(np.float32))
        if not positions: return None
        return torch.tensor(np.concatenate(positions))
    except Exception:
        return None


def _inject_pose(bg, pos):
    g = copy.deepcopy(bg)
    center = pos.mean(0)
    g["pep_a"].pos = pos - center
    if hasattr(g["pep_a"], "x") and g["pep_a"].x is not None:
        g["pep_a"].x = g["pep_a"].x.float()
    g["receptor"].pos = g["receptor"].pos - center
    if hasattr(g["pep_a"], "node_sigma_emb"):
        del g["pep_a"].node_sigma_emb
    return g


@torch.no_grad()
def extract_features(encoder, json_data: dict, base_graphs: dict,
                     device: str, batch_size: int = 32, label: str = "") -> dict:
    """Extract 96-dim encoder features for all poses.

    Returns:
        feat_map: dict[(cname, model_key, pose_idx)] -> np.array[96]
    """
    entries = []
    for cname, model_results in json_data.items():
        bg = base_graphs.get(cname)
        if bg is None: continue
        n_graph = bg["pep_a"].pos.shape[0]
        for mkey, res in model_results.items():
            poses_dir = Path(res["poses_dir"])
            rmsds = res.get("ref_rmsds", [])
            for i, rmsd in enumerate(rmsds):
                entries.append((cname, mkey, poses_dir, i, float(rmsd), n_graph))

    feat_map = {}
    n_ok = 0
    for b_start in range(0, len(entries), batch_size):
        batch = entries[b_start: b_start + batch_size]
        graphs, keys = [], []
        for cname, mkey, poses_dir, i, rmsd, n_graph in batch:
            pdb = poses_dir / f"pose_{i}.pdb"
            if not pdb.exists(): continue
            bg = base_graphs[cname]
            pos = _load_pose_positions(str(pdb))
            if pos is None: continue
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
                keys.append((cname, mkey, i))
            except Exception:
                continue

        if not graphs: continue
        gbatch = Batch.from_data_list(graphs).to(device)

        # Hook the first Linear of confidence_predictor to capture 96-dim input
        feats_captured = []
        def _hook(mod, inp, out):
            feats_captured.append(inp[0].detach().cpu())
        cp = encoder.encoder.confidence_predictor
        first_linear = next(m for m in (cp.net.modules() if hasattr(cp, "net") else cp.modules())
                            if isinstance(m, nn.Linear))
        handle = first_linear.register_forward_hook(_hook)
        try:
            encoder(gbatch)
        except Exception:
            handle.remove()
            continue
        handle.remove()

        if not feats_captured: continue
        feats = feats_captured[0].numpy()  # shape [batch, 96]
        for ki, key in enumerate(keys):
            if ki < len(feats):
                feat_map[key] = feats[ki]
                n_ok += 1

        if b_start % (batch_size * 10) == 0:
            log.info("  %s: %d / %d poses extracted", label, n_ok, len(entries))

    log.info("%s: extracted %d / %d pose features", label, n_ok, len(entries))
    return feat_map


def load_or_extract(encoder, json_data, csv_path, tmp_dir, device, cache_path, label):
    if Path(cache_path).exists():
        log.info("Loading cached features: %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    log.info("Building base graphs for %s...", label)
    base_graphs = _build_base_graphs(json_data, csv_path, tmp_dir)
    log.info("Extracting features for %s...", label)
    feat_map = extract_features(encoder, json_data, base_graphs, device, label=label)
    with open(cache_path, "wb") as f:
        pickle.dump(feat_map, f)
    log.info("Saved feature cache: %s", cache_path)
    return feat_map


# ── dataset helpers ───────────────────────────────────────────────────────────

def build_dataset(feat_map: dict, json_data: dict) -> dict:
    """Build per-complex pose list from feat_map.

    Returns:
        {cname: [(feat_vec, rmsd), ...]}
    """
    ds = {}
    for (cname, mkey, pose_idx), feat in feat_map.items():
        rmsd = json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])[pose_idx] \
               if pose_idx < len(json_data.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])) \
               else None
        if rmsd is None: continue
        if cname not in ds:
            ds[cname] = []
        ds[cname].append((feat, float(rmsd)))
    # Remove complexes with < 2 poses (can't build pairs)
    return {k: v for k, v in ds.items() if len(v) >= 2}


def split_complexes(complexes: list, train_frac: float = 0.85, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(complexes))
    n_train = max(1, int(len(complexes) * train_frac))
    train = [complexes[i] for i in idx[:n_train]]
    val   = [complexes[i] for i in idx[n_train:]]
    return train, val


def build_pairs(ds: dict, complexes: list) -> list:
    """Build (feat_i, feat_j, label) where label=1 iff rmsd_i < rmsd_j."""
    pairs = []
    for cname in complexes:
        poses = ds.get(cname, [])
        if len(poses) < 2: continue
        for (fi, ri), (fj, rj) in combinations(poses, 2):
            if abs(ri - rj) < 1e-6: continue
            label = 1.0 if ri < rj else 0.0
            pairs.append((fi.astype(np.float32), fj.astype(np.float32), label))
    return pairs


def build_pairs_random_labels(ds: dict, complexes: list, seed: int = 99) -> list:
    """Same as build_pairs but labels are randomly permuted within each complex."""
    rng = np.random.RandomState(seed)
    pairs = []
    for cname in complexes:
        poses = ds.get(cname, [])
        if len(poses) < 2: continue
        c_pairs = []
        for (fi, ri), (fj, rj) in combinations(poses, 2):
            if abs(ri - rj) < 1e-6: continue
            c_pairs.append((fi.astype(np.float32), fj.astype(np.float32)))
        # Random labels
        labels = rng.randint(0, 2, len(c_pairs)).astype(float)
        for (fi, fj), lbl in zip(c_pairs, labels):
            pairs.append((fi, fj, lbl))
    return pairs


# ── training ──────────────────────────────────────────────────────────────────

def bpr_loss(si, sj, label):
    margin = (si - sj) * (label * 2.0 - 1.0)
    return -F.logsigmoid(margin).mean()


def _pairs_to_tensors(pairs: list) -> tuple:
    """Convert list of (fi, fj, label) to stacked tensors."""
    fi  = torch.tensor(np.stack([p[0] for p in pairs], axis=0), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in pairs], axis=0), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    return fi, fj, lbl


def train_head(head: nn.Module, train_pairs: list, val_pairs: list,
               epochs: int = 30, lr: float = 1e-3, batch_size: int = 512) -> dict:
    """Train head with BPR loss using mini-batches. Returns final-epoch metrics."""
    if not train_pairs:
        return {"train_acc": float("nan"), "val_acc": float("nan"), "overfit_gap": float("nan")}

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    tr_fi, tr_fj, tr_lbl = _pairs_to_tensors(train_pairs)
    va_fi, va_fj, va_lbl = (_pairs_to_tensors(val_pairs) if val_pairs
                             else (None, None, None))

    def _acc(fi, fj, lbl):
        if fi is None: return float("nan")
        head.eval()
        with torch.no_grad():
            si = head(fi).squeeze(-1)
            sj = head(fj).squeeze(-1)
            pred = (si > sj).float()
        return ((pred - lbl).abs() < 0.5).float().mean().item()

    best_val_acc = -1.0
    best_state = None
    final_trn_acc = float("nan")
    final_val_acc = float("nan")

    n = len(train_pairs)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b: b + batch_size]
            si = head(tr_fi[idx]).squeeze(-1)
            sj = head(tr_fj[idx]).squeeze(-1)
            loss = bpr_loss(si, sj, tr_lbl[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        trn_acc = _acc(tr_fi[:min(2000, n)], tr_fj[:min(2000, n)], tr_lbl[:min(2000, n)])
        val_acc = _acc(va_fi, va_fj, va_lbl)
        if not math.isnan(val_acc) and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
        final_trn_acc = trn_acc
        final_val_acc = val_acc

    if best_state is not None:
        head.load_state_dict(best_state)

    overfit_gap = (final_trn_acc - final_val_acc) \
                  if not (math.isnan(final_trn_acc) or math.isnan(final_val_acc)) \
                  else float("nan")
    return {
        "train_acc": final_trn_acc,
        "val_acc":   final_val_acc,
        "overfit_gap": overfit_gap,
    }


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_tau(head: nn.Module, ds: dict, complexes: list) -> tuple[float, float]:
    """Returns (mean_tau, mean_top1_rmsd) over given complexes."""
    head.eval()
    taus, tops = [], []
    with torch.no_grad():
        for cname in complexes:
            poses = ds.get(cname, [])
            if len(poses) < 2: continue
            feats  = np.array([p[0] for p in poses], dtype=np.float32)
            rmsds  = np.array([p[1] for p in poses], dtype=np.float32)
            scores = head(torch.tensor(feats)).squeeze(-1).numpy()
            tau, _ = scipy_stats.kendalltau(-scores, rmsds)
            if math.isnan(tau): continue
            taus.append(tau)
            top1 = float(rmsds[np.argmax(scores)])
            tops.append(top1)
    mean_tau  = float(np.mean(taus))  if taus else float("nan")
    mean_top1 = float(np.mean(tops))  if tops else float("nan")
    return mean_tau, mean_top1


def run_experiment(name: str, arch: str, train_ds: dict, train_complexes: list,
                   val_complexes: list, eval_ds_bench: dict, eval_bench_complexes: list,
                   eval_ds_gen: dict, eval_gen_complexes: list,
                   epochs: int = 30, random_labels: bool = False) -> dict:
    """Train one head and evaluate on both bench300 and gen_ood eval sets."""
    head = make_head(arch)
    pair_fn = build_pairs_random_labels if random_labels else build_pairs
    train_pairs = pair_fn(train_ds, train_complexes)
    val_pairs   = build_pairs(train_ds, val_complexes)

    log.info("  %s arch=%s  train_pairs=%d  val_pairs=%d",
             name, arch, len(train_pairs), len(val_pairs))

    metrics = train_head(head, train_pairs, val_pairs, epochs=epochs)
    tau_bench, top1_bench = eval_tau(head, eval_ds_bench, eval_bench_complexes)
    tau_gen,   top1_gen   = eval_tau(head, eval_ds_gen,   eval_gen_complexes)

    return {
        "experiment":   name,
        "arch":         arch,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs":  len(val_pairs),
        "tau_bench300": tau_bench,
        "top1_bench300": top1_bench,
        "tau_gen_ood":  tau_gen,
        "top1_gen_ood": top1_gen,
        **metrics,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--epochs",  type=int, default=30)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--no-cache", action="store_true", help="Re-extract features")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    bench_json = json.load(open(BENCH_JSON))
    gen_json   = json.load(open(GEN_JSON))

    # ── 2. Feature extraction (GPU, frozen BN) ────────────────────────────────
    if args.no_cache:
        for p in [OUT / "feats_bench300.pkl", OUT / "feats_gen_ood.pkl"]:
            p.unlink(missing_ok=True)

    log.info("Loading pretrained encoder (eval mode — BN frozen)...")
    encoder = load_encoder(device)

    bench_feats = load_or_extract(
        encoder, bench_json, BENCH_CSV, TMP_BENCH, device,
        OUT / "feats_bench300.pkl", "bench300"
    )
    gen_feats = load_or_extract(
        encoder, gen_json, GEN_CSV, TMP_GEN, device,
        OUT / "feats_gen_ood.pkl", "gen_ood"
    )

    # Free GPU memory — all subsequent work is CPU
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 3. Build per-complex datasets ─────────────────────────────────────────
    bench_ds = build_dataset(bench_feats, bench_json)
    gen_ds   = build_dataset(gen_feats,   gen_json)
    log.info("bench300 complexes with ≥2 poses: %d", len(bench_ds))
    log.info("gen_ood  complexes with ≥2 poses: %d", len(gen_ds))

    bench_complexes = sorted(bench_ds.keys())
    gen_complexes   = sorted(gen_ds.keys())

    bench_train_c, bench_val_c = split_complexes(bench_complexes, 0.85, args.seed)
    gen_train_c,   gen_val_c   = split_complexes(gen_complexes,   0.85, args.seed)

    log.info("bench300 split: %d train / %d val complexes", len(bench_train_c), len(bench_val_c))
    log.info("gen_ood  split: %d train / %d val complexes", len(gen_train_c),   len(gen_val_c))

    rows = []

    # ── E1: gen_ood → gen_ood ─────────────────────────────────────────────────
    log.info("\n=== E1: gen_ood → gen_ood ===")
    for arch in ARCHS:
        r = run_experiment(
            "E1_ood_to_ood", arch,
            train_ds=gen_ds, train_complexes=gen_train_c, val_complexes=gen_val_c,
            eval_ds_bench=bench_ds, eval_bench_complexes=bench_val_c,
            eval_ds_gen=gen_ds,     eval_gen_complexes=gen_val_c,
            epochs=args.epochs,
        )
        rows.append(r)
        log.info("  tau_gen_val=%.4f  tau_bench_val=%.4f  trn_acc=%.3f  val_acc=%.3f",
                 r["tau_gen_ood"], r["tau_bench300"], r["train_acc"], r["val_acc"])

    # ── E2: bench300 → bench300 ───────────────────────────────────────────────
    log.info("\n=== E2: bench300 → bench300 ===")
    for arch in ARCHS:
        r = run_experiment(
            "E2_bench_to_bench", arch,
            train_ds=bench_ds, train_complexes=bench_train_c, val_complexes=bench_val_c,
            eval_ds_bench=bench_ds, eval_bench_complexes=bench_val_c,
            eval_ds_gen=gen_ds,     eval_gen_complexes=gen_val_c,
            epochs=args.epochs,
        )
        rows.append(r)
        log.info("  tau_bench_val=%.4f  tau_gen_val=%.4f  trn_acc=%.3f  val_acc=%.3f",
                 r["tau_bench300"], r["tau_gen_ood"], r["train_acc"], r["val_acc"])

    # ── E3: gen_ood → bench300 ────────────────────────────────────────────────
    log.info("\n=== E3: gen_ood → bench300 (current failure) ===")
    for arch in ARCHS:
        r = run_experiment(
            "E3_ood_to_bench", arch,
            train_ds=gen_ds,   train_complexes=gen_train_c,   val_complexes=gen_val_c,
            eval_ds_bench=bench_ds, eval_bench_complexes=bench_val_c,
            eval_ds_gen=gen_ds,     eval_gen_complexes=gen_val_c,
            epochs=args.epochs,
        )
        rows.append(r)
        log.info("  tau_bench_val=%.4f  trn_acc=%.3f  val_acc=%.3f",
                 r["tau_bench300"], r["train_acc"], r["val_acc"])

    # ── E4: bench300 → gen_ood ────────────────────────────────────────────────
    log.info("\n=== E4: bench300 → gen_ood ===")
    for arch in ARCHS:
        r = run_experiment(
            "E4_bench_to_ood", arch,
            train_ds=bench_ds, train_complexes=bench_train_c, val_complexes=bench_val_c,
            eval_ds_bench=bench_ds, eval_bench_complexes=bench_val_c,
            eval_ds_gen=gen_ds,     eval_gen_complexes=gen_val_c,
            epochs=args.epochs,
        )
        rows.append(r)
        log.info("  tau_gen_val=%.4f  tau_bench_val=%.4f  trn_acc=%.3f  val_acc=%.3f",
                 r["tau_gen_ood"], r["tau_bench300"], r["train_acc"], r["val_acc"])

    # ── E5: mixed training ────────────────────────────────────────────────────
    log.info("\n=== E5: mixed training ===")
    for bench_frac, label in [(0.25, "25B_75G"), (0.50, "50B_50G"), (0.75, "75B_25G")]:
        gen_frac = 1.0 - bench_frac
        # Sample train complexes proportionally
        rng = np.random.RandomState(args.seed)
        n_b = max(1, int(len(bench_train_c) * bench_frac))
        n_g = max(1, int(len(gen_train_c)   * gen_frac))
        b_sample = list(rng.choice(bench_train_c, min(n_b, len(bench_train_c)), replace=False))
        g_sample = list(rng.choice(gen_train_c,   min(n_g, len(gen_train_c)),   replace=False))

        # Build combined dataset (merge dicts, prefix keys to avoid collision)
        combined_ds = {}
        for c in b_sample:
            combined_ds[f"B_{c}"] = bench_ds[c]
        for c in g_sample:
            combined_ds[f"G_{c}"] = gen_ds[c]
        # Val: use bench val for checkpoint selection
        bench_val_ds = {f"B_{c}": bench_ds[c] for c in bench_val_c}

        for arch in ARCHS:
            head = make_head(arch)
            train_pairs = build_pairs(combined_ds, list(combined_ds.keys()))
            val_pairs   = build_pairs(bench_val_ds, list(bench_val_ds.keys()))
            log.info("  E5 %s arch=%s  train_pairs=%d  val_pairs=%d",
                     label, arch, len(train_pairs), len(val_pairs))
            metrics = train_head(head, train_pairs, val_pairs, epochs=args.epochs)
            tau_bench, top1_bench = eval_tau(head, bench_ds, bench_val_c)
            tau_gen,   top1_gen   = eval_tau(head, gen_ds,   gen_val_c)
            rows.append({
                "experiment":    f"E5_{label}",
                "arch":          arch,
                "n_train_pairs": len(train_pairs),
                "n_val_pairs":   len(val_pairs),
                "tau_bench300":  tau_bench,
                "top1_bench300": top1_bench,
                "tau_gen_ood":   tau_gen,
                "top1_gen_ood":  top1_gen,
                **metrics,
            })
            log.info("  tau_bench=%.4f  tau_gen=%.4f  trn_acc=%.3f  val_acc=%.3f",
                     tau_bench, tau_gen, metrics["train_acc"], metrics["val_acc"])

    # ── E6: memorization test (v2 only) ───────────────────────────────────────
    log.info("\n=== E6: memorization test ===")
    for ds_name, ds, train_c, val_c in [
        ("bench300", bench_ds, bench_train_c, bench_val_c),
        ("gen_ood",  gen_ds,   gen_train_c,   gen_val_c),
    ]:
        head = make_head("v2")
        rand_pairs = build_pairs_random_labels(ds, train_c)
        real_pairs = build_pairs(ds, train_c)
        val_rand   = build_pairs_random_labels(ds, val_c)
        val_real   = build_pairs(ds, val_c)

        # Real labels
        metrics_real = train_head(head, real_pairs, val_real, epochs=args.epochs)
        tau_bench_real, _ = eval_tau(head, bench_ds, bench_val_c)
        tau_gen_real,   _ = eval_tau(head, gen_ds,   gen_val_c)

        # Random labels — fresh head
        head_rand = make_head("v2")
        metrics_rand = train_head(head_rand, rand_pairs, val_rand, epochs=args.epochs)
        tau_bench_rand, _ = eval_tau(head_rand, bench_ds, bench_val_c)
        tau_gen_rand,   _ = eval_tau(head_rand, gen_ds,   gen_val_c)

        rows.append({
            "experiment":    f"E6_real_{ds_name}",
            "arch":          "v2",
            "n_train_pairs": len(real_pairs),
            "n_val_pairs":   len(val_real),
            "tau_bench300":  tau_bench_real,
            "top1_bench300": float("nan"),
            "tau_gen_ood":   tau_gen_real,
            "top1_gen_ood":  float("nan"),
            **metrics_real,
        })
        rows.append({
            "experiment":    f"E6_random_{ds_name}",
            "arch":          "v2",
            "n_train_pairs": len(rand_pairs),
            "n_val_pairs":   len(val_rand),
            "tau_bench300":  tau_bench_rand,
            "top1_bench300": float("nan"),
            "tau_gen_ood":   tau_gen_rand,
            "top1_gen_ood":  float("nan"),
            **metrics_rand,
        })
        log.info("  %s REAL:   trn_acc=%.3f  val_acc=%.3f  tau_bench=%.4f  tau_gen=%.4f",
                 ds_name, metrics_real["train_acc"], metrics_real["val_acc"],
                 tau_bench_real, tau_gen_real)
        log.info("  %s RANDOM: trn_acc=%.3f  val_acc=%.3f  tau_bench=%.4f  tau_gen=%.4f",
                 ds_name, metrics_rand["train_acc"], metrics_rand["val_acc"],
                 tau_bench_rand, tau_gen_rand)

    # ── E7: feature ceiling ───────────────────────────────────────────────────
    log.info("\n=== E7: feature ceiling (linear vs v2) ===")
    for ds_name, ds, train_c, val_c in [
        ("bench300", bench_ds, bench_train_c, bench_val_c),
        ("gen_ood",  gen_ds,   gen_train_c,   gen_val_c),
    ]:
        for arch in ["linear", "v2"]:
            r = run_experiment(
                f"E7_{ds_name}", arch,
                train_ds=ds, train_complexes=train_c, val_complexes=val_c,
                eval_ds_bench=bench_ds, eval_bench_complexes=bench_val_c,
                eval_ds_gen=gen_ds,     eval_gen_complexes=gen_val_c,
                epochs=args.epochs,
            )
            rows.append(r)
            log.info("  %s arch=%s  tau_bench=%.4f  tau_gen=%.4f",
                     ds_name, arch, r["tau_bench300"], r["tau_gen_ood"])

    # ── 4. Save results and produce matrix ────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diagnosis_results.csv", index=False)
    log.info("\nSaved: %s/diagnosis_results.csv", OUT)

    _write_report(df, OUT / "diagnosis_matrix.md")
    log.info("Saved: %s/diagnosis_matrix.md", OUT)

    # Print summary table to stdout
    print("\n" + "="*80)
    print("TRANSFER MATRIX — Kendall τ (held-out complexes)")
    print("="*80)
    _print_matrix(df)
    print("\n")
    _print_conclusions(df)


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return "  —  "
    return f"{v:+.3f}"


def _print_matrix(df: pd.DataFrame):
    archs = ARCHS
    print(f"\n{'Architecture':<10}  {'Train':>12}  {'Eval Bench300':>15}  {'Eval Gen_OOD':>14}")
    print("-" * 60)
    for exp_name, train_label, eval_col in [
        ("E1_ood_to_ood",     "gen_ood",   "tau_gen_ood"),
        ("E2_bench_to_bench", "bench300",  "tau_bench300"),
        ("E3_ood_to_bench",   "gen_ood",   "tau_bench300"),
        ("E4_bench_to_ood",   "bench300",  "tau_gen_ood"),
    ]:
        subset = df[df["experiment"] == exp_name]
        for arch in archs:
            row = subset[subset["arch"] == arch]
            if row.empty: continue
            tau = row[eval_col].values[0]
            print(f"  {arch:<8}   {train_label:<12}  →  {_fmt(tau):>14}   (trn={row['train_acc'].values[0]:.2f} val={row['val_acc'].values[0]:.2f})")
        print()


def _print_conclusions(df: pd.DataFrame):
    print("="*80)
    print("CONCLUSIONS")
    print("="*80)

    # Q1: Does gen_ood contain learnable signal?
    e1 = df[(df["experiment"] == "E1_ood_to_ood") & (df["arch"] == "v2")]
    tau_ood_ood = e1["tau_gen_ood"].values[0] if not e1.empty else float("nan")
    e2 = df[(df["experiment"] == "E2_bench_to_bench") & (df["arch"] == "v2")]
    tau_b_b     = e2["tau_bench300"].values[0] if not e2.empty else float("nan")
    print(f"\nQ1: Does gen_ood contain a learnable ranking signal?")
    print(f"    E1 (gen→gen) v2 tau = {_fmt(tau_ood_ood)}   |   E2 (bench→bench) v2 tau = {_fmt(tau_b_b)}")
    if not math.isnan(tau_ood_ood):
        if tau_ood_ood > 0.10:
            print("    → gen_ood HAS learnable signal (tau > 0.10). Problem is transfer, not data quality.")
        elif tau_ood_ood > 0.03:
            print("    → gen_ood has WEAK signal (0.03 < tau < 0.10). Both data quality and transfer are issues.")
        else:
            print("    → gen_ood has ESSENTIALLY NO learnable signal (tau ≤ 0.03). Data is fundamentally flawed.")

    # Q2: Distribution shift vs data quality
    e3 = df[(df["experiment"] == "E3_ood_to_bench") & (df["arch"] == "v2")]
    tau_ood_bench = e3["tau_bench300"].values[0] if not e3.empty else float("nan")
    e4 = df[(df["experiment"] == "E4_bench_to_ood") & (df["arch"] == "v2")]
    tau_b_ood   = e4["tau_gen_ood"].values[0] if not e4.empty else float("nan")
    print(f"\nQ2: Is the main problem data quality or distribution shift?")
    print(f"    E3 (ood→bench) tau = {_fmt(tau_ood_bench)}  |  E4 (bench→ood) tau = {_fmt(tau_b_ood)}")
    if not math.isnan(tau_ood_bench) and not math.isnan(tau_b_b):
        gap = tau_b_b - tau_ood_bench
        print(f"    bench→bench minus ood→bench gap = {gap:+.3f}")
        if gap > 0.15:
            print("    → Large gap: distribution shift is a major factor.")
        else:
            print("    → Small gap: distribution shift is minor; data quality dominates.")

    # Q3: Memorization
    e6_real  = df[(df["experiment"] == "E6_real_gen_ood") & (df["arch"] == "v2")]
    e6_rand  = df[(df["experiment"] == "E6_random_gen_ood") & (df["arch"] == "v2")]
    if not e6_real.empty and not e6_rand.empty:
        trn_real = e6_real["train_acc"].values[0]
        trn_rand = e6_rand["train_acc"].values[0]
        print(f"\nQ3: Is the model memorizing?")
        print(f"    Real labels train acc = {trn_real:.3f}  |  Random labels train acc = {trn_rand:.3f}")
        if trn_rand > 0.60:
            print("    → High random-label train acc: model IS memorizing arbitrary patterns.")
        else:
            print("    → Low random-label train acc: model learns genuine structure, not pure memorization.")

    # Q4: Which dataset for future training
    print(f"\nQ4: Recommended training dataset:")
    if not math.isnan(tau_b_b) and not math.isnan(tau_ood_ood):
        if tau_b_b > tau_ood_ood + 0.05:
            print(f"    → bench300 (tau {_fmt(tau_b_b)} vs gen_ood {_fmt(tau_ood_ood)})")
        elif tau_ood_ood > tau_b_b + 0.05:
            print(f"    → gen_ood (tau {_fmt(tau_ood_ood)} vs bench300 {_fmt(tau_b_b)})")
        else:
            print(f"    → Similar (bench={_fmt(tau_b_b)}, gen={_fmt(tau_ood_ood)}); mixing may help")

    # Q5: Does mixing help
    mix_rows = df[df["experiment"].str.startswith("E5_")]
    if not mix_rows.empty:
        print(f"\nQ5: Does mixing datasets help?")
        print(f"    {'Mix':>12}  {'arch':>6}  {'τ bench':>10}  {'τ gen':>10}")
        for _, row in mix_rows[mix_rows["arch"] == "v2"].iterrows():
            print(f"    {row['experiment']:>12}  {row['arch']:>6}  {_fmt(row['tau_bench300']):>10}  {_fmt(row['tau_gen_ood']):>10}")

    # Q6: Best clean generalization estimate
    print(f"\nQ6: Best clean held-out generalization estimate (v2 head):")
    best_row = None
    best_tau = -99.0
    for _, row in df[df["arch"] == "v2"].iterrows():
        tau_b = row["tau_bench300"]
        if not math.isnan(tau_b) and tau_b > best_tau:
            best_tau = tau_b
            best_row = row
    if best_row is not None:
        print(f"    {best_row['experiment']}  →  τ_bench={_fmt(best_row['tau_bench300'])}  τ_gen={_fmt(best_row['tau_gen_ood'])}")


def _write_report(df: pd.DataFrame, path: Path):
    lines = ["# Confidence Model Diagnosis Report\n",
             f"Date: 2026-06-01  |  n(bench300)={df['n_train_pairs'].max()}  |  Epochs=30\n\n"]

    lines.append("## Transfer Matrix — Kendall τ on held-out complexes\n\n")
    lines.append("| Experiment | Arch | Train Data | τ Bench300 | τ Gen_OOD | Train Acc | Val Acc | Overfit Gap |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['arch']} | "
            f"{_fmt(row['tau_bench300'])} | {_fmt(row['tau_gen_ood'])} | "
            f"{row['train_acc']:.3f} | {row['val_acc']:.3f} | {row['overfit_gap']:.3f} |\n"
        )

    lines.append("\n## 2×2 Transfer Matrix (v2 head)\n\n")
    lines.append("```\n")
    lines.append(f"              Eval Bench300   Eval Gen_OOD\n")

    def _get(exp, col):
        r = df[(df["experiment"] == exp) & (df["arch"] == "v2")]
        return r[col].values[0] if not r.empty else float("nan")

    bb = _get("E2_bench_to_bench", "tau_bench300")
    bg = _get("E4_bench_to_ood",   "tau_gen_ood")
    gb = _get("E3_ood_to_bench",   "tau_bench300")
    gg = _get("E1_ood_to_ood",     "tau_gen_ood")

    lines.append(f"Train Bench300   {_fmt(bb):>13}   {_fmt(bg):>12}\n")
    lines.append(f"Train Gen_OOD    {_fmt(gb):>13}   {_fmt(gg):>12}\n")
    lines.append("```\n\n")

    with open(path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
