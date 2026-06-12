#!/usr/bin/env python3
"""
pepset_benchmark.py — 4-method comparison on PepSet (new dataset, n=189).

PepSet complexes are completely disjoint from bench300 (different PDB IDs,
different proteins, different peptide lengths). Clean OOD evaluation.

Methods compared:
  ref2015      PyRosetta ref2015 score-only (lower = better)
  confidence_v1 Trained ConfidenceModel checkpoint (higher = better)
  e0_head      Cached-feature head trained on bench300 (higher = better)
  ensemble     0.6 × conf_v1_norm + 0.4 × (−ref2015_norm)

Pipeline per complex:
  1. RAPiDock inference → 20 poses (via run_rapidock.py subprocess)
  2. Cα RMSD vs crystal reference
  3. All 4 scoring methods
  4. Kendall τ per method

Usage (rapidock env):
  /home/igem/miniconda3/envs/rapidock/bin/python3 -u scripts/pepset_benchmark.py \
      --n-complexes 50 --n-samples 20 --out-dir logs/pepset_benchmark
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import pickle
import re
import subprocess
import sys
import time
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock"))

PEPSET_DIR  = REPO / "datasets" / "pepset"
PRETRAINED  = REPO / "third_party" / "RAPiDock" / "train_models" / \
              "CGTensorProductEquivariantModel" / "rapidock_global.pt"
PARAMS_YML  = REPO / "train_models" / "confidence_model" / "model_parameters.yml"
CONF_CKPT   = REPO / "train_models" / "confidence_model" / "confidence_model.pt"
FEAT_BENCH  = REPO / "logs" / "diagnosis" / "feats_bench300.pkl"
BENCH_JSON  = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
RUN_RAPIDOCK = REPO / "src" / "hybridock_pep" / "sampling" / "run_rapidock.py"
RAPIDOCK_PY  = REPO / "third_party" / "RAPiDock"
MODEL_DIR    = RAPIDOCK_PY / "train_models" / "CGTensorProductEquivariantModel"
RAPIDOCK_ENV_PY = "/home/igem/miniconda3/envs/rapidock/bin/python3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
for _h in logging.root.handlers:
    if hasattr(_h, "stream"):
        _h.stream = os.fdopen(os.dup(_h.stream.fileno()), "w", buffering=1)
log = logging.getLogger("pepset_bench")


# ── V2Head (no input LayerNorm — bench features have correct scale already) ──

class V2Head(nn.Module):
    def __init__(self, in_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(),           nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


def bpr_loss(si, sj, label):
    return -F.logsigmoid((si - sj) * (label * 2.0 - 1.0)).mean()


# ── E0 head training ──────────────────────────────────────────────────────────

def train_e0_head() -> V2Head:
    """Train E0 head on bench300 cached features, return best checkpoint."""
    with open(FEAT_BENCH, "rb") as f: bench_feats = pickle.load(f)
    bench_json = json.load(open(BENCH_JSON))

    ds = {}
    for (cname, mkey, pi), feat in bench_feats.items():
        rmsds = bench_json.get(cname, {}).get(mkey, {}).get("ref_rmsds", [])
        if pi >= len(rmsds): continue
        ds.setdefault(cname, []).append((feat.astype(np.float32), float(rmsds[pi])))
    ds = {k: v for k, v in ds.items() if len(v) >= 2}

    rng  = np.random.RandomState(42)
    keys = sorted(ds.keys())
    idx  = rng.permutation(len(keys)); n = int(0.85 * len(keys))
    train_c = [keys[i] for i in idx[:n]]; val_c = [keys[i] for i in idx[n:]]

    pairs = []
    for c in train_c:
        for (fi, ri), (fj, rj) in combinations(ds[c], 2):
            if abs(ri - rj) < 1e-6: continue
            pairs.append((fi, fj, 1.0 if ri < rj else 0.0))

    fi  = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj  = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lbl = torch.tensor([p[2] for p in pairs],           dtype=torch.float32)
    # val pairs for early stopping
    vpairs = []
    for c in val_c:
        for (fi2, ri), (fj2, rj) in combinations(ds[c], 2):
            if abs(ri - rj) < 1e-6: continue
            vpairs.append((fi2, fj2, 1.0 if ri < rj else 0.0))
    vfi = torch.tensor(np.stack([p[0] for p in vpairs]), dtype=torch.float32)
    vfj = torch.tensor(np.stack([p[1] for p in vpairs]), dtype=torch.float32)

    head = V2Head(); torch.manual_seed(0)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc, best_state = -1.0, None; n_pairs = len(pairs)

    for ep in range(50):
        head.train()
        perm = torch.randperm(n_pairs)
        for b in range(0, n_pairs, 512):
            idx2 = perm[b: b+512]
            loss = bpr_loss(head(fi[idx2]).squeeze(-1), head(fj[idx2]).squeeze(-1), lbl[idx2])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(vfi).squeeze(-1) > head(vfj).squeeze(-1)).float().mean().item()
        if acc > best_acc: best_acc = acc; best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    # quick eval τ
    from scipy import stats as sp
    taus = []
    head.eval()
    with torch.no_grad():
        for c in val_c:
            poses = ds[c]
            feats  = torch.tensor(np.array([p[0] for p in poses], dtype=np.float32))
            rmsds  = np.array([p[1] for p in poses])
            scores = head(feats).squeeze(-1).numpy()
            tau, _ = sp.kendalltau(-scores, rmsds)
            if not math.isnan(tau): taus.append(tau)
    log.info("  E0 head (bench300 val) τ=%.4f", np.mean(taus) if taus else float("nan"))
    return head


# ── RAPiDock inference via subprocess ────────────────────────────────────────

def run_inference(name: str, seq: str, receptor: str, pep_pdb: str,
                  cx_dir: Path, n_samples: int, seed: int) -> list[Path]:
    """Run RAPiDock inference via run_rapidock.py shim. Returns list of pose PDBs."""
    poses_dir = cx_dir / "poses"
    existing  = sorted(poses_dir.glob("pose_*.pdb"))
    if len(existing) >= n_samples:
        return existing

    raw_out = str((cx_dir / "poses_raw").resolve())
    cmd = [
        RAPIDOCK_ENV_PY, str(RUN_RAPIDOCK),
        "--peptide",         seq,
        "--receptor",        str(Path(receptor).resolve()),
        "--output-dir",      raw_out,
        "--n-samples",       str(n_samples),
        "--rapidock-dir",    str(RAPIDOCK_PY.resolve()),
        "--model-dir",       str(MODEL_DIR.resolve()),
        "--ckpt",            "rapidock_global.pt",
        "--scoring-function","none",
        "--seed",            str(seed),
    ]
    log.debug("  cmd: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log.warning("  inference failed (rc=%d): %s", result.returncode,
                    result.stderr[-500:] if result.stderr else "")
        return []

    # Rename rank*.pdb → pose_N.pdb (same as rapidock_runner.py)
    raw_dir  = cx_dir / "poses_raw" / "poses_raw"
    poses_dir.mkdir(parents=True, exist_ok=True)
    rank_files = sorted(raw_dir.glob("rank*.pdb"),
                        key=lambda p: int(re.search(r"rank(\d+)", p.stem).group(1)))
    pose_paths = []
    for i, rf in enumerate(rank_files):
        dest = poses_dir / f"pose_{i}.pdb"
        rf.rename(dest)
        pose_paths.append(dest)

    log.info("  inference: %d poses", len(pose_paths))
    return pose_paths


# ── RMSD computation ──────────────────────────────────────────────────────────

def compute_rmsds(pose_paths: list[Path], ref_pdb: str) -> list[float]:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    rmsds = []
    try:
        ref_u  = mda.Universe(ref_pdb)
        ref_ca = ref_u.select_atoms("name CA")
    except Exception: return [float("nan")] * len(pose_paths)

    for p in pose_paths:
        try:
            u  = mda.Universe(str(p))
            ca = u.select_atoms("name CA")
            if len(ca) != len(ref_ca): rmsds.append(float("nan")); continue
            rmsds.append(float(rms.rmsd(ca.positions, ref_ca.positions, superposition=True)))
        except Exception: rmsds.append(float("nan"))
    return rmsds


# ── scoring methods ───────────────────────────────────────────────────────────

def score_ref2015(pose_paths: list[Path], receptor: str) -> list[float]:
    try:
        import pyrosetta
        pyrosetta.init(" ".join(["-mute", "all", "-ignore_unrecognized_res",
                                 "-use_input_sc", "-ex1", "-ex2aro"]))
        sfxn = pyrosetta.create_score_function("ref2015")
        import tempfile
        scores = []
        for p in pose_paths:
            try:
                import tempfile as tf
                with tf.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                    tmp.write(open(receptor).read().encode())
                    tmp.write(b"TER\n")
                    tmp.write(open(str(p), "rb").read())
                    tmp_path = tmp.name
                pose = pyrosetta.pose_from_pdb(tmp_path)
                scores.append(float(sfxn(pose)))
                os.unlink(tmp_path)
            except Exception: scores.append(float("nan"))
        return scores
    except ImportError:
        log.debug("PyRosetta unavailable — ref2015 NaN")
        return [float("nan")] * len(pose_paths)


def _load_pose_positions(pdb: str):
    import MDAnalysis as mda
    try:
        u = mda.Universe(pdb); pos = []
        for res in u.residues:
            heavy = res.atoms.select_atoms("not type H")
            if not heavy.select_atoms("name CA").n_atoms: continue
            pos.append(heavy.positions.astype(np.float32))
        return torch.tensor(np.concatenate(pos)) if pos else None
    except Exception: return None


def _inject_pose_bg(bg, pos):
    from utils.diffusion_utils import set_time
    g = copy.deepcopy(bg); c = pos.mean(0)
    g["pep_a"].pos    = pos - c
    g["pep_a"].x      = g["pep_a"].x.float()
    g["receptor"].pos = g["receptor"].pos - c
    if hasattr(g["pep_a"], "node_sigma_emb"): del g["pep_a"].node_sigma_emb
    set_time(g, 0., 0., 0., 0., 1, device="cpu")
    return g


def score_confidence_v1(pose_paths: list[Path], bg, device: str) -> list[float]:
    if not CONF_CKPT.exists():
        return [float("nan")] * len(pose_paths)
    try:
        import yaml
        from models.model import ConfidenceModel
        from argparse import Namespace
        from torch_geometric.data import Batch

        with open(PARAMS_YML) as f: params = yaml.safe_load(f)
        params["confidence_mode"] = True
        model = ConfidenceModel(Namespace(**params))
        ckpt  = torch.load(CONF_CKPT, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval().to(device)
        n_g = bg["pep_a"].pos.shape[0]

        scores = []
        for p in pose_paths:
            pos = _load_pose_positions(str(p))
            if pos is None or pos.shape[0] != n_g:
                scores.append(float("nan")); continue
            try:
                g = _inject_pose_bg(bg, pos)
                batch = Batch.from_data_list([g]).to(device)
                with torch.no_grad(): scores.append(float(model(batch).item()))
                del batch; torch.cuda.empty_cache()
            except Exception: scores.append(float("nan")); torch.cuda.empty_cache()
        return scores
    except Exception as e:
        log.warning("conf_v1 scoring error: %s", e)
        return [float("nan")] * len(pose_paths)


def score_e0_head(pose_paths: list[Path], bg, head: V2Head,
                  encoder, device: str) -> list[float]:
    """Extract features via encoder hook, then score with E0 head."""
    from torch_geometric.data import Batch

    n_g = bg["pep_a"].pos.shape[0]
    cp  = encoder.encoder.confidence_predictor
    first_lin = next(m for m in (cp.net.modules() if hasattr(cp,"net") else cp.modules())
                     if isinstance(m, nn.Linear))
    captured = []
    def _hook(m, inp, out): captured.append(inp[0].detach().cpu())
    handle = first_lin.register_forward_hook(_hook)

    head.eval(); scores = []
    for p in pose_paths:
        pos = _load_pose_positions(str(p))
        if pos is None or pos.shape[0] != n_g:
            scores.append(float("nan")); continue
        try:
            g = _inject_pose_bg(bg, pos)
            captured.clear()
            batch = Batch.from_data_list([g]).to(device)
            with torch.no_grad(): encoder(batch)
            del batch; torch.cuda.empty_cache()
            if captured:
                with torch.no_grad():
                    scores.append(float(head(captured[0].float()).squeeze(-1).item()))
            else: scores.append(float("nan"))
        except Exception: scores.append(float("nan")); torch.cuda.empty_cache()

    handle.remove()
    return scores


def kendall_tau(scores, rmsds) -> float:
    from scipy import stats as sp
    valid = [(s, r) for s, r in zip(scores, rmsds)
             if not math.isnan(float(s)) and not math.isnan(float(r))]
    if len(valid) < 2: return float("nan")
    s = np.array([v[0] for v in valid])
    r = np.array([v[1] for v in valid])
    tau, _ = sp.kendalltau(-s, r)
    return float(tau)


def top1_rmsd(scores, rmsds) -> float:
    valid = [(i, s) for i, s in enumerate(scores) if not math.isnan(float(s))]
    if not valid: return float("nan")
    best_i = max(valid, key=lambda x: x[1])[0]
    return float(rmsds[best_i]) if not math.isnan(float(rmsds[best_i])) else float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-complexes", type=int, default=50)
    ap.add_argument("--n-samples",   type=int, default=20)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--out-dir",     default="logs/pepset_benchmark")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--skip-done",   action="store_true")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device=%s  n_cx=%d  n_samples=%d", device, args.n_complexes, args.n_samples)

    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build pepset CSV ───────────────────────────────────────────────────
    rows = []
    for pd_dir in sorted(PEPSET_DIR.iterdir()):
        if not pd_dir.is_dir(): continue
        pdb = pd_dir.name
        rec = pd_dir / f"{pdb}_rec_unbound_pocket.pdb"
        pep = pd_dir / f"{pdb}_pep_ref.pdb"
        sf  = pd_dir / f"{pdb}_peptide_sequence"
        if not (rec.exists() and pep.exists() and sf.exists()): continue
        seq = sf.read_text().strip()
        if not seq or len(seq) < 4 or len(seq) > 30: continue   # skip degenerate lengths
        rows.append({"name": pdb, "receptor": str(rec), "peptide_pdb": str(pep), "seq": seq})
        if len(rows) >= args.n_complexes: break
    log.info("%d pepset complexes selected", len(rows))

    # ── 2. Load models ────────────────────────────────────────────────────────
    log.info("Loading pretrained encoder...")
    import yaml
    from models.model import ConfidenceModel
    from argparse import Namespace as NS
    from utils.inference_utils import InferenceDataset
    with open(PARAMS_YML) as f: params = yaml.safe_load(f)
    params["confidence_mode"] = True
    encoder = ConfidenceModel(NS(**params))
    ckpt = torch.load(PRETRAINED, map_location="cpu")
    encoder.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    encoder.eval().to(device)

    log.info("Training E0 head on bench300...")
    head_e0 = train_e0_head()

    # ── 3. Per-complex benchmark ──────────────────────────────────────────────
    results = []
    t0 = time.time()

    for i, row in enumerate(rows):
        name = row["name"]; seq = row["seq"]
        receptor = row["receptor"]; pep_pdb = row["peptide_pdb"]
        cx_dir = out_dir / name; cx_dir.mkdir(exist_ok=True)
        result_path = cx_dir / "result.json"

        if args.skip_done and result_path.exists():
            r = json.load(open(result_path))
            results.append(r); log.info("[%d/%d] %s — cached", i+1, len(rows), name)
            continue

        log.info("[%d/%d] %s (seq=%s len=%d)", i+1, len(rows), name, seq, len(seq))

        # Build base graph
        tmp = str(cx_dir / "tmp_base"); os.makedirs(tmp, exist_ok=True)
        try:
            ds_b = InferenceDataset(
                output_dir=tmp, complex_name_list=[name],
                protein_description_list=[receptor],
                peptide_description_list=[pep_pdb],
                lm_embeddings=True, lm_embeddings_pep=False,
                conformation_type=None, conformation_partial="1:1:1",
            )
            bg = ds_b.get(0)
            if bg is None: log.warning("  base graph failed — skip"); continue
        except Exception as e:
            log.warning("  base graph error: %s — skip", e); continue

        # RAPiDock inference
        pose_paths = run_inference(name, seq, receptor, pep_pdb, cx_dir,
                                   args.n_samples, args.seed)
        if len(pose_paths) < 2:
            log.warning("  too few poses — skip"); continue

        # RMSD
        rmsds = compute_rmsds(pose_paths, pep_pdb)
        n_valid = sum(1 for r in rmsds if not math.isnan(r))
        if n_valid < 2:
            log.warning("  too few valid RMSDs — skip"); continue
        log.info("  RMSDs: n=%d mean=%.2f min=%.2f",
                 n_valid, np.nanmean(rmsds), np.nanmin(rmsds))

        # Scoring
        s_ref  = score_ref2015(pose_paths, receptor)
        s_conf = score_confidence_v1(pose_paths, bg, device)
        s_e0   = score_e0_head(pose_paths, bg, head_e0, encoder, device)

        # Ensemble: normalise each method, blend
        def _norm(arr):
            a = np.array(arr, dtype=float)
            valid = ~np.isnan(a)
            if valid.sum() < 2: return np.full_like(a, float("nan"))
            a[valid] = (a[valid] - a[valid].mean()) / (a[valid].std() + 1e-8)
            return a
        ref_n  = _norm([-s for s in s_ref])   # lower ref2015 = better → flip
        conf_n = _norm(s_conf)
        both   = ~np.isnan(ref_n) & ~np.isnan(conf_n)
        s_ens  = np.full(len(pose_paths), float("nan"))
        if both.sum() > 1:
            s_ens[both] = 0.6 * conf_n[both] + 0.4 * ref_n[both]

        r = {
            "name": name, "n_poses": len(pose_paths), "n_valid_rmsd": n_valid,
            "min_rmsd": float(np.nanmin(rmsds)), "mean_rmsd": float(np.nanmean(rmsds)),
            "tau_ref2015":   kendall_tau(s_ref,        rmsds),
            "tau_conf_v1":   kendall_tau(s_conf,       rmsds),
            "tau_e0_head":   kendall_tau(s_e0,         rmsds),
            "tau_ensemble":  kendall_tau(s_ens.tolist(),rmsds),
            "top1_ref2015":  top1_rmsd([-x for x in s_ref], rmsds),
            "top1_conf_v1":  top1_rmsd(s_conf, rmsds),
            "top1_e0_head":  top1_rmsd(s_e0,   rmsds),
        }
        json.dump(r, open(result_path, "w"), indent=2)
        results.append(r)

        elapsed = time.time() - t0
        eta_min = elapsed / (i+1) * (len(rows)-i-1) / 60
        log.info("  τ ref2015=%.3f conf_v1=%.3f e0=%.3f ens=%.3f  ETA=%.0f min",
                 r["tau_ref2015"], r["tau_conf_v1"], r["tau_e0_head"],
                 r["tau_ensemble"], eta_min)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    if not results:
        log.error("No results — check pipeline"); return

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    log.info("\n" + "="*60)
    log.info("PEPSET BENCHMARK — %d complexes", len(df))
    log.info("="*60)
    log.info("  %-22s  %8s  %8s  %10s", "Method", "τ (mean)", "τ (std)", "top1 Å")
    log.info("  " + "-"*52)
    for tcol, t1col in [("tau_ref2015","top1_ref2015"), ("tau_conf_v1","top1_conf_v1"),
                         ("tau_e0_head","top1_e0_head"), ("tau_ensemble","tau_ensemble")]:
        tv = df[tcol].dropna()
        t1v = df[t1col].dropna() if t1col in df.columns else pd.Series(dtype=float)
        t1_str = f"{t1v.mean():.3f}" if len(t1v) else "n/a"
        log.info("  %-22s  %8.4f  %8.4f  %10s", tcol, tv.mean() if len(tv) else float("nan"),
                 tv.std() if len(tv) else 0, t1_str)

    log.info("="*60)
    log.info("Results: %s", out_dir / "results.csv")


if __name__ == "__main__":
    main()
