#!/usr/bin/env python3
"""
pose_contact_ranker.py — Routes 1 + 2 pose ranking experiments.

Route 1 (ContactMLP):
  Pose-specific Cβ contact geometry → RMSD regression (Huber loss).
  For each pose: K-nearest receptor Cβ distances + contact counts at 4/6/8Å,
  pooled mean/max/std over peptide residues → 69-dim fixed vector.
  These features VARY per pose; burial/encoder barely do.

Route 2 (EncMLP-Reg):
  Same 96-dim ESM2 encoder features as retrain_ranker_n100.py BUT with
  RMSD regression instead of BPR pairwise ranking.
  Directly tests whether BPR was the loss bottleneck (enc τ was 0.053).

Blend:
  z(-contact_pred) + z(-enc_pred) + z(burial)
  Also: contact-only vs burial baseline vs BSA proxy.

5-fold CV on gen_n100 (57 complexes, 100 poses each).
Baseline reference from retrain_ranker_n100.py:
  burial τ = +0.123 ±0.084, burial+enc(BPR) τ = +0.116 ±0.060
  ref2015 top-1 RMSD = 4.59Å Hit@2Å = 10.5%

Usage (rapidock env):
  python3 scripts/pose_contact_ranker.py           # extract + CV
  python3 scripts/pose_contact_ranker.py --skip-extract  # skip extraction
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

D    = REPO / "logs" / "diagnosis"
OUT  = REPO / "logs" / "training_campaign"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")

GEN_N100_JSON    = REPO / "logs" / "gen_n100" / "benchmark_results.json"
GEN_N100_ENC     = D / "feats_gen_n100.pkl"
GEN_N100_PHYS    = D / "feats_gen_n100_physics.pkl"
CONTACT_FEAT_PKL = D / "feats_gen_n100_contact.pkl"

K_NEIGHBORS  = 20        # top-K nearest receptor Cβ per peptide residue
CUTOFF_A     = 12.0      # distance cutoff for receptor interface
EPOCHS       = 120
SEEDS        = (0, 1, 2)
FOLDS        = 5
BURIAL_IDX   = [0, 2, 1]  # fa_atr, fa_sol, fa_rep indices in phys vector
BURIAL_SIGN  = np.array([-1.0, 1.0, 1.0])
CONTACT_DIM  = 3 * (K_NEIGHBORS + 3)  # 69
ENC_DIM      = 96


# ── helpers ───────────────────────────────────────────────────────────────────

def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _cx_z(M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), M.std(0)
    return (M - mu) / np.where(sd < 1e-9, 1.0, sd)


# ── Cβ / Cα extraction from PDB ───────────────────────────────────────────────

def _read_cb(pdb_path: str) -> np.ndarray:
    """Read Cβ coordinates (Cα for Gly). Returns [N, 3] float32 array."""
    coords: list[list[float]] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            res_name  = line[17:20].strip()
            want = "CA" if res_name == "GLY" else "CB"
            if atom_name != want:
                continue
            try:
                coords.append([float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])])
            except ValueError:
                continue
    if coords:
        return np.array(coords, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


# ── Contact feature extraction ────────────────────────────────────────────────

def contact_feats(pose_pdb: str, rec_pdb: str) -> np.ndarray | None:
    """
    69-dim pose-specific contact feature vector.

    For each peptide residue: distances to K nearest receptor Cβ atoms +
    contact counts at 4 / 6 / 8 Å cutoffs. Pooled over peptide residues
    with mean, max, std → shape (3 * (K+3),) = (69,).

    Returns None if Cβ extraction fails (< 2 pep residues or < 4 rec residues).
    """
    pep = _read_cb(pose_pdb)
    rec = _read_cb(rec_pdb)
    if len(pep) < 2 or len(rec) < 4:
        return None

    D = cdist(pep, rec)  # [n_pep, n_rec]

    # K nearest receptor neighbours
    n_keep = min(D.shape[1], K_NEIGHBORS)
    knn = np.sort(D, axis=1)[:, :n_keep]
    if n_keep < K_NEIGHBORS:
        pad = np.full((len(pep), K_NEIGHBORS - n_keep), CUTOFF_A, dtype=np.float32)
        knn = np.concatenate([knn, pad], axis=1)

    # Contact counts (normalised)
    c4 = (D < 4.0).sum(axis=1, keepdims=True).astype(np.float32) / 10.0
    c6 = (D < 6.0).sum(axis=1, keepdims=True).astype(np.float32) / 20.0
    c8 = (D < 8.0).sum(axis=1, keepdims=True).astype(np.float32) / 30.0

    per_res = np.concatenate(
        [knn / CUTOFF_A, c4, c6, c8], axis=1
    )  # [n_pep, K+3]

    feat = np.concatenate([
        per_res.mean(axis=0),
        per_res.max(axis=0),
        per_res.std(axis=0),
    ]).astype(np.float32)  # [69]

    return feat


def extract_all_contact(bjson: dict, enc_all: dict) -> dict:
    """Extract contact features for all keys in enc_all and cache to pkl."""
    keys = sorted(enc_all.keys())
    rec_cache: dict[str, bool] = {}
    results: dict = {}
    n_ok = n_fail = 0
    t0 = time.time()
    print(f"Extracting contact features for {len(keys)} poses...", flush=True)

    for i, k in enumerate(keys):
        cn, mk, pi = k
        entry    = bjson.get(cn, {}).get(mk, {})
        pose_pdb = str(Path(entry.get("poses_dir", "")) / f"pose_{pi}.pdb")
        rec_pdb  = str(BASE / cn / f"{cn}_protein_pocket.pdb")

        if not Path(pose_pdb).exists() or not Path(rec_pdb).exists():
            results[k] = None
            n_fail += 1
            continue

        feat = contact_feats(pose_pdb, rec_pdb)
        if feat is not None:
            results[k] = feat
            n_ok += 1
        else:
            results[k] = None
            n_fail += 1

        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(keys) - i - 1)
            print(f"  {i+1}/{len(keys)}  ok={n_ok} fail={n_fail}  "
                  f"{el:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"Done. {n_ok} ok  {n_fail} fail  {(time.time()-t0):.1f}s", flush=True)
    return results


# ── Pool loader ───────────────────────────────────────────────────────────────

def load_pool(contact_all: dict, enc_all: dict,
              phys_all: dict, bjson: dict) -> dict:
    pool: dict = {}
    n_miss = 0
    for k, cv in contact_all.items():
        cn, mk, pi = k
        ev = enc_all.get(k)
        pv = phys_all.get(k)
        if cv is None or ev is None:
            n_miss += 1
            continue
        rr = bjson.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(rr):
            n_miss += 1
            continue
        pool.setdefault(cn, []).append({
            "contact": np.asarray(cv, np.float32),
            "enc":     np.asarray(ev, np.float32),
            "phys":    np.asarray(pv, np.float32) if pv is not None
                       else np.zeros(16, np.float32),
            "rmsd":    float(rr[pi]),
            "pi":      pi,
        })
    pool = {c: v for c, v in pool.items() if len(v) >= 10}
    print(f"Pool: {len(pool)} complexes  ({n_miss} entries skipped)")
    return pool


# ── MLP training (lazy torch) ─────────────────────────────────────────────────

def _train_mlp(X: np.ndarray, y: np.ndarray,
               seed: int, hidden: int = 64,
               epochs: int = EPOCHS) -> object:
    """
    Train a small MLP for RMSD regression (Huber loss).
    X: [N, in_dim] per-complex z-normalised features.
    y: [N] RMSD values in Å.
    Returns eval-mode model.
    """
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(seed)

    in_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Dropout(0.30),
        nn.Linear(hidden, 32), nn.GELU(),
        nn.Dropout(0.20),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    N  = len(Xt)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for b in range(0, N, 128):
            idx = perm[b: b + 128]
            loss = nn.functional.huber_loss(model(Xt[idx]), yt[idx], delta=2.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                vl = nn.functional.huber_loss(model(Xt), yt).item()
            print(f"    ep{ep+1}/{epochs}  huber={vl:.4f}", flush=True)
            model.train()

    return model.eval()


def _score_mlp(model, X: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


# ── CV engine ─────────────────────────────────────────────────────────────────

def cv(pool: dict) -> tuple[dict, dict]:
    """5-fold CV.  Returns (cx_results, fold_taus)."""
    cxs  = sorted(pool)
    rng  = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    folds = [[cxs[i] for i in perm[f::FOLDS]] for f in range(FOLDS)]

    cx_results: dict = {}
    fold_taus: dict = {
        "burial":       [],
        "enc_bpr":      [],   # reference from earlier run (not recomputed here)
        "enc_reg":      [],   # Route 2: encoder + RMSD regression
        "contact":      [],   # Route 1: contact geometry + RMSD regression
        "blend":        [],   # Route 1 + Route 2 + burial z-blend
    }

    for fi in range(FOLDS):
        val_cxs = folds[fi]
        tr_cxs  = [c for c in cxs if c not in set(val_cxs)]
        print(f"\nFold {fi+1}/{FOLDS}  train={len(tr_cxs)}  val={len(val_cxs)}", flush=True)

        # Build training arrays
        X_contact_tr, X_enc_tr, y_tr = [], [], []
        for c in tr_cxs:
            rows = pool[c]
            C = _cx_z(np.array([r["contact"] for r in rows]))
            E = _cx_z(np.array([r["enc"]     for r in rows]))
            y = np.array([r["rmsd"] for r in rows])
            X_contact_tr.append(C)
            X_enc_tr.append(E)
            y_tr.append(y)
        X_contact_tr = np.vstack(X_contact_tr)
        X_enc_tr     = np.vstack(X_enc_tr)
        y_tr         = np.concatenate(y_tr)

        cx_seed: dict = {c: {} for c in val_cxs}
        tau_acc: dict = {k: [] for k in fold_taus}

        for sd in SEEDS:
            print(f"  seed {sd}:", flush=True)

            # Route 1: contact RMSD regression
            print(f"    training ContactMLP...", flush=True)
            m_contact = _train_mlp(X_contact_tr, y_tr, seed=sd,
                                   hidden=64, epochs=EPOCHS)

            # Route 2: encoder RMSD regression
            print(f"    training EncMLP-Reg...", flush=True)
            m_enc = _train_mlp(X_enc_tr, y_tr, seed=sd,
                               hidden=64, epochs=EPOCHS)

            for c in val_cxs:
                rows  = pool[c]
                rmsd  = np.array([r["rmsd"]    for r in rows])
                C     = _cx_z(np.array([r["contact"] for r in rows]))
                E     = _cx_z(np.array([r["enc"]     for r in rows]))
                P     = np.array([r["phys"]    for r in rows])

                # burial baseline (from phys)
                burial = _cx_z(P[:, BURIAL_IDX]) @ BURIAL_SIGN

                # contact prediction (lower predicted RMSD = better pose)
                contact_pred = _score_mlp(m_contact, C)  # [N] predicted RMSD
                enc_pred     = _score_mlp(m_enc,     E)  # [N] predicted RMSD

                # blend: negate preds so higher = better, then z-sum with burial
                blend = _z(-contact_pred) + _z(-enc_pred) + _z(burial)

                # per-complex τ (use negative pred so higher rank = lower RMSD)
                scores = {
                    "burial":  burial,
                    "enc_reg": -enc_pred,
                    "contact": -contact_pred,
                    "blend":   blend,
                }
                for name, score in scores.items():
                    t, _ = sp.kendalltau(-score if name != "burial" else -score, rmsd)
                    # Note: τ > 0 means lower score → lower RMSD (correctly signed)
                    if not math.isnan(t):
                        tau_acc[name].append(t)

                # Top-k metrics
                for name, score in scores.items():
                    srt = np.argsort(-score)
                    for tk in [1, 5, 10, 25]:
                        cx_seed[c].setdefault(f"{name}_top{tk}", []).append(
                            float(rmsd[srt[:tk]].min()))

                # RAPiDock top-1 + oracle
                p0 = next((r["rmsd"] for r in rows if r["pi"] == 0), rmsd[0])
                cx_seed[c].setdefault("rapd_top1", []).append(float(p0))
                cx_seed[c].setdefault("oracle",    []).append(float(rmsd.min()))

        # Average over seeds → one value per complex
        for c in val_cxs:
            cx_results[c] = {k: float(np.mean(vs))
                             for k, vs in cx_seed[c].items()}

        for name in ["burial", "enc_reg", "contact", "blend"]:
            fold_taus[name].append(float(np.mean(tau_acc[name])))

        print(f"\n  Fold {fi+1} τ:")
        for name, tv in fold_taus.items():
            if tv:
                print(f"    {name:<15} τ = {tv[-1]:+.4f}")
        print(flush=True)

    return cx_results, fold_taus


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(cx_results: dict, fold_taus: dict) -> None:
    N = len(cx_results)
    print(f"\n{'='*72}")
    print(f"CONTACT RANKER CV  ({N} complexes, {FOLDS}-fold, {len(SEEDS)} seeds)")
    print(f"{'='*72}")

    print(f"\nKendall τ (mean ± std over folds):")
    print(f"  {'Strategy':<18}  {'τ mean':>8}  {'τ std':>8}")
    print(f"  {'-'*38}")
    # reference numbers from retrain_ranker_n100
    ref_baseline = {"burial": (0.1227, 0.0838),
                    "enc_bpr": (0.0534, 0.0188)}
    for name, tv in fold_taus.items():
        if not tv:
            ref = ref_baseline.get(name)
            tag = f"  ← ref: {ref[0]:+.4f} ±{ref[1]:.4f}" if ref else "  (not computed)"
            print(f"  {name:<18}  {tag}")
        else:
            mean_t, std_t = float(np.mean(tv)), float(np.std(tv))
            ref = ref_baseline.get(name)
            delta = f"  Δ={mean_t-ref[0]:+.4f} vs ref" if ref else ""
            print(f"  {name:<18}  {mean_t:+.4f}  {std_t:.4f}{delta}")

    print(f"\nTop-k RMSD (mean over {N} held-out complexes):")
    print(f"  {'Metric':<22}  {'burial':>8}  {'enc_reg':>8}  {'contact':>8}  {'blend':>8}  {'oracle':>8}")
    print(f"  {'-'*66}")

    oracle_top1 = np.mean([cx_results[c]["oracle"] for c in cx_results])
    rapd_top1   = np.mean([cx_results[c]["rapd_top1"] for c in cx_results])
    print(f"  {'RAPiDock pose_0':<22}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}  {oracle_top1:>6.2f}Å")
    print(f"    (top-1 RMSD = {rapd_top1:.2f}Å)")

    for tk in [1, 5, 10, 25]:
        strats = ["burial", "enc_reg", "contact", "blend"]
        vals = {s: np.array([cx_results[c][f"{s}_top{tk}"] for c in cx_results])
                for s in strats}
        orc  = np.array([cx_results[c]["oracle"] for c in cx_results])

        row_rmsd = "  " + f"Mean RMSD  top-{tk:<2}  "
        row_hit  = "  " + f"  Hit@2Å   top-{tk:<2}  "
        for s in strats:
            v = vals[s]
            row_rmsd += f"  {v.mean():>6.2f}Å"
            row_hit  += f"  {100*np.mean(v<=2.0):>6.1f}%"
        row_rmsd += f"  {orc.mean():>6.2f}Å"
        row_hit  += f"  {100*np.mean(orc<=2.0):>6.1f}%"
        print(row_rmsd)
        print(row_hit)
        print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip contact feature extraction (reuse cached pkl)")
    a = ap.parse_args()

    bjson    = json.load(open(GEN_N100_JSON))
    enc_all  = pickle.load(open(GEN_N100_ENC,  "rb"))
    phys_all = pickle.load(open(GEN_N100_PHYS, "rb"))

    # ── Step 1: contact features ──────────────────────────────────────────────
    if a.skip_extract and CONTACT_FEAT_PKL.exists():
        print(f"Loading cached contact features from {CONTACT_FEAT_PKL}", flush=True)
        contact_all = pickle.load(open(CONTACT_FEAT_PKL, "rb"))
    else:
        contact_all = extract_all_contact(bjson, enc_all)
        pickle.dump(contact_all, open(CONTACT_FEAT_PKL, "wb"), protocol=4)
        print(f"Saved → {CONTACT_FEAT_PKL}")

    # ── Step 2: pool ──────────────────────────────────────────────────────────
    print("\nLoading pool...", flush=True)
    pool = load_pool(contact_all, enc_all, phys_all, bjson)
    if len(pool) < 5:
        print("ERROR: too few complexes with all features present.")
        sys.exit(1)

    # ── Step 3: CV ────────────────────────────────────────────────────────────
    cx_results, fold_taus = cv(pool)

    # ── Step 4: summary ───────────────────────────────────────────────────────
    print_summary(cx_results, fold_taus)

    results = {c: cx_results[c] for c in cx_results}
    results["_meta"] = {
        "n_complexes": len(pool), "folds": FOLDS,
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "contact_dim": CONTACT_DIM, "enc_dim": ENC_DIM,
        "k_neighbors": K_NEIGHBORS, "cutoff_A": CUTOFF_A,
        "loss": "huber_delta2",
    }
    out_path = OUT / "contact_ranker_cv.json"
    import json as _json
    out_path.write_text(_json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
