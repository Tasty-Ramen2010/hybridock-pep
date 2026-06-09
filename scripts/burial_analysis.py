#!/usr/bin/env python3
"""
burial_analysis.py — Interrogate the ref2015 "burial axis" and test three
hypotheses for breaking past the physics ranking ceiling:

  (1) slim+signed per-complex-normalized physics head  (burial ceiling)
  (2) per-complex-normalized fusion of physics + encoder
  (3) geometric-consensus features (encoder-space + Cα), the only candidate
      orthogonal signal

Mechanism probes:
  - PCA of the 4 collinear energy terms (is it really one axis?)
  - U-shape test (is native-ness non-monotone in burial?)
  - consensus↔burial orthogonality

All on existing pkls + on-disk pose PDBs. CPU, no GPU. 5-fold CV, held-out
complexes, trained MLP head w/ BPR. Restricted to the 115 complexes that have
physics features, so every condition is apples-to-apples.
"""
from __future__ import annotations
import json, math, pickle, random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
D = REPO / "logs" / "diagnosis"
BJSON = json.load(open(REPO / "logs" / "analysis_bench300" / "benchmark_results.json"))

TERMS = ["fa_atr","fa_rep","fa_sol","fa_intra_rep","fa_elec","hbond_bb_sc","hbond_sc",
         "hbond_lr_bb","hbond_sr_bb","rama_prepro","fa_dun","p_aa_pp","interface_ddG",
         "total_score","resp_delta_e","resp_ca_disp"]
BURIAL_IDX = [0, 2, 1]      # fa_atr, fa_sol, fa_rep
BURIAL_SIGN = np.array([-1.0, 1.0, 1.0])   # fa_atr anti-correlated → flip

torch.set_num_threads(4)


# ── load ────────────────────────────────────────────────────────────────────
def load():
    phys = pickle.load(open(D / "feats_bench300_physics.pkl", "rb"))
    enc  = pickle.load(open(D / "feats_bench300.pkl", "rb"))
    pool = {}   # cname -> list of dict(phys, enc, rmsd, key)
    for k, pv in phys.items():
        cn, mk, pi = k
        r = BJSON.get(cn, {}).get(mk, {}).get("ref_rmsds", [])
        if pi >= len(r) or k not in enc:
            continue
        pool.setdefault(cn, []).append(
            {"phys": np.asarray(pv, np.float64),
             "enc":  np.asarray(enc[k], np.float64),
             "rmsd": float(r[pi]), "key": k})
    pool = {c: v for c, v in pool.items() if len(v) >= 3}
    return pool


def ca_coords(pdb: Path) -> np.ndarray | None:
    cs = []
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA":
            cs.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return np.asarray(cs) if cs else None


def add_ca_consensus(pool: dict):
    """For each complex, consensus_i = mean pairwise peptide Cα-RMSD (no align,
    shared receptor frame) of pose i to the other poses. Lower = more central."""
    for cn, rows in pool.items():
        coords = []
        for r in rows:
            cnr, mk, pi = r["key"]
            pdb = BJSON[cnr][mk]["poses_dir"]
            c = ca_coords(Path(pdb) / f"pose_{pi}.pdb")
            coords.append(c)
        n = len(rows)
        L = min(c.shape[0] for c in coords if c is not None)
        valid = [c[:L] if c is not None else None for c in coords]
        for i, r in enumerate(rows):
            if valid[i] is None:
                r["consensus"] = np.nan; continue
            ds = []
            for j in range(n):
                if i == j or valid[j] is None:
                    continue
                ds.append(np.sqrt(np.mean(np.sum((valid[i] - valid[j])**2, axis=1))))
            r["consensus"] = float(np.mean(ds)) if ds else np.nan


def per_cx_z(vals: np.ndarray) -> np.ndarray:
    mu, sd = vals.mean(0), vals.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (vals - mu) / sd


# ── mechanism probes ──────────────────────────────────────────────────────────
def mechanism(pool: dict):
    print("\n" + "="*70)
    print("MECHANISM: is 'burial' really one axis?")
    print("="*70)
    # PCA of the 4 collinear terms, per complex, avg variance explained by PC1
    pc1_var = []
    burial_taus, u_fracs = [], []
    for cn, rows in pool.items():
        M = np.array([r["phys"][[0,2,1,13]] for r in rows])   # atr,sol,rep,total
        if M.shape[0] < 3 or np.any(M.std(0) < 1e-9):
            continue
        Mz = per_cx_z(M)
        u, s, vt = np.linalg.svd(Mz - Mz.mean(0), full_matrices=False)
        pc1_var.append((s[0]**2) / (s**2).sum())
        # signed burial coordinate
        b = per_cx_z(np.array([r["phys"][BURIAL_IDX] for r in rows])) @ BURIAL_SIGN
        rmsd = np.array([r["rmsd"] for r in rows])
        t, _ = sp.kendalltau(b, rmsd)
        if not math.isnan(t):
            burial_taus.append(t)
        # U-shape probe: split poses into low/mid/high burial terciles, mean rmsd
        order = np.argsort(b)
        k = max(1, len(order)//3)
        lo, hi = rmsd[order[:k]].mean(), rmsd[order[-k:]].mean()
        mid = rmsd[order[k:-k]].mean() if len(order) > 2*k else (lo+hi)/2
        # U-shape if mid < both ends (native in the middle of burial)
        u_fracs.append(1.0 if (mid < lo and mid < hi) else 0.0)
    print(f"  PC1 variance explained (mean):  {np.mean(pc1_var):.3f}  "
          f"→ {'ONE dominant axis' if np.mean(pc1_var)>0.7 else 'multi-axis'}")
    print(f"  signed-burial coordinate τ:     {np.mean(burial_taus):+.4f}")
    print(f"  U-shape fraction (native=mid):  {np.mean(u_fracs):.2f}  "
          f"→ {'non-monotone, transform may help' if np.mean(u_fracs)>0.4 else 'monotone, transform wont help'}")


def consensus_probe(pool: dict):
    print("\n" + "="*70)
    print("CONSENSUS: orthogonal to burial? does it rank?")
    print("="*70)
    enc_taus, ca_taus, orth = [], [], []
    for cn, rows in pool.items():
        rmsd = np.array([r["rmsd"] for r in rows])
        # encoder-space consensus: mean cosine distance to other poses
        E = np.array([r["enc"] for r in rows])
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sim = En @ En.T
        np.fill_diagonal(sim, np.nan)
        enc_cons = np.nanmean(1 - sim, axis=1)   # high = atypical
        if np.std(enc_cons) > 1e-9:
            t, _ = sp.kendalltau(enc_cons, rmsd)
            if not math.isnan(t): enc_taus.append(t)
        # Cα consensus
        cc = np.array([r.get("consensus", np.nan) for r in rows])
        if np.sum(~np.isnan(cc)) >= 3 and np.nanstd(cc) > 1e-9:
            t, _ = sp.kendalltau(cc[~np.isnan(cc)], rmsd[~np.isnan(cc)])
            if not math.isnan(t): ca_taus.append(t)
            # orthogonality to burial
            b = per_cx_z(np.array([r["phys"][BURIAL_IDX] for r in rows])) @ BURIAL_SIGN
            m = ~np.isnan(cc)
            if m.sum() >= 3 and np.std(b[m]) > 1e-9:
                c, _ = sp.spearmanr(cc[m], b[m])
                if not math.isnan(c): orth.append(abs(c))
    print(f"  encoder-space consensus τ:  {np.mean(enc_taus):+.4f}  (n={len(enc_taus)})")
    print(f"  Cα consensus τ:             {np.mean(ca_taus):+.4f}  (n={len(ca_taus)})")
    print(f"  |Cα-consensus ↔ burial| ρ:  {np.mean(orth):.3f}  "
          f"→ {'ORTHOGONAL (new signal!)' if np.mean(orth)<0.4 else 'redundant with burial'}")


# ── CV head ─────────────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,64), nn.LayerNorm(64), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(64,1))
    def forward(self, x): return self.net(x)


def feat_for(rows, mode):
    """Return (n, d) per-complex-normalized feature matrix for a condition."""
    P = np.array([r["phys"] for r in rows])
    E = np.array([r["enc"] for r in rows])
    cc = np.array([r.get("consensus", np.nan) for r in rows]).reshape(-1,1)
    cc = np.nan_to_num(cc, nan=np.nanmean(cc) if np.any(~np.isnan(cc)) else 0.0)
    parts = []
    if "physslim" in mode:
        b = P[:, BURIAL_IDX] * BURIAL_SIGN
        parts.append(per_cx_z(b))
    if "phys14" in mode:
        parts.append(per_cx_z(P[:, :14]))
    if "enc" in mode:
        parts.append(per_cx_z(E))
    if "cons" in mode:
        parts.append(per_cx_z(cc))
    return np.concatenate(parts, 1)


def make_pairs(pool, cxs, mode):
    pairs = []
    for c in cxs:
        rows = pool[c]
        Fm = feat_for(rows, mode)
        r = np.array([x["rmsd"] for x in rows])
        for i, j in combinations(range(len(r)), 2):
            if abs(r[i]-r[j]) > 1e-6:
                pairs.append((Fm[i], Fm[j], 1.0 if r[i] < r[j] else 0.0))
    return pairs


def cv(pool, mode, ens=False, epochs=50, seeds=(0,1,2), folds=5):
    cxs = sorted(pool)
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    fold = [[cxs[i] for i in perm[f::folds]] for f in range(folds)]
    taus = []
    for fi in range(folds):
        val = fold[fi]; tr = [c for c in cxs if c not in set(val)]
        seed_taus = []
        for sd in seeds:
            if ens:
                # train two heads (physslim, enc), z-blend outputs per complex
                heads = {}
                for m in ("physslim", "enc"):
                    pr = make_pairs(pool, tr, m)
                    heads[m] = train(pr, feat_for(pool[tr[0]], m).shape[1], sd, epochs)
                ts = []
                for c in val:
                    rows = pool[c]; rmsd = np.array([x["rmsd"] for x in rows])
                    s = np.zeros(len(rows))
                    for m in ("physslim","enc"):
                        sc = score(heads[m], feat_for(rows, m))
                        s = s + (sc - sc.mean())/(sc.std()+1e-9)
                    t,_ = sp.kendalltau(-s, rmsd)
                    if not math.isnan(t): ts.append(t)
                seed_taus.append(np.mean(ts))
            else:
                pr = make_pairs(pool, tr, mode)
                h = train(pr, feat_for(pool[tr[0]], mode).shape[1], sd, epochs)
                ts = []
                for c in val:
                    rows = pool[c]; rmsd = np.array([x["rmsd"] for x in rows])
                    sc = score(h, feat_for(rows, mode))
                    t,_ = sp.kendalltau(-sc, rmsd)
                    if not math.isnan(t): ts.append(t)
                seed_taus.append(np.mean(ts))
        taus.append(np.mean(seed_taus))
    return float(np.mean(taus)), float(np.std(taus))


def train(pairs, d, seed, epochs):
    torch.manual_seed(seed)
    h = Head(d)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
    fi = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    fj = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    lb = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    n = len(pairs)
    for ep in range(epochs):
        h.train()
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b:b+512]
            loss = -F.logsigmoid((h(fi[idx]).squeeze(-1)-h(fj[idx]).squeeze(-1))*(lb[idx]*2-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return h


def score(h, Fm):
    h.eval()
    with torch.no_grad():
        return h(torch.tensor(Fm, dtype=torch.float32)).squeeze(-1).numpy()


def main():
    print("Loading pools + computing Cα consensus (115 cx, ~2300 poses)...")
    pool = load()
    add_ca_consensus(pool)
    print(f"  {len(pool)} complexes")

    mechanism(pool)
    consensus_probe(pool)

    print("\n" + "="*70)
    print("5-FOLD CV (held-out complexes, per-cx-normalized inputs, 115-cx set)")
    print("="*70)
    configs = [
        ("phys14 (current static)",      "phys14",            False),
        ("physslim (signed burial)",     "physslim",          False),
        ("encoder",                      "enc",               False),
        ("physslim + encoder (fused)",   "physslim+enc",      False),
        ("physslim + enc + consensus",   "physslim+enc+cons", False),
        ("output-ensemble (current best)","physslim",         True),
    ]
    results = {}
    for name, mode, ens in configs:
        m, s = cv(pool, mode, ens=ens)
        results[name] = m
        print(f"  {name:<34} τ = {m:+.4f} ± {s:.3f}")

    best = max(results, key=results.get)
    print(f"\n  BEST: {best}  τ={results[best]:+.4f}")
    print(f"  (production combined14_CLIP ≈ 0.20, prior ensemble ≈ 0.236)")
    json.dump(results, open(REPO/"logs"/"training_campaign"/"burial_analysis.json","w"), indent=2)


if __name__ == "__main__":
    main()
