#!/usr/bin/env python3
"""
contact_gnn_ranker.py — Pose-conditioned contact GNN for ranking.

THE HYPOTHESIS (Routes 1+2 diagnosis):
  ContactMLP failed (τ≈0) because anonymous distance features can't
  generalise — a 4Å contact means different chemistry in different
  complexes. The pooled ESM encoder failed (τ≈0.05) because it's
  near-constant across the 100 poses of one complex.

  FIX: a GNN on the POSE-SPECIFIC contact graph with CHEMISTRY-AWARE
  node features. Node features (AA identity / ESM) are static, but
  WHICH residues are in contact changes per pose → output varies per
  pose AND knows the chemistry. This is DiffDock's confidence head idea
  applied to our N=100 homogeneous data.

  Peptide-internal geometry edges are included so the GNN can see the
  peptide's own conformation (the "great interface, wrong tail" case
  that decouples global RMSD from interface energy).

EXPERIMENT LADDER (--node-feat):
  onehot : 21-dim AA one-hot   (tests "does residue identity help?")
  esm    : ESM-2 650M per-res  (tests "does evolutionary context add?")
           requires feats_gen_n100_esm.pkl (extract_esm_n100.py)

Baseline to beat:  ref2015 / BSA+clash  τ = +0.14,  top-1 Hit@2Å = 10.5%
Oracle ceiling:    τ = 1.0,             top-1 Hit@2Å = 49.1%

Run in rapidock env (GPU):
  python3 scripts/contact_gnn_ranker.py --node-feat onehot
  python3 scripts/contact_gnn_ranker.py --node-feat esm
"""
from __future__ import annotations

import argparse, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy import stats as sp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

D    = REPO / "logs" / "diagnosis"
OUT  = REPO / "logs" / "training_campaign"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")

GEN_N100_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
GEN_N100_ENC  = D / "feats_gen_n100.pkl"
ESM_PKL       = D / "feats_gen_n100_esm.pkl"

CONTACT_CUT  = 8.0     # Å for cross / internal contact edges
PEP_SEQ_EDGE = True    # add sequential peptide backbone edges
FOLDS        = 5
SEEDS        = (0, 1, 2)
EPOCHS       = 60
HIDDEN       = 48
RBF_N        = 16
RBF_MAX      = 12.0

AA3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_IDX = {a: i for i, a in enumerate(AA3)}
N_AA = len(AA3)  # 20; index 20 = unknown


# ── PDB reader: per-residue (resname, Cβ coord) in order ──────────────────────

def read_residues(pdb_path: str) -> tuple[list[str], np.ndarray]:
    """Return (resnames, coords[N,3]) using Cβ (Cα for Gly)."""
    res_atoms: dict[tuple, dict] = {}
    order: list[tuple] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in ("CA", "CB"):
                continue
            resname = line[17:20].strip()
            chain   = line[21]
            resseq  = line[22:27]  # include insertion code
            key = (chain, resseq)
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            if key not in res_atoms:
                res_atoms[key] = {"name": resname}
                order.append(key)
            res_atoms[key][atom] = xyz

    names, coords = [], []
    for key in order:
        a = res_atoms[key]
        xyz = a.get("CB") or a.get("CA")
        if xyz is None:
            continue
        names.append(a["name"])
        coords.append(xyz)
    return names, np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), np.float32)


def aa_onehot(names: list[str]) -> np.ndarray:
    M = np.zeros((len(names), N_AA + 1), dtype=np.float32)
    for i, n in enumerate(names):
        M[i, AA_IDX.get(n, N_AA)] = 1.0
    return M


def rbf(d: np.ndarray) -> np.ndarray:
    centers = np.linspace(0.0, RBF_MAX, RBF_N, dtype=np.float32)
    width = (RBF_MAX / RBF_N)
    return np.exp(-((d[:, None] - centers[None, :]) ** 2) / (2 * width ** 2)).astype(np.float32)


# ── Build per-complex static data + per-pose graphs ───────────────────────────

def build_pool(node_feat: str) -> dict:
    """
    Returns pool[cn] = {
        "pep_names","rec_names","rec_coords",
        "pep_feat","rec_feat",                # node features [n,F]
        "poses": [ {"pep_coords","rmsd","pi"} ... ]
    }
    """
    import torch
    bjson   = json.load(open(GEN_N100_JSON))
    enc_all = pickle.load(open(GEN_N100_ENC, "rb"))
    cxs     = sorted(set(k[0] for k in enc_all))

    esm = None
    if node_feat == "esm":
        if not ESM_PKL.exists():
            print(f"ERROR: {ESM_PKL} missing. Run extract_esm_n100.py first.")
            sys.exit(1)
        esm = pickle.load(open(ESM_PKL, "rb"))

    pool: dict = {}
    t0 = time.time()
    for ci, cn in enumerate(cxs):
        mk = "pretrained"
        entry = bjson.get(cn, {}).get(mk, {})
        rr = entry.get("ref_rmsds", [])
        poses_dir = Path(entry.get("poses_dir", ""))
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        if not rec_pdb.exists() or len(rr) < 10:
            continue

        rec_names, rec_coords = read_residues(str(rec_pdb))
        if len(rec_names) < 4:
            continue

        # peptide identity from pose_0 (same sequence across poses)
        p0 = poses_dir / "pose_0.pdb"
        if not p0.exists():
            continue
        pep_names, _ = read_residues(str(p0))
        if len(pep_names) < 2:
            continue

        # node features
        if node_feat == "onehot":
            pep_feat = aa_onehot(pep_names)
            rec_feat = aa_onehot(rec_names)
        else:  # esm
            ed = esm.get(cn)
            if ed is None:
                continue
            pep_feat = np.asarray(ed["pep"], np.float32)
            rec_feat = np.asarray(ed["rec"], np.float32)
            # guard length mismatch (rare; truncate to min)
            if pep_feat.shape[0] != len(pep_names):
                m = min(pep_feat.shape[0], len(pep_names))
                pep_feat, pep_names = pep_feat[:m], pep_names[:m]
            if rec_feat.shape[0] != len(rec_names):
                m = min(rec_feat.shape[0], len(rec_names))
                rec_feat, rec_names, rec_coords = rec_feat[:m], rec_names[:m], rec_coords[:m]

        # per-pose peptide coords + rmsd
        poses = []
        for pi in range(len(rr)):
            pp = poses_dir / f"pose_{pi}.pdb"
            if not pp.exists():
                continue
            _, pc = read_residues(str(pp))
            if pc.shape[0] != len(pep_names):
                m = min(pc.shape[0], len(pep_names))
                pc = pc[:m]
            poses.append({"pep_coords": pc, "rmsd": float(rr[pi]), "pi": pi})
        if len(poses) < 10:
            continue

        pool[cn] = {
            "pep_names": pep_names, "rec_names": rec_names,
            "rec_coords": rec_coords,
            "pep_feat": pep_feat, "rec_feat": rec_feat,
            "poses": poses,
        }
        if (ci + 1) % 10 == 0:
            print(f"  built {ci+1}/{len(cxs)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"Pool: {len(pool)} complexes, "
          f"{sum(len(v['poses']) for v in pool.values())} poses "
          f"({time.time()-t0:.0f}s)", flush=True)
    return pool


def make_graph(cx: dict, pose: dict):
    """Build a PyG Data object for one pose."""
    import torch
    from torch_geometric.data import Data

    pep_feat = cx["pep_feat"]; rec_feat = cx["rec_feat"]
    pep_xyz  = pose["pep_coords"]; rec_xyz = cx["rec_coords"]
    P = pep_feat.shape[0]; R = rec_feat.shape[0]
    Pc = min(P, pep_xyz.shape[0]); P = Pc
    pep_feat = pep_feat[:P]; pep_xyz = pep_xyz[:P]

    x = np.concatenate([pep_feat, rec_feat], axis=0)            # [P+R, F]
    is_pep = np.zeros(P + R, dtype=np.float32); is_pep[:P] = 1.0

    src, dst, eattr = [], [], []

    # cross edges: peptide i — receptor j  (pose-specific!)
    dmat = np.sqrt(((pep_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1))  # [P,R]
    pi, rj = np.where(dmat < CONTACT_CUT)
    for a, b in zip(pi, rj):
        d = dmat[a, b]
        src += [a, P + b]; dst += [P + b, a]
        eattr += [d, d]  # symmetric

    # peptide internal edges: sequential + spatial contacts
    pdm = np.sqrt(((pep_xyz[:, None, :] - pep_xyz[None, :, :]) ** 2).sum(-1))
    for a in range(P):
        for b in range(a + 1, P):
            if (PEP_SEQ_EDGE and b == a + 1) or pdm[a, b] < CONTACT_CUT:
                d = pdm[a, b]
                src += [a, b]; dst += [b, a]; eattr += [d, d]

    # receptor internal edges (static context)
    rdm = np.sqrt(((rec_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1))
    ri, rk = np.where(rdm < CONTACT_CUT)
    for a, b in zip(ri, rk):
        if a < b:
            d = rdm[a, b]
            src += [P + a, P + b]; dst += [P + b, P + a]; eattr += [d, d]

    if not src:  # degenerate; self-loop
        src, dst, eattr = [0], [0], [0.0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(rbf(np.array(eattr, np.float32)), dtype=torch.float32)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index, edge_attr=edge_attr,
        is_pep=torch.tensor(is_pep, dtype=torch.float32),
        y=torch.tensor([pose["rmsd"]], dtype=torch.float32),
    )
    return data


# ── GNN model ─────────────────────────────────────────────────────────────────

def build_model(in_dim: int):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

    class GNN(nn.Module):
        def __init__(self, in_dim, hidden=HIDDEN, n_layers=3):
            super().__init__()
            self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
            self.edge_proj = nn.Linear(RBF_N, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(n_layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden))
                self.convs.append(GINEConv(mlp, edge_dim=hidden, train_eps=True))
                self.norms.append(nn.LayerNorm(hidden))
            self.drop = nn.Dropout(0.30)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(),
                nn.Dropout(0.20), nn.Linear(hidden, 1))

        def forward(self, data):
            x = self.in_proj(data.x)
            ea = self.edge_proj(data.edge_attr)
            for conv, norm in zip(self.convs, self.norms):
                h = conv(x, data.edge_index, ea)
                x = x + self.drop(norm(h).relu())
            # pool peptide nodes only
            pep_mask = data.is_pep.bool()
            batch = data.batch
            xm = global_mean_pool(x[pep_mask], batch[pep_mask])
            xx = global_max_pool(x[pep_mask], batch[pep_mask])
            return self.head(torch.cat([xm, xx], dim=-1)).squeeze(-1)

    return GNN(in_dim)


# ── loss config ───────────────────────────────────────────────────────────────
# LOSS determines training objective AND how predictions map to a ranking score.
#   huber  : regress RMSD            → score = -pred (lower predicted RMSD better)
#   whuber : RMSD-weighted Huber     → score = -pred  (weight=exp(-rmsd/2),
#            near-native poses dominate the gradient → targets top-k / Hit@2Å)
#   bce    : near-native (<=2Å) BCE  → score = +pred (higher logit = near-native)
#            pos_weight handles the 3.7% positive imbalance
LOSS = "huber"          # set by --loss in main()
NEAR_NATIVE = 2.0       # Å threshold for bce target


def _compute_loss(pred, y):
    import torch
    import torch.nn.functional as F
    if LOSS == "huber":
        return F.huber_loss(pred, y, delta=2.0)
    if LOSS == "whuber":
        w = torch.exp(-y / 2.0)                       # near-native poses weighted up
        per = F.huber_loss(pred, y, delta=2.0, reduction="none")
        return (w * per).sum() / (w.sum() + 1e-9)
    if LOSS == "bce":
        target = (y <= NEAR_NATIVE).float()
        frac = target.mean().clamp(1e-3, 1 - 1e-3)
        pos_weight = (1 - frac) / frac                # up-weight rare positives
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
    raise ValueError(LOSS)


def _pred_to_score(preds: np.ndarray) -> np.ndarray:
    """Map model output to a ranking score where HIGHER = better pose."""
    if LOSS in ("huber", "whuber"):
        return -preds                                 # predicted RMSD; lower better
    return preds                                      # bce logit; higher better


# ── train / eval one fold (GPU pre-batched, no DataLoader collation) ──────────

def train_eval(pool, tr_cxs, val_cxs, graphs, seed):
    import torch
    from torch_geometric.data import Batch
    torch.set_num_threads(8)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    in_dim = graphs[tr_cxs[0]][0].x.shape[1]
    model = build_model(in_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Pre-batch the whole train set onto GPU ONCE (avoids per-epoch CPU collation,
    # which was the bottleneck for 1280-dim ESM features).
    train_graphs = [g for c in tr_cxs for g in graphs[c]]
    rng_local = np.random.RandomState(seed)
    N = len(train_graphs)
    big = Batch.from_data_list(train_graphs).to(dev)
    y_all = big.y

    # mini-batch by slicing pre-built per-graph batches is awkward on a merged
    # Batch; instead chunk the graph list into fixed GPU-resident sub-batches.
    CHUNK = 512
    chunks = []
    order0 = list(range(N))
    for s in range(0, N, CHUNK):
        sub = [train_graphs[i] for i in order0[s:s + CHUNK]]
        chunks.append(Batch.from_data_list(sub).to(dev))

    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        perm = rng_local.permutation(len(chunks))
        for ci in perm:
            b = chunks[ci]
            pred = model(b)
            loss = _compute_loss(pred, b.y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"    ep{ep+1}/{EPOCHS} {LOSS}={tot/len(chunks):.4f}", flush=True)

    # eval per complex (each complex = one GPU batch)
    model.eval()
    res = {}
    with torch.no_grad():
        for c in val_cxs:
            gl = graphs[c]
            b = Batch.from_data_list(gl).to(dev)
            preds = model(b).cpu().numpy()
            rmsd  = np.array([g.y.item() for g in gl])
            res[c] = (preds, rmsd, [p["pi"] for p in pool[c]["poses"]])
    return res


# ── main ─────────────────────────────────────────────────────────────────────

def learning_curve(pool, graphs):
    """
    Does held-out τ rise as we add training complexes?
    Fixed held-out test set; train on increasing subsets of the rest.
    Answers: is complex-count the bottleneck (→ generate more) or not (→ ceiling)?
    """
    cxs = sorted(pool)
    rng = np.random.RandomState(13)
    perm = list(rng.permutation(cxs))
    test_cxs = perm[:15]                 # fixed held-out
    train_pool_cxs = perm[15:]           # 42 available for training
    sizes = [10, 20, 30, 42]
    print(f"\nLEARNING CURVE  (fixed test={len(test_cxs)}, train pool={len(train_pool_cxs)})")
    print(f"  {'N_train':>8}  {'τ mean':>8}  {'τ std':>8}")
    print(f"  {'-'*30}")
    curve = []
    for n in sizes:
        seed_taus = []
        for sd in SEEDS:
            tr = train_pool_cxs[:n]
            res = train_eval(pool, tr, test_cxs, graphs, sd)
            taus = []
            for c in test_cxs:
                preds, rmsd, _ = res[c]
                t, _ = sp.kendalltau(_pred_to_score(preds), -rmsd)
                if not math.isnan(t):
                    taus.append(t)
            seed_taus.append(float(np.mean(taus)))
        m, s = float(np.mean(seed_taus)), float(np.std(seed_taus))
        curve.append({"n": n, "tau": m, "std": s})
        print(f"  {n:>8}  {m:+.4f}  {s:.4f}", flush=True)
    # verdict
    slope = curve[-1]["tau"] - curve[0]["tau"]
    print(f"\n  Δτ ({sizes[0]}→{sizes[-1]} complexes) = {slope:+.4f}")
    if slope > 0.03 and curve[-1]["tau"] >= curve[-2]["tau"] - 0.01:
        print("  → RISING & not plateaued: MORE COMPLEXES should help. Generate.")
    elif slope > 0.03:
        print("  → rising but flattening: marginal gains from more data.")
    else:
        print("  → FLAT: complex count is NOT the lever. True ceiling. Don't generate.")
    (OUT / "gnn_learning_curve.json").write_text(json.dumps(curve, indent=2))
    print(f"  Saved → {OUT/'gnn_learning_curve.json'}")
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-feat", choices=["onehot", "esm"], default="onehot")
    ap.add_argument("--loss", choices=["huber", "whuber", "bce"], default="huber",
                    help="training objective: huber (full-list), whuber "
                         "(near-native-weighted), bce (near-native classify)")
    ap.add_argument("--learning-curve", action="store_true",
                    help="run learning curve instead of full CV")
    a = ap.parse_args()

    global LOSS
    LOSS = a.loss
    print(f"Node features: {a.node_feat}   Loss: {LOSS}", flush=True)
    pool = build_pool(a.node_feat)
    if len(pool) < 5:
        print("ERROR: pool too small."); sys.exit(1)

    # Pre-build all graphs once (reused across folds/seeds)
    print("Building graphs...", flush=True)
    t0 = time.time()
    graphs = {c: [make_graph(pool[c], p) for p in pool[c]["poses"]] for c in pool}
    print(f"  {sum(len(v) for v in graphs.values())} graphs "
          f"({time.time()-t0:.0f}s)", flush=True)

    if a.learning_curve:
        learning_curve(pool, graphs)
        return

    cxs = sorted(pool)
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(cxs))
    folds = [[cxs[i] for i in perm[f::FOLDS]] for f in range(FOLDS)]

    cx_results: dict = {}
    fold_taus: list[float] = []

    for fi in range(FOLDS):
        val_cxs = folds[fi]
        tr_cxs  = [c for c in cxs if c not in set(val_cxs)]
        print(f"\nFold {fi+1}/{FOLDS}  train={len(tr_cxs)} val={len(val_cxs)}", flush=True)

        seed_pred: dict = {c: [] for c in val_cxs}
        for sd in SEEDS:
            print(f"  seed {sd}", flush=True)
            res = train_eval(pool, tr_cxs, val_cxs, graphs, sd)
            for c in val_cxs:
                seed_pred[c].append(res[c][0])

        seed_taus = []
        for c in val_cxs:
            preds = np.mean(seed_pred[c], axis=0)      # avg model output over seeds
            gl = graphs[c]
            rmsd = np.array([g.y.item() for g in gl])
            pis  = [p["pi"] for p in pool[c]["poses"]]
            score = _pred_to_score(preds)              # higher = better pose
            t, _ = sp.kendalltau(score, -rmsd)
            tau = float(t) if not math.isnan(t) else 0.0
            seed_taus.append(tau)

            srt = np.argsort(-score)
            topk = {tk: float(rmsd[srt[:tk]].min()) for tk in [1, 5, 10, 25]}
            p0 = next((rmsd[i] for i, pp in enumerate(pis) if pp == 0), rmsd[0])
            cx_results[c] = {"tau": tau, "oracle": float(rmsd.min()),
                             "rapd_top1": float(p0),
                             **{f"top{tk}": v for tk, v in topk.items()}}
        fold_taus.append(float(np.mean(seed_taus)))
        print(f"  fold τ = {fold_taus[-1]:+.4f}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    N = len(cx_results)
    taus = np.array([cx_results[c]["tau"] for c in cx_results])
    print(f"\n{'='*70}")
    print(f"CONTACT GNN RANKER ({a.node_feat})  —  {N} complexes, {FOLDS}-fold, {len(SEEDS)} seeds")
    print(f"{'='*70}")
    print(f"\nKendall τ = {taus.mean():+.4f} ± {np.std(fold_taus):.4f} (fold std)")
    print(f"  per-complex τ std = {taus.std():.4f}")
    print(f"\n  Baseline ref2015/BSA+clash:  τ = +0.14,  top-1 Hit@2Å = 10.5%")
    print(f"  Oracle ceiling:              τ = 1.00,  top-1 Hit@2Å = 49.1%")

    print(f"\nTop-k RMSD & Hit@2Å (mean over {N} complexes):")
    print(f"  {'Metric':<14} {'GNN RMSD':>10} {'GNN Hit@2Å':>12} {'oracle':>9}")
    print(f"  {'-'*48}")
    orc = np.array([cx_results[c]["oracle"] for c in cx_results])
    rapd = np.array([cx_results[c]["rapd_top1"] for c in cx_results])
    print(f"  {'RAPiDock p0':<14} {rapd.mean():>8.2f}Å {100*np.mean(rapd<=2):>10.1f}% {orc.mean():>7.2f}Å")
    for tk in [1, 5, 10, 25]:
        v = np.array([cx_results[c][f"top{tk}"] for c in cx_results])
        print(f"  {'top-'+str(tk):<14} {v.mean():>8.2f}Å {100*np.mean(v<=2):>10.1f}% {orc.mean():>7.2f}Å")

    out_path = OUT / f"contact_gnn_{a.node_feat}_{a.loss}_cv.json"
    out_path.write_text(json.dumps(
        {c: cx_results[c] for c in cx_results} |
        {"_meta": {"node_feat": a.node_feat, "n": N, "tau_mean": float(taus.mean()),
                   "fold_taus": fold_taus}}, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
