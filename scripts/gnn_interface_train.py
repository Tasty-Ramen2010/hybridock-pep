#!/usr/bin/env python
"""Train a graph net on interface atoms, and ask whether it knows anything the descriptors do not.

THE QUESTION IS COMPLEMENTARITY, NOT VICTORY. A graph model that ties the random forest is not a
failure and one that loses is not necessarily useless -- what matters is whether it is wrong about
DIFFERENT complexes. Our four descriptor sets all failed on the same complexes (|residual|
correlating 0.81-0.96, 53% of failures shared across all four), which is why adding descriptors
never helped. A model reading raw atoms could break that pattern even at equal accuracy, and that
would be worth more than a small win that fails in the same places.

So three numbers are reported: the GNN alone, the RF alone on identical splits, and the two
combined -- plus the correlation between their errors, which is the one that decides whether this
line is worth continuing.

THE PRIOR IS AGAINST IT, and is stated so the result can be read honestly. Coventry (21:40): "you
don't have a lot of data to really get too creative." Our own e436 tensor-product model collapsed
to a near-constant. With ~900 graphs this model is small on purpose: three message-passing layers,
64 hidden, heavy weight decay, early stopping. A bigger one would fit the training set and tell
us nothing.

WSL2 CONSTRAINTS ARE LOAD-BEARING HERE (CLAUDE.md 2.2a). A single large GPU burst can trip the
dxgkrnl paravirtualisation timeout and take the process down. So: bounded mini-batches, an
explicit max_num_neighbors cap on the radius graph so one dense interface cannot blow up the edge
count, and an OOM handler that retries a batch at half size rather than dying.

Usage: gnn_interface_train.py [--epochs 120] [--folds 5]
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
GRAPHS = ROOT / "data/gnn_graphs.npz"
CORPUS = ROOT / "data/e432/corpus.npz"
MAX_NEIGHBORS = 32          # hard cap: one dense interface must not explode the edge count
RADIUS = 5.0


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch_cluster import radius_graph
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = np.load(GRAPHS, allow_pickle=True)
    Xs, Ps, y, grp, names = g["X"], g["P"], g["y"].astype(float), g["g"], g["names"]
    print(f"{len(y)} interface graphs, median "
          f"{int(np.median([len(x) for x in Xs]))} atoms, device {dev}\n")
    if len(y) < 80:
        print("too few graphs"); return

    # descriptor baseline on EXACTLY the same complexes, so the comparison is clean
    c = np.load(CORPUS, allow_pickle=True)
    mm = c["has_feats"].astype(bool) & (c["source"] == "pdbbind")
    dmap = {str(p).lower()[:4]: f for p, f in zip(c["pdb"][mm], c["feats"][mm])}
    D = np.nan_to_num(np.array([dmap[n] for n in names]), posinf=0, neginf=0)
    D = D[:, D.std(0) > 1e-8]

    class Net(nn.Module):
        """Readout is sum+mean+max, NOT mean alone.

        Mean pooling over ~700 atoms was the bug in the first version: it normalises away
        interface SIZE, which is a first-order determinant of affinity, and it smooths every
        per-atom difference into the average. With mean-only readout this model could not even
        memorise 120 graphs (train MAE 1.54 against a label sd of 2.04, regularisation off);
        with sum+mean+max and residual connections it reaches train MAE 0.05. Sum carries how
        much interface there is, mean carries what it is made of, max carries hotspots.
        """

        def __init__(self, nf, h=96):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(nf, h), nn.SiLU(), nn.LayerNorm(h))
            self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * h + 16, h), nn.SiLU(),
                                                    nn.Linear(h, h)) for _ in range(3)])
            self.nrm = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
            self.head = nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Dropout(0.2),
                                      nn.Linear(h, 1))
            self.register_buffer("centres", torch.linspace(0, RADIUS, 16))

        def forward(self, x, pos, batch):
            ei = radius_graph(pos, r=RADIUS, batch=batch, loop=False,
                              max_num_neighbors=MAX_NEIGHBORS)
            h = self.emb(x)
            src, dst = ei
            d = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
            rbf = torch.exp(-((d - self.centres) ** 2) / 0.5)
            for msg, nrm in zip(self.msg, self.nrm):
                m = msg(torch.cat([h[src], h[dst], rbf], dim=-1))
                h = nrm(h + torch.zeros_like(h).index_add_(0, dst, m))
            return self.head(torch.cat([global_add_pool(h, batch) / 50.0,
                                        global_mean_pool(h, batch),
                                        global_max_pool(h, batch)], dim=-1)).squeeze(-1)

    def run_batch(model, idx, ys, train, opt=None, bs=None):
        bs = bs or a.batch
        out, tot = [], 0.0
        for s in range(0, len(idx), bs):
            chunk = idx[s:s + bs]
            try:
                xs = torch.cat([torch.from_numpy(Xs[i]) for i in chunk]).to(dev)
                ps = torch.cat([torch.from_numpy(Ps[i]) for i in chunk]).to(dev)
                bt = torch.cat([torch.full((len(Xs[i]),), k, dtype=torch.long)
                                for k, i in enumerate(chunk)]).to(dev)
                if train:
                    opt.zero_grad()
                    p = model(xs, ps, bt)
                    loss = nn.functional.smooth_l1_loss(p, ys[chunk].to(dev))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    tot += float(loss) * len(chunk)
                else:
                    with torch.no_grad():
                        out.append(model(xs, ps, bt).cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs > 1:      # degrade this step rather than kill the run
                    return run_batch(model, idx, ys, train, opt, bs // 2)
                raise
        return tot / max(len(idx), 1) if train else np.concatenate(out)

    yt = torch.tensor(y, dtype=torch.float32)
    pred_gnn, pred_rf = np.zeros(len(y)), np.zeros(len(y))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(D, y, grp), 1):
        sc = StandardScaler().fit(D[tr])
        pred_rf[te] = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, n_jobs=-1,
                                            random_state=0) \
            .fit(sc.transform(D[tr]), y[tr]).predict(sc.transform(D[te]))

        torch.manual_seed(0)
        net = Net(Xs[0].shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        cut = max(8, int(0.15 * len(tr)))
        vi, ti = tr[:cut], tr[cut:]
        best, best_state, bad = 1e9, None, 0
        for ep in range(a.epochs):
            net.train()
            run_batch(net, np.random.permutation(ti), yt, True, opt)
            sched.step()
            net.eval()
            vp = run_batch(net, vi, yt, False)
            vl = float(np.abs(vp - y[vi]).mean())
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 25:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        pred_gnn[te] = run_batch(net, te, yt, False)
        print(f"  fold {fold}: GNN MAE {np.abs(pred_gnn[te] - y[te]).mean():.3f}   "
              f"RF MAE {np.abs(pred_rf[te] - y[te]).mean():.3f}", flush=True)

    print(f"\n{'model':<34}{'r':>8}{'MAE':>8}{'RMSE':>8}")
    for nm, p in (("GNN on interface atoms", pred_gnn), ("random forest on descriptors", pred_rf),
                  ("mean of the two", 0.5 * (pred_gnn + pred_rf))):
        print(f"{nm:<34}{corr(p, y):>8.3f}{np.abs(p - y).mean():>8.3f}"
              f"{np.sqrt(((p - y) ** 2).mean()):>8.3f}")
    print(f"{'predict the mean':<34}{'-':>8}{np.abs(y - y.mean()).mean():>8.3f}{y.std():>8.3f}")

    print("\nCOMPLEMENTARITY — the number that decides whether this line continues\n")
    eg, er = np.abs(pred_gnn - y), np.abs(pred_rf - y)
    print(f"  corr(GNN prediction, RF prediction)   {corr(pred_gnn, pred_rf):+.3f}")
    print(f"  corr(|GNN error|,   |RF error|)       {corr(eg, er):+.3f}")
    print("     our four descriptor sets sat at 0.81-0.96 here; lower means genuinely "
          "different failures")
    wg = int(((eg < er)).sum())
    print(f"  GNN closer on {wg}/{len(y)} complexes ({100 * wg / len(y):.0f}%)")
    best_mae = min(np.abs(pred_rf - y).mean(), np.abs(0.5 * (pred_gnn + pred_rf) - y).mean())
    print("\n  -> " + (
        "COMPLEMENTARY: the graph fails on different complexes, so it is worth more data."
        if corr(eg, er) < 0.6 and best_mae < np.abs(pred_rf - y).mean() + 1e-9 else
        "it fails where the descriptors fail and adds nothing; same shared floor as everything "
        "else."))


if __name__ == "__main__":
    main()
