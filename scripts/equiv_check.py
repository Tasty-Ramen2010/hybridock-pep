#!/usr/bin/env python
"""Upstream-with-port vs fork: same checkpoint, same cached graph, same t -> same scores?

Run once from each tree (module names clash, so separate processes):
  cd third_party/RAPiDock           && python equiv_check.py upstream  OUT
  cd third_party/RAPiDock_finetuned && python equiv_check.py fork      OUT
then:  python equiv_check.py compare OUT
"""
from __future__ import annotations

import glob
import sys

import torch

CKPT = ("/home/igem/unknown_software/third_party/RAPiDock_finetuned/"
        "longft_tanh/rapidock_finetuned_epoch010.pt")
MD = "/home/igem/unknown_software/third_party/RAPiDock/train_models/CGTensorProductEquivariantModel"
# FROZEN copies: the live RecentSet dir holds two cached graphs per complex (an old
# upper-case one and one the running benchmark writes), so globbing it is not stable.
GRAPH_GLOB = "/tmp/claude-1000/frozen_graphs/*/*.pt"


def run(tag: str, out: str) -> None:
    import yaml
    from argparse import Namespace
    sys.path.insert(0, ".")
    from torch_geometric.loader import DataLoader
    from utils.diffusion_utils import set_time
    import inference as INF

    args = Namespace(**yaml.full_load(open(f"{MD}/model_parameters.yml")))
    # Seed BEFORE construction: one model instance is bit-deterministic pass-to-pass, but
    # separate processes differ on the torsion heads, so something built randomly in
    # __init__ survives load_state_dict. Seeding here makes both trees build it identically.
    import random
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = INF.load_model(args, CKPT, "cpu").eval()
    graphs = sorted(glob.glob(GRAPH_GLOB))[:3]
    res = {}
    for g in graphs:
        for t in (0.9, 0.5, 0.1):
            # Re-load per pass: forward() does copy.copy(_data) (SHALLOW), and
            # get_updated_peptide_feature writes into data['pep_a'] in place, so reusing one
            # loaded graph across passes makes each pass depend on the previous ones. That
            # alone invalidated earlier "divergence" numbers from this script.
            data = torch.load(g, map_location="cpu", weights_only=False)
            # Seed before EVERY forward: single-threaded runs still differed (5e-3) on the
            # torsion heads only, so the forward itself draws random numbers. Seeding makes
            # same-tree runs bit-identical, so any cross-tree difference left is real.
            import random
            import numpy as np
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            batch = next(iter(DataLoader([data], batch_size=1)))
            set_time(batch, t, t, t, t, 1, torch.device("cpu"))
            with torch.no_grad():
                o = model(batch)
            res[f"{g.split('/')[-1].replace('_graph_v1.pt','')}@{t}"] = {k: (v.detach().clone() if v is not None else None)
                                             for k, v in o.items()}
    torch.save(res, f"{out}_{tag}.pt")
    print(f"{tag}: {len(res)} forward passes saved")


def compare(out: str) -> None:
    a = torch.load(f"{out}_upstream.pt")
    b = torch.load(f"{out}_fork.pt")
    worst = 0.0
    for k in sorted(a):
        for head in a[k]:
            x, y = a[k][head], b[k][head]
            if x is None or y is None:
                continue
            d = (x - y).abs().max().item()
            rel = d / (y.abs().max().item() + 1e-12)
            worst = max(worst, rel)
            print(f"  {k:14s} {head:22s} max|diff|={d:.3e}  rel={rel:.3e}")
    print(f"WORST relative difference: {worst:.3e}  -> {'EQUIVALENT' if worst < 1e-4 else 'DIFFERENT'}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        run(sys.argv[1], sys.argv[2])
