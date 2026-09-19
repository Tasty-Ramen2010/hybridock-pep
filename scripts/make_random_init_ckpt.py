#!/usr/bin/env python3
"""Save an UNTRAINED (random-init) RAPiDock checkpoint as the true zero point.

Why: the from-scratch run's progress was being judged against the *pretrained*
model (4.68A), which only ever says "not there yet" and cannot distinguish
"learning slowly" from "not learning at all". The control that actually answers
that is the model before any training. Benchmarking random-init -> ep1 -> ep63
gives a real trajectory instead of a single uninterpretable point.

Written in the format inference.py's load_model() expects: a dict with "model"
(state_dict) and "ema_weights" (shadow_params list), because inference loads the
state dict and then OVERWRITES it with the EMA weights via copy_to(). Saving only
"model" would silently leave the EMA half undefined.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "RAPiDock_finetuned"))

from utils.utils import get_model  # noqa: E402


def main() -> None:
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 43

    torch.manual_seed(seed)
    with open(model_dir / "model_parameters.yml") as fh:
        args = Namespace(**yaml.full_load(fh))

    model = get_model(args, no_parallel=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    shadow = [p.detach().cpu().clone() for p in model.parameters()]

    torch.save(
        {
            "epoch": 0,
            "model": sd,
            "ema_weights": {"decay": 0.999, "num_updates": 0, "shadow_params": shadow},
        },
        out_path,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {out_path}  ({n:,} random-init params, {len(shadow)} ema shadow tensors)")


if __name__ == "__main__":
    main()
