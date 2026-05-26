"""RAPiDock-Reloaded inference shim — executed inside the rapidock conda env (Python 3.10).

This script is called by rapidock_runner.py via the rapidock env's python3 binary:
    /path/to/miniconda3/envs/rapidock/bin/python3 /abs/path/run_rapidock.py [args]

It seeds all RNGs (torch, numpy, random) BEFORE calling rd_inference.main() so that
--seed N is honoured for reproducibility even though inference.py has no --seed arg.

Uses RAPiDock-Reloaded (Tasty-Ramen2010/RAPiDock-Reloaded) which runs on:
  - CUDA (Linux/Windows, RTX 5070 Blackwell CC 12.0 with PyTorch 2.7 + cu128)
  - MPS (macOS Apple Silicon — automatic via PYTORCH_ENABLE_MPS_FALLBACK)
  - CPU (fallback)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional


def _seed_everything(seed):
    # type: (int) -> None
    """Seed torch, torch.cuda, numpy, and random before inference starts.

    Called BEFORE any RAPiDock import so that ESM embedding computation and
    all downstream RNG calls are deterministic (D-08).

    Args:
        seed: Integer RNG seed to apply to all backends.
    """
    import torch
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    # type: () -> None
    """Parse CLI args, seed RNGs, and invoke rd_inference.main()."""
    parser = argparse.ArgumentParser(
        description="RAPiDock-Reloaded inference shim (Python 3.10, rapidock env)"
    )
    parser.add_argument(
        "--peptide",
        required=True,
        help="Peptide sequence (single-letter AA codes)",
    )
    parser.add_argument(
        "--receptor",
        required=True,
        help="Absolute path to receptor PDB",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Absolute path to output directory (parent of complex_name subdir)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        dest="n_samples",
        help="Number of RAPiDock inference passes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (optional); sets torch/numpy/random seeds before inference",
    )
    parser.add_argument(
        "--rapidock-dir",
        required=True,
        dest="rapidock_dir",
        help="Absolute path to RAPiDock source directory (contains inference.py)",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        dest="model_dir",
        help="Absolute path to train_models/CGTensorProductEquivariantModel/",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint filename (e.g. rapidock_local.pt)",
    )
    parser.add_argument(
        "--scoring-function",
        default="none",
        dest="scoring_function",
        help=(
            "RAPiDock scoring function: none (default, diffusion-order ranking), "
            "confidence (requires confidence model), or ref2015 (requires PyRosetta). "
            "Use 'none' to skip re-ranking; HybriDock-Pep re-scores with Vina/AD4."
        ),
    )
    args = parser.parse_args()

    # Seed BEFORE any torch/numpy/RAPiDock import (D-08)
    # Doing this here ensures ESM embeddings and all diffusion steps are reproducible
    if args.seed is not None:
        _seed_everything(args.seed)

    # Resolve all paths to absolute (D-07: conda run has unpredictable cwd)
    receptor_abs = str(Path(args.receptor).resolve())
    output_dir_abs = str(Path(args.output_dir).resolve())
    rapidock_dir_abs = str(Path(args.rapidock_dir).resolve())
    model_dir_abs = str(Path(args.model_dir).resolve())

    # Inject RAPiDock source onto sys.path so utils/ and inference.py are importable
    if rapidock_dir_abs not in sys.path:
        sys.path.insert(0, rapidock_dir_abs)

    # Import RAPiDock argument parser and inference entry point
    from utils.inference_parsing import get_parser as rd_get_parser  # type: ignore[import]
    import inference as rd_inference  # type: ignore[import]

    # Build RAPiDock args Namespace from defaults, then override with our values
    rd_args = rd_get_parser().parse_args([])
    rd_args.protein_description = receptor_abs
    rd_args.peptide_description = args.peptide
    rd_args.output_dir = output_dir_abs
    rd_args.complex_name = "poses_raw"  # files land at output_dir/poses_raw/rank*.pdb
    rd_args.N = args.n_samples
    rd_args.model_dir = model_dir_abs
    rd_args.ckpt = args.ckpt
    rd_args.scoring_function = args.scoring_function
    rd_args.fastrelax = False  # CLAUDE.md §2.5: ref2015 fails on C-terminal cysteine
    rd_args.save_visualisation = False  # saves diffusion frames only; not needed here
    rd_args.config = None  # no YAML override

    # Parser defaults are None for these; the YAML config normally supplies them.
    # Since we set config=None, set the RAPiDock YAML defaults explicitly.
    if rd_args.inference_steps is None:
        rd_args.inference_steps = 16
    if rd_args.actual_steps is None:
        rd_args.actual_steps = 16
    if rd_args.batch_size is None:
        rd_args.batch_size = 4
    if rd_args.conformation_partial is None:
        rd_args.conformation_partial = "1:1:1"

    # Patch RAPiDock's silent exception swallower to emit full tracebacks.
    # inference.py catches Exception and prints only str(e), losing the traceback.
    import traceback as _tb
    _orig_process = rd_inference.process_complex

    def _process_with_traceback(*a, **kw):
        # type: (...) -> None
        try:
            return _orig_process(*a, **kw)
        except Exception as _e:
            _tb.print_exc()
            raise

    rd_inference.process_complex = _process_with_traceback

    # Invoke RAPiDock inference — writes rank*.pdb to {output_dir}/poses_raw/
    rd_inference.main(rd_args)


if __name__ == "__main__":
    main()
