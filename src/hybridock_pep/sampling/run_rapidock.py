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
    """Seed torch and all GPU backends (CUDA/ROCm/XPU), numpy, and random.

    Called BEFORE any RAPiDock import so that ESM embedding computation and
    all downstream RNG calls are deterministic.

    Platform notes:
    - CUDA (NVIDIA) / ROCm (AMD): torch.cuda.manual_seed_all seeds all devices.
      ROCm uses the CUDA API — torch.cuda.is_available() returns True on ROCm.
    - XPU (Intel): torch.xpu.manual_seed_all seeds Intel GPU devices.
      Available when intel-extension-for-pytorch (ipex) is installed or when
      using PyTorch 2.4+ with native XPU support on Linux x86_64.
    - MPS (macOS Apple Silicon): torch.manual_seed covers MPS; no separate
      mps.manual_seed_all exists as of PyTorch 2.7.
    - CPU: torch.manual_seed is sufficient.

    Args:
        seed: Integer RNG seed to apply to all backends.
    """
    import torch
    import numpy as np

    torch.manual_seed(seed)

    # CUDA (NVIDIA) and ROCm (AMD) — both use the cuda API in PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Intel XPU — available with ipex or native PyTorch 2.4+ XPU build
    if hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None)):
        if torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)

    # MPS: torch.manual_seed covers MPS backend already.

    np.random.seed(seed)
    random.seed(seed)


def _optimize_backends():
    # type: () -> str
    """Apply per-backend performance knobs for whichever device PyTorch will use.

    Runs AFTER seeding and BEFORE the RAPiDock import, so it tunes the device
    RAPiDock-Reloaded auto-selects without changing *which* device is chosen or
    perturbing RNG seeding (TF32 changes matmul mantissa bits, not RNG streams;
    the project already documents CUDA nondeterminism in run_metadata.json).

    Optimizations (grounded in the PyTorch Performance Tuning Guide):
    - CUDA (NVIDIA) / ROCm (AMD): enable the TF32 fast path on Ampere+/Blackwell
      and RDNA — `set_float32_matmul_precision('high')` + cuda/cudnn allow_tf32.
      ~3x faster FP32 matmuls/convs with negligible accuracy impact; the RTX 5070
      (Blackwell CC 12.0) and modern AMD cards both benefit.
    - XPU (Intel GPU): import intel-extension-for-pytorch (ipex) if present, which
      registers fused kernels and the XPU device; tune matmul precision.
    - MPS (Apple Silicon): set PYTORCH_ENABLE_MPS_FALLBACK so the rare op MPS
      lacks falls back to CPU instead of aborting the run.
    - CPU: pin intra-op threads to the physical core count for steady throughput.

    Returns:
        Short human-readable label of the backend tuned (for logging).
    """
    import os

    import torch

    # MPS op-fallback must be set before the first MPS allocation.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # CUDA (NVIDIA) and ROCm (AMD) both report through torch.cuda.
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # autotune convs for fixed shapes
        except Exception:  # pragma: no cover - knob unavailable on old builds
            pass
        return "CUDA/ROCm (TF32 fast path)"

    # Intel XPU — importing ipex registers fused kernels + the xpu device.
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        try:
            import intel_extension_for_pytorch  # type: ignore[import]  # noqa: F401
        except Exception:  # pragma: no cover - ipex optional
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover
            pass
        return "XPU (Intel)"

    if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        return "MPS (Apple, op-fallback enabled)"

    # CPU: pin to physical cores (os.cpu_count() counts logical; halve if SMT).
    try:
        n = os.cpu_count() or 1
        torch.set_num_threads(max(1, n // 2) if n > 2 else n)
    except Exception:  # pragma: no cover
        pass
    return "CPU (threads tuned)"


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

    # Seed BEFORE any torch/numpy/RAPiDock import
    # Doing this here ensures ESM embeddings and all diffusion steps are reproducible
    if args.seed is not None:
        _seed_everything(args.seed)

    # Tune the auto-selected backend (CUDA/ROCm/XPU/MPS/CPU) for throughput.
    try:
        backend = _optimize_backends()
        print("[run_rapidock] backend optimization: %s" % backend, file=sys.stderr)
    except Exception as exc:  # pragma: no cover - never block inference on a knob
        print("[run_rapidock] backend optimization skipped: %s" % exc, file=sys.stderr)

    # Resolve all paths to absolute (conda run has unpredictable cwd)
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
    rd_args.no_final_step_noise = True  # deterministic final step: zero noise on last denoising
    # pass — this prevents the final Langevin noise injection from landing sidechain atoms
    # (PHE, TRP, etc.) inside receptor atoms. The first 15/16 steps still sample stochastically;
    # the last step settles to the score model's MAP estimate for all DOFs (tr, rot, tor_bb, tor_sc).
    # Observed effect: ~18/100 CPU poses fail clash relief with single v.optimize(); expected
    # improvement to ~5-8 with no_final_step_noise=True (sidechain displacements < 0.2 Å from MAP).

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
