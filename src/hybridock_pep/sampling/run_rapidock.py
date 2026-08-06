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
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Must be set before the FIRST `import torch` anywhere in this process.
# _optimize_backends() below imports torch (unconditionally, before
# inference.py is ever reached) and on macOS that collides with conda's
# llvm-openmp (both link libomp.dylib), aborting with "OMP: Error #15" —
# the SIGABRT kills Stage 1 before it can generate a single pose. inference.py
# guards against this too, but only for imports that happen after it loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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


def _total_ram_gb():
    # type: () -> float
    """Physical RAM in GB, or 16.0 if it cannot be determined."""
    try:
        import os as _os
        return (_os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")) / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def _default_batch_cap():
    # type: () -> int
    """Largest per-step batch that fits in RAM without swapping.

    Batching more poses per denoising step cuts MPS dispatch overhead, but each
    batched pose costs roughly 0.5 GB of peak working set (GPU-side activations
    plus the CPU-side graph copies sampling.py holds). Exceed physical RAM and
    the run does not degrade gracefully -- it falls off a cliff into swap.

    Measured on a 16 GB M3, --n-samples 100, 1YCR/MDM2 + 12-mer:

        batch   4    8   16   20    24    32
        wall   98s  70s  62s  64s   83s  398s
        swap  -24M -80M -32M +638M +1.8G +4.8G

    Batch 32 is 6.4x slower than batch 16 purely from paging, and worse than
    the batch of 4 this wrapper used to hardcode.

    Note how asymmetric that curve is. Undershooting the optimum costs ~15%
    (batch 8 is 70s against 62s); overshooting costs 640%. The throughput
    curve is nearly flat from 8 to 20 and then falls off a cliff, so this
    deliberately aims *below* the measured optimum: budget 45% of physical
    memory at ~0.5 GB/pose, giving 15 on this 16 GB machine (which reports
    17.2 GB) against a measured optimum of 16 and first swapping at 20.
    Guessing low is nearly free; guessing high is not.

    Override with HYBRIDOCK_RAPIDOCK_BATCH if you know your machine.
    """
    cap = int((_total_ram_gb() * 0.45) / 0.5)
    return max(2, min(32, cap))


def _compute_batch_size(n_samples, cap=None):
    # type: (int, int) -> int
    """Pick how many poses' diffusion-step forward passes get batched together.

    Diffusion sampling batches poses into forward passes per denoising step.
    This wrapper used to hardcode 4 regardless of --n-samples, forcing e.g. a
    10-pose run into 3 separate MPS forward passes per step instead of 1.

    Each MPS kernel dispatch carries real fixed overhead (MPS lacks CUDA-style
    Tensor Core batching and has documented higher per-op dispatch latency --
    https://github.com/pytorch/pytorch/issues/122123), so 3 calls of ~3 samples
    cost far more than 1 call of 10. Measured: n=10 went from 19.5s
    (batch_size=4) to 10.6s (batch_size=10).

    The upper bound is a memory limit, not a throughput one -- see
    :func:`_default_batch_cap` for the measurements behind it. Batching is not
    free above that point; it is catastrophic.

    Args:
        n_samples: The requested --n-samples / rd_args.N value.
        cap: Maximum poses per batch. Defaults to the RAM-derived cap.

    Returns:
        The batch size to pass as rd_args.batch_size.
    """
    env = os.environ.get("HYBRIDOCK_RAPIDOCK_BATCH")
    if env:
        try:
            forced = int(env)
            if forced > 0:
                return min(n_samples, forced)
        except ValueError:
            pass
    if cap is None:
        cap = _default_batch_cap()
    return min(n_samples, cap)


def _disable_jit_fusion_if_arch_unsupported(torch):
    # type: (object) -> bool
    """Turn off TorchScript GPU fusion when the wheel cannot codegen for this GPU.

    PyTorch's TensorExpr fuser compiles fused kernels at runtime with NVRTC,
    passing the device's compute capability as `-arch`. NVRTC only accepts an
    architecture the shipped toolkit knows. On a GPU newer than the wheel —
    the DGX Spark's GB10 is sm_121 while torch 2.7.0+cu128 reports
    `arch_list == ['sm_90', 'sm_100', 'sm_120']` — every fused kernel dies with

        nvrtc: error: invalid value for --gpu-architecture (-arch)

    RAPiDock swallows that into "0 poses generated", so Stage 1 fails on
    hardware where everything else (cuBLAS, cuDNN, eager kernels) works fine.

    Only the runtime *codegen* path is broken, so disabling GPU fusion is
    enough; the same ops then run as unfused eager kernels. Applied only when
    the device's capability is genuinely missing from the build, so machines
    with a matching wheel keep the fuser.

    Returns:
        True if fusion was disabled, False if the wheel supports this GPU.
    """
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
    except Exception:  # pragma: no cover - ROCm and odd builds
        return False
    # ROCm reports gfx* targets; this whole failure mode is NVRTC-specific.
    if not any(a.startswith("sm_") for a in arch_list):
        return False
    if "sm_{}{}".format(major, minor) in arch_list:
        return False
    for setter, arg in (
        ("_jit_set_texpr_fuser_enabled", False),
        ("_jit_override_can_fuse_on_gpu", False),
        ("_jit_set_nvfuser_enabled", False),
        ("_jit_set_profiling_executor", False),
    ):
        try:
            getattr(torch._C, setter)(arg)
        except Exception:  # pragma: no cover - knob absent on some builds
            pass
    sys.stderr.write(
        "[run_rapidock] GPU is sm_{}{} but this PyTorch build targets {} — "
        "disabling TorchScript GPU fusion so NVRTC is never asked to compile "
        "for an architecture it does not know. Sampling runs unfused.\n".format(
            major, minor, ", ".join(arch_list)
        )
    )
    return True


def _try_patch_metal_e3nn():
    # type: () -> str
    """Patch e3nn's tensor products with metal-e3nn's fused Metal kernel.

    metal_e3nn's patch_(model) requires per-model patching, so we wrap
    torch.nn.Module.__init__ to auto-patch any module containing TensorProducts.
    This runs once per model, with negligible overhead.

    Returns a short label suffix for logging: " + metal-e3nn" if the patch
    was actually applied, "" otherwise (including when disabled via
    METAL_E3NN_DISABLE=1 or if metal_e3nn is not installed).
    """
    if os.environ.get("METAL_E3NN_DISABLE"):
        return ""
    try:
        import metal_e3nn  # type: ignore[import-not-found]
        import torch.nn as nn

        # Monkey-patch nn.Module to auto-patch metal-e3nn on any model loaded
        original_init = nn.Module.__init__

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            original_init(self, *args, **kwargs)
            try:
                metal_e3nn.patch_(self)
            except Exception:
                pass  # Silently skip if patch fails (no TensorProducts, etc)

        nn.Module.__init__ = patched_init
        sys.stderr.write("[run_rapidock] metal-e3nn auto-patching enabled\n")
        return " + metal-e3nn"
    except ImportError:  # pragma: no cover
        return ""
    except Exception as e:  # pragma: no cover - other failures are silent
        sys.stderr.write(f"[run_rapidock] metal-e3nn setup failed: {e}\n")
        return ""


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
      lacks falls back to CPU instead of aborting the run, then patch e3nn's
      tensor products with metal-e3nn (github.com/Tasty-Ramen2010/metal-e3nn)
      if it's installed — a separate, standalone project, not bundled here.
      3.1-6.7x faster on Apple GPUs per its own benchmarks; purely additive
      (falls through to stock e3nn if the package is absent, or if
      METAL_E3NN_DISABLE=1 is set) so nothing here depends on it existing.
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
        label = "CUDA/ROCm (TF32 fast path)"
        if _disable_jit_fusion_if_arch_unsupported(torch):
            label += ", JIT GPU fusion off"
        return label

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
        return "MPS (Apple, op-fallback enabled)" + _try_patch_metal_e3nn()

    # CPU: pin to physical cores (os.cpu_count() counts logical; halve if SMT).
    try:
        n = os.cpu_count() or 1
        torch.set_num_threads(max(1, n // 2) if n > 2 else n)
    except Exception:  # pragma: no cover
        pass
    return "CPU (threads tuned)"


PROGRESS_PREFIX = "[[HDP-PROGRESS]]"


def _install_progress_probe(rd_inference, rd_args):
    # type: (object, object) -> None
    """Emit machine-readable sampling progress on stdout for the parent process.

    Pose generation is the longest silent stretch of a run (minutes on MPS with
    no output at default verbosity), which is exactly when a non-expert decides
    the tool has hung. RAPiDock has no progress callback, so we derive one.

    The denoising loop runs `actual_steps` diffusion steps, and inside each step
    iterates the batched pose set — so the model's forward is called exactly
    once per (step, batch). Both counts are known up front, which makes total
    forward calls a genuine, monotonic progress denominator rather than a guess.

    Wrapping the *model* rather than editing RAPiDock keeps this entirely on our
    side of the submodule boundary: third_party/RAPiDock is a pinned git
    submodule, so any edit there would be unpushable and lost on re-clone.

    Never fatal: any failure here silently leaves sampling unmonitored rather
    than breaking the run that the user actually asked for.
    """
    try:
        _orig_sampling = rd_inference.sampling
    except AttributeError:
        return

    def _sampling_with_progress(data_list, model, args, *a, **kw):
        try:
            steps = kw.get("actual_steps") or kw.get("inference_steps") or 0
            batch_size = kw.get("batch_size") or 1
            n_batches = max(1, -(-len(data_list) // batch_size))  # ceil div
            total = int(steps) * n_batches
        except Exception:
            total = 0
        if total <= 0 or not hasattr(model, "forward"):
            return _orig_sampling(data_list, model, args, *a, **kw)

        state = {"n": 0}
        _orig_forward = model.forward

        def _counting_forward(*fa, **fkw):
            state["n"] += 1
            # stdout is the parent's structured channel; RAPiDock's own chatter
            # goes to stderr, so these lines never interleave with it.
            print("%s %d %d" % (PROGRESS_PREFIX, min(state["n"], total), total))
            sys.stdout.flush()
            return _orig_forward(*fa, **fkw)

        model.forward = _counting_forward
        try:
            return _orig_sampling(data_list, model, args, *a, **kw)
        finally:
            model.forward = _orig_forward

    rd_inference.sampling = _sampling_with_progress


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
        rd_args.batch_size = _compute_batch_size(rd_args.N)
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

    _install_progress_probe(rd_inference, rd_args)

    # Invoke RAPiDock inference — writes rank*.pdb to {output_dir}/poses_raw/
    rd_inference.main(rd_args)


if __name__ == "__main__":
    main()
