"""Centralized hardware/accelerator tuning for HybriDock-Pep's compute paths.

One place that picks — and tunes — the device every heavy stage runs on, so the
tool gets the best of whatever silicon it lands on with no user flags:

  * **OpenMM** (Stage 1.5 clash-relief minimization + Stage 3.5 MM-GBSA) —
    platform priority **CUDA (NVIDIA) → HIP (AMD ROCm) → OpenCL (Intel/Apple GPU)
    → CPU**, mixed precision on the CUDA/HIP fast paths, physical-core thread
    pinning on CPU.
  * **AutoDock Vina** — physical-core thread count via :func:`cpu_threads`.
  * **RAPiDock (torch) inference** is tuned separately in
    ``sampling/run_rapidock.py::_optimize_backends`` because it runs inside the
    ``rapidock`` conda env where torch is importable (CUDA/ROCm TF32 fast path,
    Intel XPU ipex, Apple MPS op-fallback, CPU threads).

Grounded in the OpenMM Platform guide: CUDA for NVIDIA, **HIP for AMD** (OpenCL is
"usually slower" on AMD), OpenCL for Intel/Apple, CPU otherwise; **"mixed"
precision** computes forces in single and integrates in double — near-double
accuracy at near-single speed (energy drift 0.22 vs 3.98 kJ/mol/ns for single).
Refs: https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html
      https://docs.openmm.org/latest/developerguide/07_cuda_platform.html
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: GPU platforms tried in order; CUDA/HIP take identical properties (HIP mirrors CUDA).
_GPU_PLATFORMS: tuple[tuple[str, dict[str, str]], ...] = (
    ("CUDA", {"DeviceIndex": "0", "Precision": "mixed"}),    # NVIDIA
    ("HIP", {"DeviceIndex": "0", "Precision": "mixed"}),     # AMD (ROCm); same props as CUDA
    ("OpenCL", {"DeviceIndex": "0", "Precision": "single"}),  # Intel / Apple — single = widest compat
)


def cpu_threads() -> int:
    """Physical-core thread count for FP-heavy compute (MD minimization, Vina).

    Honors ``OPENMM_CPU_THREADS`` when set. Otherwise uses half the logical core
    count as a physical-core proxy (≥1) — molecular-mechanics work scales with
    physical, not SMT, cores, so over-subscribing logical cores wastes context
    switches without adding throughput.

    Returns:
        Thread count ≥ 1.
    """
    env = os.environ.get("OPENMM_CPU_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    n = os.cpu_count() or 1
    return max(1, n // 2) if n > 2 else n


def openmm_platform(force_cpu: bool = False) -> tuple[Any, dict[str, str]]:
    """Return ``(openmm.Platform, properties)`` for the fastest available backend.

    Priority **CUDA → HIP → OpenCL → CPU**. GPU platforms request mixed precision
    (CUDA/HIP) or single (OpenCL, widest device support); CPU pins ``Threads`` to
    the physical-core count. A platform whose runtime is unavailable raises at
    ``getPlatformByName`` and is skipped, so this never hard-fails.

    Args:
        force_cpu: Short-circuit to the thread-pinned CPU platform.

    Returns:
        Tuple of (platform, properties dict) ready for ``openmm.Context``.
    """
    import openmm  # noqa: PLC0415 — heavy optional dep, imported lazily

    cpu_props = {"Threads": str(cpu_threads())}
    if force_cpu:
        return openmm.Platform.getPlatformByName("CPU"), cpu_props

    for name, props in _GPU_PLATFORMS:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            logger.debug("OpenMM: selected %s platform", name)
            return platform, props
        except Exception:  # noqa: BLE001 — platform simply not built into this OpenMM
            continue

    logger.debug("OpenMM: no GPU platform available, using thread-pinned CPU")
    return openmm.Platform.getPlatformByName("CPU"), cpu_props
