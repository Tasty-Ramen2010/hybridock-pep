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


#: Result of the one-time platform probe below. A usable backend never changes
#: within a process, and re-probing is not free: a FAILED CUDA context strands
#: about 100 MB of device memory that is never reclaimed, so probing per pose
#: turned one broken install into a 10 GB leak and an OOM kill at pose 40/100.
_PLATFORM_CACHE: tuple[Any, dict[str, str]] | None = None


def _platform_runs(openmm: Any, platform: Any, props: dict[str, str]) -> bool:
    """Return True if ``platform`` can actually build a Context and step it.

    ``Platform.getPlatformByName`` only proves the platform was COMPILED into
    this OpenMM build — not that it works on this machine. The gap between those
    two is not hypothetical: on Colab, conda-forge's OpenMM 8.6 ships nvrtc
    13.3 while the driver is 580 (CUDA 13.0), so every CUDA Context dies at
    module load with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. The old code handed
    back that unusable platform, each caller's Context construction failed and
    leaked, and the real work quietly fell through to OpenCL — so the GPU looked
    idle, the memory climbed, and nothing was ever logged above DEBUG.

    Building one throwaway two-particle Context here costs milliseconds and
    converts that failure mode into a single WARNING plus a working fallback.
    """
    try:
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        integrator = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(system, integrator, platform, props)
        ctx.setPositions([(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)])
        ctx.getState(getEnergy=True).getPotentialEnergy()
        del ctx, integrator, system
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "do not use this"
        logger.debug("OpenMM: platform %s rejected by probe: %s",
                     platform.getName(), exc)
        return False


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

    global _PLATFORM_CACHE
    if _PLATFORM_CACHE is not None:
        return _PLATFORM_CACHE

    for name, props in _GPU_PLATFORMS:
        try:
            platform = openmm.Platform.getPlatformByName(name)
        except Exception:  # noqa: BLE001 — platform simply not built into this OpenMM
            continue
        if not _platform_runs(openmm, platform, props):
            logger.warning(
                "OpenMM: the %s platform is present but cannot run — skipping it. "
                "Minimization and MM-GBSA will use the next backend.", name
            )
            continue
        logger.debug("OpenMM: selected %s platform", name)
        _PLATFORM_CACHE = (platform, props)
        return _PLATFORM_CACHE

    logger.debug("OpenMM: no usable GPU platform, using thread-pinned CPU")
    _PLATFORM_CACHE = (openmm.Platform.getPlatformByName("CPU"), cpu_props)
    return _PLATFORM_CACHE
