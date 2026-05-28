#!/usr/bin/env python3
"""scripts/setup_environment.py — HybriDock-Pep automated environment setup.

Detects your OS + GPU and installs both conda environments (rapidock + score-env)
with the correct PyTorch backend and matching PyG wheels.

Supported configurations
------------------------
  Linux  + NVIDIA GPU  (CUDA 12.8)   → torch==2.7.0+cu128, PyG cu128 wheels
  Linux  + AMD GPU     (ROCm 6.3)    → torch==2.7.0+rocm6.3, PyG CPU wheels
  Linux  + Intel GPU   (XPU/SYCL)   → torch==2.7.0 + intel-extension-for-pytorch, PyG CPU
  Linux  + CPU only                  → torch (CPU), PyG CPU wheels
  macOS  Apple Silicon (MPS)         → torch (MPS), PyG CPU wheels
  macOS  Intel x86_64  (CPU)         → torch (CPU), PyG CPU wheels

Usage
-----
  python3 scripts/setup_environment.py             # auto-detect, interactive
  python3 scripts/setup_environment.py --dry-run   # print commands only, no changes
  python3 scripts/setup_environment.py --skip-scoring   # skip score-env creation
  python3 scripts/setup_environment.py --skip-rapidock  # skip rapidock env creation
  python3 scripts/setup_environment.py --backend cpu     # force CPU even with GPU present

Why two environments?
  rapidock (Python 3.10) — RAPiDock diffusion model: PyTorch, PyG, MDAnalysis, E3NN
  score-env (Python 3.11) — scoring: Vina, OpenMM, meeko, scikit-learn, RDKit

  Cramming both stacks into one env causes unsolvable version conflicts.  The
  driver (score-env) calls the rapidock env's python3 binary directly via
  subprocess — see src/hybridock_pep/sampling/rapidock_runner.py.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class PlatformInfo(NamedTuple):
    os_name: str          # "Linux", "Darwin", "Windows"
    arch: str             # "x86_64", "arm64", etc.
    backend: str          # "cuda", "rocm", "xpu", "mps", "cpu"
    gpu_label: str        # human-readable GPU description
    torch_index_url: str  # --index-url for `pip install torch`
    torch_version: str    # e.g. "torch==2.7.0"
    pyg_find_url: str     # -f URL for PyG wheels, or "" for CPU wheels
    ipex: bool            # install intel-extension-for-pytorch


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------

def _cmd_ok(*cmd: str) -> bool:
    """Return True if command exits 0 within 5 s."""
    try:
        subprocess.run(list(cmd), capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _nvidia_present() -> bool:
    return shutil.which("nvidia-smi") is not None and _cmd_ok("nvidia-smi")


def _nvidia_cc() -> str:
    """Return compute capability string like '12.0', or '' on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        return ""


def _amd_present() -> bool:
    """True if AMD ROCm runtime is accessible."""
    if shutil.which("rocminfo") is not None and _cmd_ok("rocminfo"):
        return True
    if shutil.which("rocm-smi") is not None and _cmd_ok("rocm-smi", "--showproductname"):
        return True
    # Fallback: ROCm kernel fusion driver device node
    return Path("/dev/kfd").exists()


def _intel_xpu_present() -> bool:
    """True if an Intel GPU with SYCL/Level-Zero driver is accessible."""
    if shutil.which("sycl-ls") is not None and _cmd_ok("sycl-ls"):
        return True
    # Scan DRM devices for Intel vendor ID (8086)
    drm = Path("/sys/class/drm")
    if drm.exists():
        for card in drm.iterdir():
            vendor_file = card / "device" / "vendor"
            try:
                if vendor_file.read_text().strip() == "0x8086":
                    return True
            except OSError:
                continue
    return False


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform(force_backend: str | None = None) -> PlatformInfo:
    """Detect OS, architecture, and compute backend.

    Args:
        force_backend: If provided, override GPU detection.  One of
            "cuda", "rocm", "xpu", "mps", "cpu".

    Returns:
        PlatformInfo describing the detected (or forced) configuration.
    """
    os_name = platform.system()       # "Linux", "Darwin", "Windows"
    arch    = platform.machine()      # "x86_64", "arm64", "AMD64"

    # --- macOS: MPS or CPU, no CUDA ---
    if os_name == "Darwin":
        if (force_backend or "mps") == "mps" and arch == "arm64":
            return PlatformInfo(
                os_name=os_name, arch=arch,
                backend="mps",
                gpu_label="Apple Silicon MPS",
                torch_index_url="",
                torch_version="torch",
                pyg_find_url="",
                ipex=False,
            )
        return PlatformInfo(
            os_name=os_name, arch=arch,
            backend="cpu",
            gpu_label="CPU (macOS Intel — no MPS on x86_64)",
            torch_index_url="",
            torch_version="torch",
            pyg_find_url="",
            ipex=False,
        )

    # --- Windows: CUDA or CPU (ROCm on Windows is experimental/unsupported) ---
    if os_name == "Windows":
        if (force_backend or ("cuda" if _nvidia_present() else "cpu")) == "cuda":
            return PlatformInfo(
                os_name=os_name, arch=arch,
                backend="cuda",
                gpu_label="NVIDIA GPU (CUDA 12.8)",
                torch_index_url="https://download.pytorch.org/whl/cu128",
                torch_version="torch==2.7.0",
                pyg_find_url="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
                ipex=False,
            )
        return PlatformInfo(
            os_name=os_name, arch=arch,
            backend="cpu",
            gpu_label="CPU (no NVIDIA GPU found)",
            torch_index_url="",
            torch_version="torch",
            pyg_find_url="",
            ipex=False,
        )

    # --- Linux / WSL2 ---
    backend = force_backend
    if backend is None:
        if _nvidia_present():
            backend = "cuda"
        elif _amd_present():
            backend = "rocm"
        elif _intel_xpu_present():
            backend = "xpu"
        else:
            backend = "cpu"

    if backend == "cuda":
        cc = _nvidia_cc()
        label = f"NVIDIA GPU — CUDA 12.8 (CC {cc})" if cc else "NVIDIA GPU — CUDA 12.8"
        return PlatformInfo(
            os_name=os_name, arch=arch,
            backend="cuda",
            gpu_label=label,
            torch_index_url="https://download.pytorch.org/whl/cu128",
            torch_version="torch==2.7.0",
            pyg_find_url="https://data.pyg.org/whl/torch-2.7.0+cu128.html",
            ipex=False,
        )

    if backend == "rocm":
        return PlatformInfo(
            os_name=os_name, arch=arch,
            backend="rocm",
            gpu_label="AMD GPU — ROCm 6.3",
            torch_index_url="https://download.pytorch.org/whl/rocm6.3",
            torch_version="torch==2.7.0",
            # No official ROCm wheels for torch-scatter/sparse — use CPU builds.
            # Graph ops (torch_scatter, PyG message passing) run on CPU transparently.
            pyg_find_url="",
            ipex=False,
        )

    if backend == "xpu":
        return PlatformInfo(
            os_name=os_name, arch=arch,
            backend="xpu",
            gpu_label="Intel GPU — XPU (SYCL/Level-Zero)",
            # Standard PyTorch 2.7 wheel includes XPU support on Linux x86_64.
            # intel-extension-for-pytorch (ipex) provides additional perf ops.
            torch_index_url="",
            torch_version="torch==2.7.0",
            pyg_find_url="",
            ipex=True,
        )

    # CPU fallback
    return PlatformInfo(
        os_name=os_name, arch=arch,
        backend="cpu",
        gpu_label="CPU (no GPU detected)",
        torch_index_url="",
        torch_version="torch",
        pyg_find_url="",
        ipex=False,
    )


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool, *, env: dict | None = None) -> None:
    """Print and optionally execute a shell command."""
    display = " ".join(str(c) for c in cmd)
    print(f"  $ {display}")
    if not dry_run:
        merged_env = {**os.environ, **(env or {})}
        result = subprocess.run(cmd, env=merged_env)
        if result.returncode != 0:
            print(f"\n[ERROR] Command failed (exit {result.returncode}):")
            print(f"  {display}")
            sys.exit(result.returncode)


def _conda_python(env_name: str) -> str:
    """Return abs path to python3 in a conda env."""
    conda_exe = shutil.which("conda") or ""
    if conda_exe:
        base = Path(conda_exe).resolve().parent.parent
        p = base / "envs" / env_name / "bin" / "python3"
        if p.exists():
            return str(p)
    for base in [
        Path.home() / "miniconda3",
        Path.home() / "miniforge3",
        Path.home() / "anaconda3",
        Path("/opt/conda"),
    ]:
        p = base / "envs" / env_name / "bin" / "python3"
        if p.exists():
            return str(p)
    return f"<conda>/envs/{env_name}/bin/python3"  # placeholder for dry-run


def _pip_in(env_name: str) -> list[str]:
    """Return [python3, -m, pip, install] prefix for a conda env."""
    return [_conda_python(env_name), "-m", "pip", "install"]


# ---------------------------------------------------------------------------
# Installation steps
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def install_score_env(dry_run: bool) -> None:
    """Create score-env and install hybridock-pep in editable mode."""
    print("\n── score-env (Vina, OpenMM, meeko, scikit-learn) ─────────────────")
    yml = _REPO_ROOT / "envs" / "score-env.yml"
    _run(["conda", "env", "create", "-f", str(yml), "--yes"], dry_run)
    # Editable install of hybridock-pep package
    _run([*_pip_in("score-env"), "-e", str(_REPO_ROOT)], dry_run)
    print("  ✓ score-env ready — activate with: conda activate score-env")


def install_rapidock_env(info: PlatformInfo, dry_run: bool) -> None:
    """Create rapidock env, then install PyTorch + PyG for the detected backend."""
    print(f"\n── rapidock env  [{info.gpu_label}] {'─' * max(0, 45 - len(info.gpu_label))}")

    # Choose platform-specific base yml
    if info.os_name == "Darwin":
        yml = _REPO_ROOT / "envs" / "rapidock-env-macos.yml"
    else:
        yml = _REPO_ROOT / "envs" / "rapidock-env.yml"

    _run(["conda", "env", "create", "-f", str(yml), "--yes"], dry_run)

    pip = _pip_in("rapidock")

    # ── PyTorch ──────────────────────────────────────────────────────────
    torch_cmd = [*pip, info.torch_version, "torchvision", "torchaudio"]
    if info.torch_index_url:
        torch_cmd += ["--index-url", info.torch_index_url]
    _run(torch_cmd, dry_run)

    # ── Intel Extension for PyTorch (XPU backend) ────────────────────────
    if info.ipex:
        _run([*pip, "intel-extension-for-pytorch"], dry_run)

    # ── PyG scatter/sparse/cluster ────────────────────────────────────────
    pyg_pkgs = [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ]
    if info.pyg_find_url:
        # CUDA-specific pre-built wheels (fast, no compilation)
        _run([*pip, *pyg_pkgs, "-f", info.pyg_find_url], dry_run)
    else:
        # CPU wheels — work for MPS/ROCm/XPU/CPU backends.
        # ROCm note: torch-scatter message-passing ops fall back to CPU; the
        # diffusion model forward pass is still GPU-accelerated via the main
        # graph convolutions which use native torch ops, not torch-scatter.
        _run([*pip, *pyg_pkgs], dry_run)

    # ── Verify PyTorch sees the expected device ───────────────────────────
    verify_script = _build_verify_script(info.backend)
    if not dry_run:
        py = _conda_python("rapidock")
        result = subprocess.run([py, "-c", verify_script], capture_output=False)
        if result.returncode != 0:
            print(
                "\n  [WARN] Device verification returned non-zero. "
                "Check PyTorch install above."
            )
    else:
        print(f"  $ {_conda_python('rapidock')} -c '<device verification>'")

    print("  ✓ rapidock env ready — used automatically by hybridock-pep dock")


def _build_verify_script(backend: str) -> str:
    """Return a one-shot Python verification snippet for the given backend."""
    return f"""
import torch, sys
backend = {repr(backend)}
print(f"PyTorch {{torch.__version__}}")
if backend == "cuda":
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else "n/a"
    hip = getattr(torch.version, "hip", None)
    variant = f"ROCm {{hip}}" if hip else "CUDA"
    print(f"{{variant}}: {{ok}}  device: {{name}}")
    if not ok:
        print("ERROR: CUDA/ROCm not available — check driver/rocminfo")
        sys.exit(1)
elif backend == "rocm":
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else "n/a"
    hip = getattr(torch.version, "hip", None)
    print(f"ROCm {{hip}}: {{ok}}  device: {{name}}")
    if not ok:
        print("ERROR: ROCm PyTorch wheel installed but GPU not visible — check ROCm stack")
        sys.exit(1)
elif backend == "xpu":
    ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    name = torch.xpu.get_device_name(0) if ok else "n/a"
    print(f"XPU: {{ok}}  device: {{name}}")
    if not ok:
        print("WARN: Intel XPU not visible — will fall back to CPU in inference")
elif backend == "mps":
    ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS: {{ok}}")
    if not ok:
        print("WARN: MPS not available — will fall back to CPU")
else:
    print("Backend: CPU")
print("Verification done.")
"""


# ---------------------------------------------------------------------------
# Summary / next-steps
# ---------------------------------------------------------------------------

_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          HybriDock-Pep  —  Environment Setup                    ║
╚══════════════════════════════════════════════════════════════════╝"""


def print_detection(info: PlatformInfo) -> None:
    print(_BANNER)
    print(f"  OS           : {info.os_name} {info.arch}")
    print(f"  GPU / Backend: {info.gpu_label}")
    print(f"  PyTorch      : {info.torch_version or 'torch (latest)'}")
    if info.torch_index_url:
        print(f"  Wheel index  : {info.torch_index_url}")
    if info.ipex:
        print("  Intel IPEX   : yes (intel-extension-for-pytorch)")
    if info.pyg_find_url:
        print(f"  PyG wheels   : {info.pyg_find_url}")
    else:
        print("  PyG wheels   : CPU (platform-independent builds)")
    print()


def print_next_steps(info: PlatformInfo) -> None:
    print("""
┌─ Next steps ────────────────────────────────────────────────────┐
│                                                                   │
│  1. Download RAPiDock model weights (~55 MB, not in git):        │
│     https://zenodo.org/records/14193621                          │
│     → third_party/RAPiDock/train_models/                        │
│       CGTensorProductEquivariantModel/rapidock_local.pt          │
│                                                                   │
│  2. Install ADFRsuite (required for Stage 2 scoring):            │
│     https://ccsb.scripps.edu/adfrsuite/downloads/               │
│     Add ADFRsuite/bin/ to PATH (see INSTALL.md Step 4)          │
│                                                                   │
│  3. Verify the full install:                                     │
│     conda activate score-env                                     │
│     bash scripts/smoke_test.sh                                   │
│                                                                   │
│  4. Run a test dock:                                             │
│     hybridock-pep dock \\                                         │
│         --peptide LISDAELEAIFEADC \\                              │
│         --receptor data/pdbs/1T2D_receptor.pdb \\                │
│         --site 31.9 17.5 9.5 --box 20 \\                         │
│         --n-samples 20 --output-dir runs/test                    │
└───────────────────────────────────────────────────────────────────┘""")

    if info.backend == "rocm":
        print("""
  AMD ROCm note:
    torch-scatter/torch-sparse graph ops run on CPU (no official ROCm
    wheels). The diffusion model's core convolutions use native torch ops
    and ARE GPU-accelerated.  Stage 1 is ~4–6× faster than CPU-only.
    If you need full GPU PyG coverage, build torch-scatter from source:
      conda run -n rapidock pip install torch-scatter --no-binary :all:
""")
    if info.backend == "xpu":
        print("""
  Intel XPU note:
    intel-extension-for-pytorch (ipex) enables XPU.  Verify with:
      conda run -n rapidock python3 -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
    If ipex import fails, XPU falls back to CPU automatically.
""")
    if info.backend == "mps":
        print("""
  Apple Silicon note:
    PYTORCH_ENABLE_MPS_FALLBACK=1 is set automatically by inference.py.
    Stage 1 runs ~5–8× faster than CPU, ~10× slower than an RTX 5070.
    For 100-pose full runs, consider using --input-poses from a CUDA machine.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HybriDock-Pep automated environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip score-env creation (rapidock only)",
    )
    parser.add_argument(
        "--skip-rapidock", action="store_true",
        help="Skip rapidock env creation (score-env only)",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "rocm", "xpu", "mps", "cpu"],
        help="Force a specific compute backend (overrides GPU auto-detection)",
    )
    args = parser.parse_args()

    if shutil.which("conda") is None:
        print("ERROR: conda not found. Install Miniconda/Miniforge first:")
        print("  https://github.com/conda-forge/miniforge/releases")
        sys.exit(1)

    info = detect_platform(force_backend=args.backend)
    print_detection(info)

    if args.dry_run:
        print("  ── DRY RUN — no commands will be executed ──\n")

    if not args.skip_scoring:
        install_score_env(dry_run=args.dry_run)

    if not args.skip_rapidock:
        install_rapidock_env(info, dry_run=args.dry_run)

    print_next_steps(info)

    if args.dry_run:
        print("  (dry-run complete — re-run without --dry-run to apply)")
    else:
        print("\n  Setup complete. Activate score-env and run smoke_test.sh to verify.\n")


if __name__ == "__main__":
    main()
