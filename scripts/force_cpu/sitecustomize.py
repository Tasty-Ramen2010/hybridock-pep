"""Make a GPU machine behave like one without a GPU, for testing the CPU path.

Put this directory on PYTHONPATH and Python imports it automatically at
interpreter startup -- before RAPiDock runs, which is what makes it work:

    PYTHONPATH=scripts/force_cpu KMP_DUPLICATE_LIB_OK=TRUE \\
        hybridock-pep dock --peptide ETFSDLWKLLPE ...

RAPiDock picks its device with a hardcoded `cuda -> mps -> xpu -> cpu` cascade
(third_party/RAPiDock/inference.py, and again in utils/inference_utils.py) and
offers no override. Rather than patching the submodule, this makes torch report
that MPS is unavailable, so the cascade genuinely falls through to CPU. Nothing
downstream is stubbed: the model runs on real CPU tensors, and the pipeline's own
log line confirms which branch was taken:

    [run_rapidock] backend optimization: CPU (threads tuned)

This exists so the "runs without a GPU" claim in RESULTS.md can be re-measured on
ordinary developer hardware instead of being taken on faith. On a machine that
genuinely has no GPU it is unnecessary -- the cascade lands on CPU by itself.

CUDA is deliberately left alone. Masking MPS is enough on Apple Silicon; to
simulate no-GPU on a CUDA box, set CUDA_VISIBLE_DEVICES="" instead, which the
driver honours without any patching.
"""

try:  # pragma: no cover - startup hook, exercised by running a dock
    import torch

    # Both are consulted: is_available() by the device cascade, is_built() by
    # some torch internals that would otherwise still route to Metal.
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
except Exception:
    # This directory lands on PYTHONPATH for BOTH conda environments, and
    # score-env has no torch. An unguarded import here would break every
    # score-env interpreter that starts, including the CLI itself.
    pass
