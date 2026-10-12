"""The shipped RAPiDock code must load EVERY checkpoint in weights/, strictly.

`longer_local.pt` (the long-peptide model) carries one layer the original RAPiDock model does not have,
`encoder.cross_type_embedding.weight`. Before the vendored model learned about it, `load_state_dict(strict=True)` raised
"Unexpected key(s) in state_dict" and every peptide of 13+ residues failed seconds into Stage 1. This loads each
checkpoint through the real `inference.load_model` in the rapidock environment, so that cannot come back unnoticed.

Skipped where the rapidock environment (torch, e3nn, torch_geometric) is not installed, e.g. the scoring-only CI job.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAPIDOCK = ROOT / "third_party" / "RAPiDock"
MODEL_DIR = RAPIDOCK / "train_models" / "CGTensorProductEquivariantModel"

SCRIPT = textwrap.dedent("""
    import json, sys, yaml, torch
    from argparse import Namespace
    sys.path.insert(0, ".")
    import inference as I
    d = "weights_dir"
    base = yaml.safe_load(open(d + "/model_parameters.yml"))
    out = {}
    for name in ("rapidock_local.pt", "rapidock_global.pt", "longer_local.pt"):
        args = Namespace(**base)
        model = I.load_model(args, d + "/" + name, torch.device("cpu"))
        out[name] = {"params": sum(p.numel() for p in model.parameters()), "cross_type": bool(args.use_cross_type_embedding)}
    print("RESULT " + json.dumps(out))
""").replace("weights_dir", str(MODEL_DIR))


def _rapidock_python() -> str | None:
    from hybridock_pep.sampling import rapidock_runner
    try:
        return rapidock_runner._find_rapidock_python()
    except RuntimeError:
        return None


@pytest.mark.skipif(not (MODEL_DIR / "model_parameters.yml").exists(), reason="RAPiDock submodule not checked out")
def test_every_shipped_checkpoint_loads_strictly_and_only_the_long_one_has_the_extra_layer(tmp_path):
    py = _rapidock_python()
    if not py:
        pytest.skip("rapidock conda environment not installed")
    weights = ROOT / "weights"
    for name in ("rapidock_local.pt", "rapidock_global.pt", "longer_local.pt"):
        if not (weights / name).is_file():
            pytest.skip(f"weights/{name} not present")
        if not (MODEL_DIR / name).exists():
            (MODEL_DIR / name).hardlink_to(weights / name) if hasattr(Path, "hardlink_to") else os.link(weights / name, MODEL_DIR / name)
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    proc = subprocess.run([py, "-c", SCRIPT], cwd=RAPIDOCK, env=env, capture_output=True, text=True, timeout=600)
    if "No module named" in proc.stderr:
        pytest.skip("rapidock environment is missing a dependency: " + proc.stderr.strip().splitlines()[-1])
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, f"load failed:\n{proc.stderr[-1500:]}"
    result = json.loads(line[len("RESULT "):])
    assert result["rapidock_local.pt"]["cross_type"] is False
    assert result["rapidock_global.pt"]["cross_type"] is False
    assert result["longer_local.pt"]["cross_type"] is True
    # the extra layer is a 4 x ns embedding: exactly that many more parameters, nothing else changed
    extra = result["longer_local.pt"]["params"] - result["rapidock_local.pt"]["params"]
    assert extra == 4 * 48
