#!/usr/bin/env python3
"""Compatibility fixes so the bundled RAPiDock runs under Python 3.8 + numpy 1.24 (the newest numpy for Python 3.8), which is what
NVIDIA's Jetson build of PyTorch requires. Idempotent; it edits the files in place and says what it changed.

    python3 deploy/agx/patch-rapidock-py38.py [path/to/third_party/RAPiDock]

1. numpy 1.24 cannot turn a list of torch tensors into an array (numpy 2 can), so the per-residue language-model embeddings are
   converted with np.asarray first.
"""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "third_party" / "RAPiDock"
FIXES = [
    (root / "dataset" / "protein_feature.py",
     "                    lm_embedding_chain[i]\n                    for i in [",
     "                    np.asarray(lm_embedding_chain[i])\n                    for i in ["),
]
changed = 0
for path, old, new in FIXES:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched: {path.relative_to(root)}")
    elif old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"patched: {path.relative_to(root)}")
        changed += 1
    else:
        sys.exit(f"cannot find the code to patch in {path} (RAPiDock changed?)")
print(f"{changed} file(s) changed")
