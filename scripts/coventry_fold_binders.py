#!/usr/bin/env python
"""Fold the 18 designed binders of Wu et al. 2025 from sequence alone.

Coventry's instruction: "if you just run these proteins through AF2 or AF3, you'll
get a pretty good structure of the binder (that's how we designed them), and then
you guys say you're good at docking."  We do exactly that with ESMFold, which is the
single-sequence folder we can run locally; no complex structure, no peptide, no
design model is used anywhere.  The C-terminal GS-His6 purification tag is stripped
before folding -- it is disordered and would only add a floppy tail near the groove.

pLDDT is written per model.  Anything below PLDDT_WARN is flagged: de novo designs
have no evolutionary neighbours, which is exactly the regime where a language-model
folder is least reliable, so a low-confidence binder is a caveat on every cell of
its column, not a silent input.

Usage: coventry_fold_binders.py [outdir]
Output: <outdir>/<binder>.pdb and logs/coventry_fold.json
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
PLDDT_WARN = 85.0
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/coventry/binders"


def strip_tag(seq: str) -> str:
    """Remove the C-terminal GS-linker + His tag used for purification."""
    return re.sub(r"(GS)?H{5,}$", "", seq.rstrip("*")).rstrip("GS") or seq


def main() -> None:
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data/coventry_binders.csv")))
    print(f"folding {len(rows)} binders -> {OUT}", flush=True)

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                 low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()               # fits comfortably on a 12 GB card
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    meta = {}
    for i, r in enumerate(rows, 1):
        name, seq = r["name"], strip_tag(r["sequence"])
        t0 = time.time()
        with torch.no_grad():
            enc = tok([seq], return_tensors="pt", add_special_tokens=False)
            out = model(enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda())
        pdb = model.output_to_pdb(out)[0]
        plddt = float(out["plddt"][0, :, 1].mean() * 100)
        (OUT / f"{name}.pdb").write_text(pdb)
        meta[name] = {"grid_label": r["grid_label"], "len_full": len(r["sequence"]),
                      "len_folded": len(seq), "plddt": round(plddt, 2),
                      "seconds": round(time.time() - t0, 1),
                      "low_confidence": plddt < PLDDT_WARN}
        flag = "  <-- LOW pLDDT" if plddt < PLDDT_WARN else ""
        print(f"  [{i:2d}/{len(rows)}] {name:12s} {len(seq):3d} aa  "
              f"pLDDT {plddt:5.1f}  {time.time() - t0:5.1f}s{flag}", flush=True)
        del out
        torch.cuda.empty_cache()

    (ROOT / "logs/coventry_fold.json").write_text(json.dumps(meta, indent=2))
    lows = [k for k, v in meta.items() if v["low_confidence"]]
    print(f"\nmean pLDDT {sum(v['plddt'] for v in meta.values()) / len(meta):.1f}")
    print(f"low-confidence ({PLDDT_WARN}): {lows or 'none'}")
    print("COVENTRY_FOLD_DONE", flush=True)
    # Hard exit. Interpreter teardown does not release the CUDA context here, so a normal
    # return leaves the process alive holding ~5 GB of host RAM and its GPU allocation
    # until something kills it -- which is exactly what happened on the first run.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
