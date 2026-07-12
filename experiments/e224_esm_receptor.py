"""E224 — ESM protein-LM receptor embeddings: does a real protein language model capture receptor binding
propensity (the hidden variable) better than ProtDCal's 0.15 / pocket-means 0.049? Runs in rapidock env
(ESM + CUDA). Embeds the 616 PPIKB receptors (>=5 peptides), mean-pools, caches → data/e224_esm_recep.npz.
Then (in score-env, separate step) tests receptor-mean prediction: ESM vs ProtDCal vs composition.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
OUT = ROOT / "data" / "e224_esm_recep.npz"


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ppikb_main_clean.jsonl")]
    fam = defaultdict(list)
    for r in rows:
        fam[r["protein_seq"]].append(r)
    recs = [(k, v) for k, v in fam.items() if len(v) >= 5 and len(k) >= 20 and set(k) <= AA]
    rseq = [k for k, _ in recs]
    ymean = np.array([float(np.mean([x["y"] for x in v])) for _, v in recs])
    print(f"embedding {len(rseq)} receptors with ESM-2...", flush=True)

    import esm  # noqa: PLC0415
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval().cuda()
    embs = []
    with torch.no_grad():
        for i, s in enumerate(rseq):
            s = s[:1022]
            _, _, toks = bc([("r", s)])
            toks = toks.cuda()
            out = model(toks, repr_layers=[30])["representations"][30][0, 1:len(s) + 1]
            embs.append(out.mean(0).cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rseq)}", flush=True)
    E = np.array(embs, dtype=np.float32)
    np.savez(OUT, emb=E, ymean=ymean, seqs=np.array(rseq, dtype=object))
    print(f"saved {OUT.name}  shape={E.shape}", flush=True)


if __name__ == "__main__":
    main()
