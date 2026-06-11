"""Extract ESM-2 peptide embeddings for crystal-65 + PEPBI seqs (rapidock env, CPU ok).

Tests the multimodal lever (PPI-Affinity / multimodal-PPI use LM embeddings). ESM embeddings
are SEQUENCE-only -> immune to the PEPBI Rosetta-model structure confound, so we can fairly
use all 391 sequences. Mean-pooled per-peptide embedding. Writes /tmp/esm_affinity.json:
{seq: [embedding...]}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def collect():
    seqs = set()
    for f in ("/tmp/e19_cr.json", "/tmp/e18_pb.json"):
        p = Path(f)
        if p.exists():
            for r in json.loads(p.read_text()):
                s = r.get("seq", "")
                if 2 <= len(s) <= 60 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    seqs.add(s)
    return sorted(seqs)


def main():
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    seqs = collect()
    print(f"ESM-2 650M embeddings for {len(seqs)} unique seqs on {dev}", flush=True)
    out = {}
    for i, s in enumerate(seqs):
        _, _, toks = bc([("p", s)])
        with torch.no_grad():
            rep = model(toks.to(dev), repr_layers=[33])["representations"][33][0]
        emb = rep[1:len(s)+1].mean(0).float().cpu().numpy()  # mean-pool residues (strip BOS)
        out[s] = emb.tolist()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seqs)}", flush=True)
    Path("/tmp/esm_affinity.json").write_text(json.dumps(out))
    print(f"wrote /tmp/esm_affinity.json ({len(out)} seqs, dim={len(next(iter(out.values())))})")


if __name__ == "__main__":
    main()
