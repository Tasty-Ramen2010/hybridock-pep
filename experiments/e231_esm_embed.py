"""E231 (embed phase, rapidock env) — embed the pilot receptor sequences with ESM-2-150M (same model +
mean-pooling as e224) so we can test ESM-vs-RISM orthogonality. Writes data/e231_pilot_esm.npz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import esm

ROOT = Path(__file__).resolve().parents[1]
man = json.load(open(ROOT / "data/e228_pilot_manifest.json"))
done = {json.loads(l)["rep_pdb"] for l in (ROOT / "data/e230_rism.jsonl").read_text().splitlines()}
recs = [r for r in man["receptors"] if r["peptides"][0]["pdb"] in done]
pdbs = [r["peptides"][0]["pdb"] for r in recs]
seqs = [r["rec_seq"].replace("/", "") for r in recs]

model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
bc = alphabet.get_batch_converter()
model.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
embs = []
with torch.no_grad():
    for i, s in enumerate(seqs):
        s = s[:1022]
        _, _, toks = bc([("r", s)])
        out = model(toks.to(dev), repr_layers=[30])["representations"][30][0, 1:len(s) + 1]
        embs.append(out.mean(0).cpu().numpy())
        print(f"  embedded {i+1}/{len(seqs)} {pdbs[i]}", flush=True)
E = np.array(embs)
np.savez(ROOT / "data/e231_pilot_esm.npz", emb=E, pdbs=np.array(pdbs, dtype=object))
print(f"=== wrote {E.shape} → data/e231_pilot_esm.npz ===")
