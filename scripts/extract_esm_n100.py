#!/usr/bin/env python3
"""
extract_esm_n100.py — Per-residue ESM-2 650M embeddings for gen_n100.

For each of the 57 gen_n100 complexes, embed:
  - the peptide sequence  → [n_pep, 1280]
  - the receptor pocket   → [n_rec, 1280]  (fragment context; pocket PDB
    is renumbered 1..N with no full-chain mapping available)

NOTE on fragment context: the pocket is a discontinuous crop renumbered
1..N, so ESM sees it as a contiguous fragment, not in full-chain context.
This is a known approximation for the architecture-validation pass. If the
ESM-GNN beats one-hot, upgrade to full-chain ESM with pocket indexing.

Uses the same model RAPiDock uses: esm2_t33_650M_UR50D (1280-dim, layer 33).

Run in rapidock env (GPU):
  python3 scripts/extract_esm_n100.py
Saves → logs/diagnosis/feats_gen_n100_esm.pkl
        { cn: {"pep": np[n_pep,1280], "rec": np[n_rec,1280]} }
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

D    = REPO / "logs" / "diagnosis"
BASE = Path("/home/igem/unknown_software/datasets/training_formatted_peppc")
GEN_N100_JSON = REPO / "logs" / "gen_n100" / "benchmark_results.json"
GEN_N100_ENC  = D / "feats_gen_n100.pkl"
OUT_PKL       = D / "feats_gen_n100_esm.pkl"

AA3to1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def read_seq(pdb_path: str) -> str:
    """One-letter sequence from Cα/Cβ residue order (matches read_residues)."""
    seen = set()
    order = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in ("CA", "CB"):
                continue
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            order.append(AA3to1.get(line[17:20].strip(), "X"))
    return "".join(order)


def main():
    import torch
    from esm import pretrained

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading esm2_t33_650M_UR50D on {dev}...", flush=True)
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model = model.eval().to(dev)
    bc = alphabet.get_batch_converter()

    def embed(seq: str) -> np.ndarray:
        if not seq:
            return np.zeros((0, 1280), np.float32)
        _, _, toks = bc([("x", seq)])
        toks = toks.to(dev)
        with torch.no_grad():
            out = model(toks, repr_layers=[33])["representations"][33]
        # strip BOS/EOS → [L, 1280]
        return out[0, 1:len(seq) + 1].cpu().numpy().astype(np.float32)

    bjson   = json.load(open(GEN_N100_JSON))
    enc_all = pickle.load(open(GEN_N100_ENC, "rb"))
    cxs = sorted(set(k[0] for k in enc_all))

    results: dict = {}
    t0 = time.time()
    for i, cn in enumerate(cxs):
        entry = bjson.get(cn, {}).get("pretrained", {})
        rec_pdb = BASE / cn / f"{cn}_protein_pocket.pdb"
        p0 = Path(entry.get("poses_dir", "")) / "pose_0.pdb"
        if not rec_pdb.exists() or not p0.exists():
            continue
        rec_seq = read_seq(str(rec_pdb))
        pep_seq = read_seq(str(p0))
        results[cn] = {"pep": embed(pep_seq), "rec": embed(rec_seq)}
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cxs)}  ({time.time()-t0:.0f}s)", flush=True)

    pickle.dump(results, open(OUT_PKL, "wb"), protocol=4)
    print(f"Done. {len(results)} complexes  ({time.time()-t0:.0f}s)")
    print(f"Saved → {OUT_PKL}")


if __name__ == "__main__":
    main()
