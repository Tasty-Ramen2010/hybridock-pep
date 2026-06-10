"""E18 Stage 3 — ESM-2 attention cooperativity (runs in the `rapidock` env).

For each unique peptide sequence, run ESM-2, average the attention maps across
layers/heads to a symmetric LxL inter-residue coupling matrix C_ij, and reduce to a
per-residue coupling = mean_j C_ij (off-diagonal). This discounts the dihedral-
independence overcount in Stage 2: coupled residues share conformational freedom.

Writes /tmp/e18_esm_coupling.json: {sequence: [per_residue_coupling,...], "_mean": x}
Consumed by e18_train_eval.py to form: ln W_eff = Σ ln n_basin_i (1 - λ·coupling_i).

Run with the rapidock python:
  <rapidock>/bin/python scripts/e18_esm_coupling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def collect_sequences():
    seqs = set()
    for f in ("/tmp/e18_cr.json", "/tmp/e18_pb.json"):
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
    n_layers = model.num_layers

    seqs = collect_sequences()
    print(f"ESM-2 coupling for {len(seqs)} unique peptide sequences on {dev}", flush=True)
    out = {}
    for i, s in enumerate(seqs):
        _, _, toks = bc([("p", s)])
        toks = toks.to(dev)
        with torch.no_grad():
            res = model(toks, repr_layers=[], need_head_weights=True, return_contacts=False)
        # attentions: [B, layers, heads, L+2, L+2] (incl BOS/EOS)
        att = res["attentions"][0]  # layers, heads, L+2, L+2
        att = att.mean(dim=(0, 1))  # average layers+heads -> (L+2, L+2)
        att = att[1:-1, 1:-1].float().cpu().numpy()  # strip BOS/EOS
        att = 0.5 * (att + att.T)  # symmetrize
        L = att.shape[0]
        if L != len(s):
            out[s] = [0.0] * len(s)
            continue
        np.fill_diagonal(att, 0.0)
        # per-residue coupling = mean off-diagonal attention, normalized to [0,1]
        per_res = att.sum(1) / max(1, L - 1)
        mx = per_res.max() if per_res.max() > 0 else 1.0
        per_res = (per_res / mx).tolist()
        out[s] = per_res
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(seqs)}", flush=True)
    allv = [v for lst in out.values() for v in lst]
    out["_mean"] = float(np.mean(allv)) if allv else 0.0
    Path("/tmp/e18_esm_coupling.json").write_text(json.dumps(out))
    print(f"wrote /tmp/e18_esm_coupling.json  (mean per-res coupling {out['_mean']:.3f})", flush=True)


if __name__ == "__main__":
    main()
