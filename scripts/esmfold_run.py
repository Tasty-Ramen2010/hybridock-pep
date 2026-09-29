#!/usr/bin/env python
"""ESMFold arm for the length-balanced benchmark.

WHY ESMFOLD AND NOT BOLTZ-2 (Ram's call, and the leakage numbers back it). ESMFold's PDB cutoff
is 2020-05-01. Measured on this bench: 0 of 387 complexes were deposited before it, 358 after,
29 undated. Boltz-2's cutoff is 2023-06-01, which swallows essentially all 387. So this is the
co-folding arm that can actually go in a figure as a competitor rather than as a memorised
ceiling.

HOW A MONOMER FOLDER DOCKS A COMPLEX. ESMFold has no multimer mode. The standard workaround is
a poly-glycine linker plus a jump in the residue index across the join, so the trunk sees two
segments that are sequence-adjacent but position-distant. The linker residues are cut out of the
output before anything is scored. This is a real approximation and it is disclosed: a linker is
not the same as a chain break, and ESMFold was never trained on complexes at all.

transformers, not fair-esm: fair-esm's esmfold pulls in openfold's CUDA extensions, which is a
build fight on aarch64. transformers vendors the openfold utilities it needs in pure PyTorch.

Writes <out>/<name>_model_0.pdb with the receptor as chain A and the peptide as chain B, which
is exactly the layout scripts/boltz2_bench_score.py already reads, so the same superposition and
scoring path covers both arms with no second implementation.

Usage: esmfold_run.py --bench <csv> --out <dir> [--linker 25] [--chunk 64]
"""
from __future__ import annotations

import argparse, csv, gc, os, sys, time
from pathlib import Path

import torch

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def ca_seq(p: str) -> str:
    s, seen = [], set()
    for l in open(p, errors="ignore"):
        if not l.startswith("ATOM") or l[12:16].strip() != "CA":
            continue
        k = (l[21], l[22:27])
        if k in seen:
            continue
        seen.add(k)
        s.append(AA3.get(l[17:20].strip(), 'X'))
    return ''.join(s)


def split_and_relabel(pdb_text: str, n_rec: int, n_link: int, n_pep: int) -> str:
    """Drop the linker residues; receptor -> chain A, peptide -> chain B, renumbered from 1."""
    out, order, idx = [], [], {}
    for l in pdb_text.splitlines():
        if not l.startswith(("ATOM", "HETATM")):
            continue
        key = l[22:27]
        if key not in idx:
            idx[key] = len(order)
            order.append(key)
        i = idx[key]
        if i < n_rec:
            ch, num = "A", i + 1
        elif i < n_rec + n_link:
            continue                      # linker, never scored
        else:
            ch, num = "B", i - n_rec - n_link + 1
        out.append(f"{l[:21]}{ch}{num:>4d} {l[27:]}")
    return "\n".join(out) + "\nEND\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--linker", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1200)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.bench)))
    # smallest first, so a memory failure on a monster costs the monster and nothing before it
    rows.sort(key=lambda r: len(ca_seq(r["receptor"])))

    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(a.chunk)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"model ready; {len(rows)} complexes, linker={a.linker}G, chunk={a.chunk}", flush=True)

    ok = fail = skip = 0
    for i, r in enumerate(rows, 1):
        name = r["name"]
        dst = out / f"{name}_model_0.pdb"
        if dst.exists():
            skip += 1
            continue
        rseq = ca_seq(r["receptor"]); pseq = r["seq"].strip().upper()
        if not rseq or not pseq or 'X' in rseq or 'X' in pseq:
            print(f"[{i}/{len(rows)}] {name}: skip (X or empty)", flush=True); fail += 1; continue
        L = len(rseq) + a.linker + len(pseq)
        if L > a.max_tokens:
            print(f"[{i}/{len(rows)}] {name}: skip, {L} tokens > --max-tokens", flush=True)
            fail += 1; continue
        seq = rseq + "G" * a.linker + pseq
        t0 = time.time()
        try:
            enc = tok([seq], return_tensors="pt", add_special_tokens=False)
            enc = {k: v.cuda() for k, v in enc.items()}
            pos = torch.arange(L, dtype=torch.long)
            # the index jump IS the chain break: without it the linker reads as ordinary
            # backbone and the two segments are modelled as one continuous chain
            pos[len(rseq) + a.linker:] += 512
            enc["position_ids"] = pos.unsqueeze(0).cuda()
            with torch.no_grad():
                o = model(**enc)
            pdb = model.output_to_pdb(o)[0]
            dst.write_text(split_and_relabel(pdb, len(rseq), a.linker, len(pseq)))
            ok += 1
            print(f"[{i}/{len(rows)}] {name}: ok  {L} tokens  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(rows)}] {name}: FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            gc.collect(); torch.cuda.empty_cache()
    print(f"\ndone: {ok} folded, {fail} failed, {skip} already present", flush=True)


if __name__ == "__main__":
    main()
