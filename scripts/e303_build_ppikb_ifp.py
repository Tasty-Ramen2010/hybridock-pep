"""E303 — build IFP + geometry for the PPIKB complexes we have raw RCSB structures for.

PPIKB ships as sequence/pocket descriptors with NO receptor/peptide split, so IFP could not be computed on
it. But data/rcsb_full/ holds the raw RCSB structures for ~1193 of the 1209 PPIKB complexes, and 804 of
those are NOT in PDBbind-925 — i.e. genuinely NEW IFP-computable data.

This script splits each raw structure into receptor + peptide chains and computes the SAME production
geometry (compute_geometry_features) and IFP (compute_ifp) used everywhere else. Correctness safeguard:
the peptide chain is chosen by sequence identity to the PPIKB peptide sequence and the choice is ASSERTED
(difflib ratio >= MIN_ID) — so we never silently split the wrong chain. Complexes that fail the assert,
lack a clear receptor, or error out are skipped and counted.

Output: data/e303_ppikb_ifp_cache.json (per complex: pdb, geom[17], ifp[19], y, q, rseq, pep_id).

Run: OMP_NUM_THREADS=1 ~/miniconda3/envs/score-env/bin/python scripts/e303_build_ppikb_ifp.py [--limit N]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import tempfile
from collections import defaultdict

import numpy as np

from hybridock_pep.scoring.geometry_features import compute_geometry_features
from hybridock_pep.scoring.interaction_map import (
    IFP_FEATURE_ORDER,
    _CRYSTAL_GEOM_ORDER,
    compute_ifp,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM = list(_CRYSTAL_GEOM_ORDER)
MIN_ID = 0.60          # min difflib identity(extracted peptide chain, PPIKB peptide seq)
MIN_REC_LEN = 25       # receptor chain must have >= this many CA residues

_3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def net_charge(seq: str) -> int:
    return sum((c in "KR") - (c in "DE") for c in seq.upper())


def parse_chains(path: str) -> dict[str, dict]:
    """Return {chain_id: {'ca': caseq, 'lines': [atom lines]}} using ATOM records, FIRST MODEL only.

    NMR ensembles repeat every chain across MODEL blocks; reading only model 1 avoids concatenating
    duplicate copies (which would inflate chain length and break the peptide-chain match).
    """
    ca: dict[str, list] = defaultdict(list)
    lines: dict[str, list] = defaultdict(list)
    started = False
    for ln in open(path):
        if ln.startswith("MODEL"):
            if started:          # second MODEL → stop (keep only model 1)
                break
            started = True
            continue
        if ln.startswith("ENDMDL"):
            break
        if not ln.startswith("ATOM"):
            continue
        ch = ln[21]
        lines[ch].append(ln)
        if ln[12:16].strip() == "CA":
            ca[ch].append(_3TO1.get(ln[17:20].strip(), "X"))
    return {c: {"ca": "".join(ca[c]), "lines": lines[c]} for c in lines}


def _pep_score(chain_ca: str, pep_seq: str) -> float:
    """Identity of a chain to the peptide: max(global ratio, containment of the chain in pep_seq).

    Containment rescues crystallographically truncated peptides (only part of the peptide resolved):
    the resolved CA-seq is still a contiguous block of pep_seq, so block_len/len(chain) ≈ 1.0.
    """
    if not chain_ca:
        return 0.0
    sm = difflib.SequenceMatcher(None, chain_ca, pep_seq)
    block = sm.find_longest_match(0, len(chain_ca), 0, len(pep_seq)).size
    return max(sm.ratio(), block / len(chain_ca))


def split(path: str, pep_seq: str):
    """Pick (receptor_chain, peptide_chain) by identity; return (rec_lines, pep_lines, rec_ca, pep_ca, id+) or None."""
    chains = parse_chains(path)
    if len(chains) < 2:
        return None
    # peptide chain = max identity to pep_seq, among chains not absurdly longer than the peptide
    cand = []
    for c, d in chains.items():
        if not d["ca"]:
            continue
        if len(d["ca"]) > len(pep_seq) + 10:   # too long to be this peptide (guards against receptor)
            continue
        cand.append((_pep_score(d["ca"], pep_seq), c))
    if not cand:
        return None
    idr, pep_c = max(cand)
    if idr < MIN_ID:
        return None
    # receptor = longest chain that is not the peptide chain
    rec_c = max((c for c in chains if c != pep_c), key=lambda c: len(chains[c]["ca"]), default=None)
    if rec_c is None or len(chains[rec_c]["ca"]) < MIN_REC_LEN:
        return None
    return (chains[rec_c]["lines"], chains[pep_c]["lines"],
            chains[rec_c]["ca"], chains[pep_c]["ca"], idr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ppikb: dict[str, dict] = {}
    for line in open(os.path.join(ROOT, "data/ppikb_clean.jsonl")):
        r = json.loads(line)
        p = (r.get("pdb") or "").lower()
        if p and r.get("seq") and r.get("protein_seq") and np.isfinite(r.get("y", np.nan)):
            ppikb.setdefault(p, r)
    cache_pdbs = {d["pdb"].lower() for d in json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))}
    todo = [p for p in ppikb if p not in cache_pdbs
            and os.path.exists(os.path.join(ROOT, f"data/rcsb_full/{p}.pdb"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"PPIKB complexes to build (new, with structure): {len(todo)}", flush=True)

    out = []
    skip = defaultdict(int)
    ids = []
    tmp = tempfile.mkdtemp(prefix="e303_")
    rec_p = os.path.join(tmp, "rec.pdb")
    pep_p = os.path.join(tmp, "pep.pdb")
    for i, pid in enumerate(todo):
        r = ppikb[pid]
        pep_seq = r["seq"]
        try:
            sp = split(os.path.join(ROOT, f"data/rcsb_full/{pid}.pdb"), pep_seq)
        except Exception:  # noqa: BLE001
            skip["parse_error"] += 1
            continue
        if sp is None:
            skip["no_valid_split"] += 1
            continue
        rec_lines, pep_lines, rec_ca, pep_ca, idr = sp
        ids.append(idr)
        open(rec_p, "w").writelines(rec_lines)
        open(pep_p, "w").writelines(pep_lines)
        try:
            f = compute_ifp(rec_p, pep_p)
            g = compute_geometry_features(pep_p, rec_p)
        except Exception:  # noqa: BLE001
            skip["feature_error"] += 1
            continue
        if g is None:
            skip["geom_none"] += 1
            continue
        g = {**g, "length": len(pep_ca)}
        try:
            geomv = [float(g[k]) for k in GEOM]
        except KeyError:
            skip["geom_key"] += 1
            continue
        out.append({"pdb": pid, "geom": geomv,
                    "ifp": [float(f[k]) for k in IFP_FEATURE_ORDER],
                    "y": float(r["y"]), "q": net_charge(pep_seq), "rseq": rec_ca, "pep_id": idr})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} processed, {len(out)} built", flush=True)

    json.dump(out, open(os.path.join(ROOT, "data/e303_ppikb_ifp_cache.json"), "w"))
    print(f"\nbuilt {len(out)} | skipped {dict(skip)}")
    if ids:
        a = np.array(ids)
        print(f"peptide-chain identity: median {np.median(a):.2f}, min {a.min():.2f}, "
              f">=0.9: {int((a>=0.9).sum())}/{len(a)}")
    print("wrote data/e303_ppikb_ifp_cache.json")


if __name__ == "__main__":
    main()
