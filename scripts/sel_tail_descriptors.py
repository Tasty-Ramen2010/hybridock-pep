#!/usr/bin/env python
"""The descriptors we never built: TAIL statistics, not means — what actually separates a
cognate from a near-miss.

THE STRUCTURAL GAP IN OUR FEATURE SET. Every one of our 18 non-Rosetta descriptors is a POOLED
AVERAGE over the whole interface -- mean burial, mean hydrophobicity, fraction of contacts that
are apolar, a product of two means. That is the wrong shape of statistic for this problem, and
the reason is simple: a near-miss in a repeat groove does not have a shifted average. It has the
right overall composition sitting in the wrong sockets. Shift a peptide by one register and the
mean apolar fraction barely moves, while a handful of contacts become catastrophic -- a buried
lysine against a buried arginine, an aspartate desolvated against leucine with nothing to pay for
it. Means average those away. That is exactly what we have been measuring.

So this computes the same contacts and keeps the TAILS instead:

  worst_pair          the single most unfavourable residue-residue contact in the interface
  n_clash_charge      buried like-charge pairs, which a correct register does not have
  n_unpaid_polar      buried polar or charged residues facing apolar with no partner -- the
                      classic unsatisfied-buried-polar penalty, absent from our set entirely
  contact_gini        how unequally contacts are distributed along the peptide. A real binder
                      anchors a few residues deeply; a mis-registered one smears contact evenly
  contact_entropy     the same thing in bits, which does not assume an ordering
  n_anchor            peptide residues buried past an anchor-like threshold
  seg_centroid/spread WHERE along the peptide the contacts sit, and how spread out

AND THE CEILING QUESTION UNDERNEATH. If these add nothing on CO-FOLDED poses -- where the geometry
is trustworthy and ref2015 already reaches 0.772 -- then no descriptor is missing and the deficit
is entirely pose generation. If they add, we have been measuring the wrong shape of thing all
along, and that is fixable without a better generator.

Usage: sel_tail_descriptors.py [--arm cofold|docked] [workers]
Output: logs/sel_tail_<arm>.jsonl
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]
BROKEN = {"n3", "n7", "pc21", "pc26"}
CUT = 4.5

APOLAR = set("AVLIMFWPCG")
POLAR = set("STNQYH")
POS, NEG = set("KR"), set("DE")
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_res(pdb: Path):
    """[(one-letter, (n_atoms,3) coords)] in file order, heavy atoms only."""
    cur, out = None, []
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM") or line[76:78].strip() == "H":
            continue
        key = (line[21], line[22:27])
        if key != cur:
            out.append([AA3.get(line[17:20].strip(), "X"), []])
            cur = key
        out[-1][1].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return [(a, np.array(c)) for a, c in out if c]


def pair_penalty(a: str, b: str) -> float:
    """Unfavourability of one buried residue-residue contact. Higher is worse."""
    if (a in POS and b in POS) or (a in NEG and b in NEG):
        return 3.0                                   # buried like charges
    if (a in POS and b in NEG) or (a in NEG and b in POS):
        return -2.0                                  # salt bridge
    if a in APOLAR and b in APOLAR:
        return -1.0                                  # packed core
    if (a in APOLAR) != (b in APOLAR):
        return 1.5 if (a in POS | NEG or b in POS | NEG) else 0.8   # unpaid desolvation
    return -0.3                                      # polar-polar


def do_cell(args: tuple) -> dict:
    pep_pdb, rec_pdb, pep, binder = args
    row = {"peptide": pep, "binder": binder, "cognate": int(pep == binder)}
    try:
        P, R = read_res(Path(pep_pdb)), read_res(Path(rec_pdb))
    except (OSError, ValueError) as exc:
        row["error"] = str(exc)
        return row
    if not P or not R:
        row["error"] = "empty"
        return row

    pens, per_res, pairs = [], [], 0
    for i, (aa, pc) in enumerate(P):
        n_i = 0
        for bb, rc in R:
            d = np.linalg.norm(pc[:, None, :] - rc[None, :, :], axis=2)
            k = int((d < CUT).sum())
            if k:
                n_i += k
                pens.append(pair_penalty(aa, bb))
                pairs += 1
        per_res.append(n_i)
    n = np.array(per_res, dtype=float)
    pe = np.array(pens) if pens else np.array([0.0])
    tot = n.sum() or 1.0
    p = n / tot
    nz = p[p > 0]
    srt = np.sort(n)
    idx = np.arange(1, len(srt) + 1)

    row["best"] = {
        "worst_pair": float(pe.max()),
        "mean_pair": float(pe.mean()),
        "n_clash_charge": float((pe >= 3.0).sum()),
        "n_unpaid_polar": float((pe == 1.5).sum()),
        "frac_bad_pairs": float((pe > 0).mean()),
        "n_saltbridge": float((pe <= -2.0).sum()),
        "contact_gini": float((2 * (idx * srt).sum()) / (len(srt) * srt.sum()) -
                              (len(srt) + 1) / len(srt)) if srt.sum() else 0.0,
        "contact_entropy": float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0,
        "n_anchor": float((n >= np.percentile(n, 75) * 1.5).sum()),
        "max_res_contacts": float(n.max()),
        "frac_res_zero": float((n == 0).mean()),
        "seg_centroid": float((p * np.arange(len(n))).sum() / max(len(n) - 1, 1)),
        "seg_spread": float(np.sqrt((p * (np.arange(len(n)) -
                                          (p * np.arange(len(n))).sum()) ** 2).sum()) /
                            max(len(n), 1)),
        "n_pairs": float(pairs),
    }
    return row


def main() -> None:
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "cofold"
    workers = next((int(a) for a in sys.argv[1:] if a.isdigit()), 10)
    out = ROOT / f"logs/sel_tail_{arm}.jsonl"

    jobs = []
    for p in ORDER:
        for b in ORDER:
            name = f"{p}__{b}"
            af3 = b in BROKEN
            if arm == "cofold":
                pep = ROOT / f"runs/coventry/{'boltz_af3' if af3 else 'boltz'}/{name}/peptide.pdb"
                if not pep.exists():
                    pep, af3 = ROOT / f"runs/coventry/boltz/{name}/peptide.pdb", False
            else:
                cand = sorted((ROOT / f"runs/coventry/hybridock_ft/{p}__{b}B").glob("rank*.pdb"))
                pep, af3 = (cand[0] if cand else Path("/nonexistent")), False
            rec = ROOT / ("datasets/coventry/binders_af3" if af3 else
                          "datasets/coventry/binders") / f"{b}_1b1.pdb"
            if pep.exists() and rec.exists():
                jobs.append((str(pep), str(rec), p, b))
    print(f"{arm}: {len(jobs)} cells, {workers} workers", flush=True)
    with out.open("w") as fh, ProcessPoolExecutor(workers) as ex:
        for k, row in enumerate(ex.map(do_cell, jobs, chunksize=4), 1):
            fh.write(json.dumps(row) + "\n")
    print(f"TAIL_DONE {out.name}", flush=True)


if __name__ == "__main__":
    main()
