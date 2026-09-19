#!/usr/bin/env python
"""Build a synthetic all-by-all selectivity benchmark from our own corpus.

THE PROBLEM THIS SOLVES.  Our calibrated dG model scores at CHANCE on the Coventry grid
(mean cognate rank 9.4 of 18; random is 9.5) while Rosetta ref2015 interface energy reaches
8.3.  The mechanism is measured: fix a peptide and swap all 18 binders, and our model moves by
0.22 kcal/mol -- less than its own ~1.2 kcal/mol error bar.  It reads the peptide, not the
pair, because it was trained for cross-complex ABSOLUTE affinity where peptide-sequence
descriptors carry the signal.  ref2015 interface energy is by construction a DIFFERENCE,
E(complex) - E(receptor) - E(peptide), so every peptide-internal and receptor-internal term
cancels and only pair physics survives.

We already knew this in June and never acted on it (project_absolute_kd_ceiling_jun14):
within-family discrimination scores tau = -0.027 for sequence-only, -0.010 for seq+pocket, and
+0.047 for structure.  Sequence models are BLIND here; structure is the only positive signal.

To train a selectivity-specific scorer we need training data shaped like the task: many
peptides against many receptors, with the true partner known.  That data does not exist at
scale in the literature, but it can be MANUFACTURED from crystal structures.  For a block of N
length-matched complexes, thread every peptide sequence onto every receptor's own cognate
peptide backbone.  The diagonal is the crystallographic pair; the off-diagonal is a peptide on
a receptor it was never selected for.  That is exactly an 18x18 grid, self-supervised, and we
can make as many as we have GPU-free CPU hours for.

WHY THREADING RATHER THAN DOCKING.  N^2 docking runs per block is unaffordable, and it would
confound the question: we would be measuring pose generation again, which we already know is
broken.  Threading gives every sequence the SAME backbone -- the receptor's own cognate one --
so pose quality is held constant and only chemistry varies.  That isolates the thing we are
trying to learn.  The cost is a train/test distribution shift (threaded here, docked at test),
which is measured explicitly rather than assumed away.

BLOCK CONSTRUCTION RULES
  * one block = N complexes whose peptides are all EXACTLY the same length (threading requires
    it), all from distinct PDB ids.
  * no two peptides in a block above SEQ_ID_MAX identity -- otherwise the "non-cognate" label
    is simply wrong, and we would be training the model to call a near-twin a non-binder.
  * --composition-matched makes the block HARDER and much more like the real target. Coventry's
    18 peptides are not unrelated: they are repeats of a handful of motifs (LKLKLK..., PVPVPV...,
    YDYDYD...) that share composition while differing in sequence. A block of unrelated peptides
    from unrelated proteins is a far easier discrimination than that. This flag requires block
    members to have SIMILAR amino-acid composition (cosine similarity >= COMP_SIM_MIN) while
    keeping sequence identity below SEQ_ID_MAX -- similar chemistry, different sequence, which is
    exactly the regime where a scorer has to actually read the interface rather than count
    hydrophobics.
  * every benchmark and holdout id is excluded, so downstream evaluations stay clean.

Usage: sel_build_blocks.py [--n 10] [--blocks 24] [--seed 0]
Output: data/sel_blocks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
CORPUS = ROOT / "data/longft_train_longgeo.csv"
OUT = ROOT / "data/sel_blocks.json"
#: two peptides more similar than this cannot sit in the same block -- the off-diagonal label
#: ("this peptide does not belong on that receptor") would be false for near-twins.
SEQ_ID_MAX = 0.40
#: for --composition-matched: minimum cosine similarity between amino-acid composition vectors
COMP_SIM_MIN = 0.55
#: peptides shorter than this have too few contacts for interface chemistry to discriminate
MIN_LEN = 8
MAX_LEN = 20


def pdb_id(cn: str) -> str:
    m = re.search(r"_([0-9][A-Za-z0-9]{3})(?:_|$)", cn)
    return m.group(1).upper() if m else cn


def seq_of(pdb: Path) -> str:
    aa3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
           "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
           "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
           "TYR": "Y", "VAL": "V", "MSE": "M"}
    out = []
    try:
        for line in pdb.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                out.append(aa3.get(line[17:20].strip(), "X"))
    except OSError:
        return ""
    return "".join(out)


def identity(a: str, b: str) -> float:
    """Ungapped identity at the best offset, normalised by the longer sequence."""
    if not a or not b:
        return 0.0
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    best = max(sum(1 for i, c in enumerate(lo) if hi[off + i] == c)
               for off in range(len(hi) - len(lo) + 1))
    return best / max(len(a), len(b))


AA20 = "ACDEFGHIKLMNPQRSTVWY"


def composition(seq: str) -> list[float]:
    """Amino-acid composition as a unit vector, for similarity between peptides."""
    import math
    v = [seq.count(a) for a in AA20]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def comp_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def excluded_ids() -> set[str]:
    """Every PDB id that appears in a benchmark or holdout, so evaluations stay honest."""
    bad = {"9CCE", "9CCF"}
    for f in ("data/bench_recentset_heldout.csv", "data/solenoid_pep_test.csv",
              "data/longft_val_scope25_authorsP.csv", "data/benchmark_expanded.csv"):
        p = ROOT / f
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            for k in ("name", "complex_name", "pdb"):
                if r.get(k):
                    bad.add(pdb_id(r[k]))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="complexes per block (block is n x n)")
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--composition-matched", action="store_true",
                    help=("require block members to share amino-acid composition. Makes the "
                          "discrimination much harder and much closer to the Coventry grid, "
                          "whose peptides are repeats of a few motifs."))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-per-len", type=int, default=3,
                    help=("cap blocks drawn from any one peptide length. Without it every "
                          "block comes from the biggest pool (8-mers), and the model would be "
                          "trained on one length while Coventry spans 8-18."))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    bad = excluded_ids()
    print(f"excluding {len(bad)} benchmark/holdout PDB ids", flush=True)

    by_len: dict[int, list[dict]] = defaultdict(list)
    seen_pdb: set[str] = set()
    for r in csv.DictReader(open(CORPUS)):
        cn = r["complex_name"]
        pid = pdb_id(cn)
        if pid in bad or pid in seen_pdb:
            continue
        n = int(r["pep_len"])
        if not (MIN_LEN <= n <= MAX_LEN):
            continue
        pep = Path(r["peptide_description"])
        rec = Path(r["protein_description"])
        if not (pep.exists() and rec.exists()):
            continue
        s = seq_of(pep)
        if len(s) != n or "X" in s:
            continue
        seen_pdb.add(pid)
        by_len[n].append({"complex": cn, "pdb": pid, "receptor": str(rec),
                          "peptide_pdb": str(pep), "seq": s, "len": n})
    print("candidates per length: "
          + " ".join(f"{k}:{len(v)}" for k, v in sorted(by_len.items())), flush=True)

    blocks = []
    # longest first: Coventry peptides run 8-18 aa and the long end is both scarcer and more
    # like the target task, so it gets first claim on its quota.
    lengths = sorted(by_len, key=lambda k: -k)
    used: set[str] = set()
    for L in lengths:
        pool = [c for c in by_len[L] if c["pdb"] not in used]
        rng.shuffle(pool)
        n_this = 0
        while len(blocks) < args.blocks and len(pool) >= args.n and n_this < args.max_per_len:
            chosen: list[dict] = []
            # Composition-matched blocks are built around a SEED rather than as a full clique.
            # Requiring every pair to clear the threshold needs a 10-clique in a graph whose
            # median edge weight is 0.44, which essentially never exists; anchoring on one seed
            # and taking its nearest neighbours by composition gives a tight, motif-like block --
            # which is how the Coventry peptides are actually organised (repeats of LK, PV, YD).
            if args.composition_matched:
                seed = pool[0]
                sv = composition(seed["seq"])
                ranked = sorted(pool[1:],
                                key=lambda c: -comp_sim(sv, composition(c["seq"])))
                chosen = [seed]
                for cand in ranked:
                    if comp_sim(sv, composition(cand["seq"])) < COMP_SIM_MIN:
                        break
                    if any(identity(cand["seq"], c["seq"]) > SEQ_ID_MAX for c in chosen):
                        continue
                    chosen.append(cand)
                    if len(chosen) == args.n:
                        break
            else:
                for cand in pool:
                    if any(identity(cand["seq"], c["seq"]) > SEQ_ID_MAX for c in chosen):
                        continue
                    chosen.append(cand)
                    if len(chosen) == args.n:
                        break
            if len(chosen) < args.n:
                break
            for c in chosen:
                used.add(c["pdb"])
            pool = [c for c in pool if c["pdb"] not in used]
            blocks.append({"block_id": len(blocks), "pep_len": L, "members": chosen})
            n_this += 1
        if len(blocks) >= args.blocks:
            break

    out_path = Path(args.out)
    out_path.write_text(json.dumps(blocks, indent=1))
    print(f"\nwrote {out_path}: {len(blocks)} blocks x {args.n}x{args.n} "
          f"= {len(blocks) * args.n * args.n} threadings")
    for b in blocks:
        ids = [m["pdb"] for m in b["members"]]
        print(f"  block {b['block_id']:2d}  len {b['pep_len']:2d}  {' '.join(ids)}")


if __name__ == "__main__":
    main()
