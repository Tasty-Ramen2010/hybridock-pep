#!/usr/bin/env python
"""Fold the SSM parent designs so the substitution model can finally see structure.

WHY. The sequence-only substitution model reaches AUC 0.729 against BLOSUM62's 0.712 -- it beats
a fifty-year-old lookup table by 0.017, and it moves our ddG MAE by 0.011 kcal/mol. The diagnosis
is not that the model is badly fitted, it is that with only wt->mut physicochemistry there is
nothing to know beyond generic substitution chemistry, which is exactly what BLOSUM encodes.

What BLOSUM cannot know is CONTEXT: putting a lysine into a buried core and putting the same
lysine on the surface are the same substitution and completely different events. That distinction
requires structure, and structure is what we do not have -- the tarball is scores only, and
Coventry said at 41:02 we would have to fold these ourselves.

THE SEQUENCES ARE RECOVERABLE WITHOUT ANY EXTERNAL LOOKUP. A saturation scan tests 19 of the 20
amino acids at each position, so the missing one is the native residue; doing that at every
position reconstructs the parent. All 171 parents come back at >=98% position coverage, including
all 56 in Coventry's validated tiers. Nothing had to be fetched or guessed.

MONOMERS, NOT COMPLEXES, AND THAT IS DELIBERATE. Folding binder+target would need each target's
sequence and cost far more GPU. The monomer alone answers the question that matters for the model:
is this position buried in the design's own core, or exposed. Combined with the per-position kill
rate we already have, that separates two failure modes the current model cannot tell apart --
a mutation that destroys the fold, and one that leaves the fold intact and breaks the interface.
A buried position with a high kill rate is structural; an EXPOSED position with a high kill rate
is interface, and those are the ones a binding model should care about.

Validated tiers first, so if this is stopped early the useful half is already done.

Usage: cao_fold_parents.py [--limit N] [--all]
Output: datasets/cao2022/folded/<parent>.pdb + logs/cao_fold.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
SSM = ROOT / "data/cao_ssm.csv"
OUT = ROOT / "datasets/cao2022/folded"
LOG = ROOT / "logs/cao_fold.jsonl"
BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
AAS = set("ACDEFGHIKLMNPQRSTVWY")
TIER_RANK = {"quite": 0, "reasonable": 1, "somewhat": 2, "": 3}


def sequences() -> list[tuple[str, str, str, str]]:
    """(tier, target, parent, sequence) — native residue at each position is the unscanned one."""
    rows = list(csv.DictReader(open(SSM)))
    seen, tier = defaultdict(set), {}
    for r in rows:
        if r["mut_aa"] in AAS:
            seen[(r["target"], r["parent"], int(r["pos"]))].add(r["mut_aa"])
        tier[(r["target"], r["parent"])] = r["tier"]
    bypar = defaultdict(dict)
    for (t, p, pos), v in seen.items():
        miss = AAS - v
        if len(miss) == 1:
            bypar[(t, p)][pos] = miss.pop()
    out = []
    for (t, p), d in bypar.items():
        n = max(d)
        if len(d) / n < 0.98:
            continue
        seq = "".join(d.get(i, "A") for i in range(1, n + 1))
        out.append((tier.get((t, p), ""), t, p, seq))
    out.sort(key=lambda r: (TIER_RANK.get(r[0], 3), -len(r[3])))
    return out


def fold(name: str, seq: str, timeout_s: int = 900) -> dict:
    dest = OUT / f"{name}.pdb"
    if dest.exists():
        return {"cached": True, "pdb": str(dest)}
    from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb
    work = Path(tempfile.mkdtemp(prefix="fold_", dir="/tmp/claude-1000"))
    try:
        spec = work / "s.yaml"
        spec.write_text("version: 1\nsequences:\n"
                        f"  - protein:\n      id: A\n      sequence: {seq}\n      msa: empty\n")
        t0 = time.time()
        pr = subprocess.run(
            [str(BOLTZ), "predict", str(spec), "--out_dir", str(work),
             "--recycling_steps", "3", "--diffusion_samples", "1",
             "--output_format", "mmcif", "--override"],
            capture_output=True, text=True, timeout=timeout_s, check=False)
        if pr.returncode != 0:
            return {"error": f"boltz exit {pr.returncode}: {pr.stderr[-300:]}"}
        cifs = sorted(work.rglob("*_model_0.cif"))
        if not cifs:
            return {"error": "no structure"}
        rec: dict = {"seconds": round(time.time() - t0, 1)}
        for j in cifs[0].parent.glob("confidence_*_model_0.json"):
            try:
                c = json.loads(j.read_text())
                rec["plddt"] = c.get("complex_plddt")
            except (OSError, ValueError):
                pass
            break
        _cif_to_pdb(cifs[0], dest)
        rec["pdb"] = str(dest)
        return rec
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout_s}s"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="include the 115 unlisted parents as well as the 56 validated ones")
    a = ap.parse_args()

    seqs = sequences()
    if not a.all:
        seqs = [s for s in seqs if s[0] in ("quite", "reasonable", "somewhat")]
    if a.limit:
        seqs = seqs[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["parent"])
    todo = [s for s in seqs if s[2] not in done]
    print(f"{len(todo)} designs to fold ({len(done)} already done); "
          f"median length {sorted(len(s[3]) for s in todo)[len(todo) // 2] if todo else 0}",
          flush=True)

    t0 = time.time()
    with LOG.open("a") as fh:
        for k, (tier, target, parent, seq) in enumerate(todo, 1):
            r = {"parent": parent, "target": target, "tier": tier, "len": len(seq), "seq": seq}
            r.update(fold(parent, seq))
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"  [{k}/{len(todo)}] {tier or 'unlisted':<11}{parent[:38]:<40}"
                  f"{r.get('seconds', 0):>6.0f}s  plddt {r.get('plddt', float('nan')):.2f}  "
                  f"eta {(len(todo) - k) * el / k / 60:.0f} min", flush=True)
    print("CAO_FOLD_DONE", flush=True)


if __name__ == "__main__":
    main()
