#!/usr/bin/env python
"""Build the training set for OUR OWN direction model — the one bit co-folding gives us.

WHAT WE ACTUALLY IMPORT FROM A CO-FOLDING MODEL.  Not its coordinates. On 9CCE the co-folded axis
identified the forward-threading poses in our own pool with 98% precision, and that single bit --
which way round the peptide runs through the groove -- is the thing we provably cannot get on our
own today: 79-81% of our poses thread backwards, more sampling does not help (N=24 -> 500 moves
best RMSD 6.74 -> 5.66 A), ref2015 prefers the forward pose only 41% of the time, and the best of
five geometric descriptors reaches AUC 0.603.

So the way to make this OURS rather than Boltz's is to learn that one bit from our own data. The
training pairs are free, and they are the hardest possible negatives: thread a peptide onto its
own crystal backbone FORWARDS (label 1) and REVERSED (label 0). The two poses have identical
composition, identical length, the same receptor, the same backbone and near-identical burial.
Everything cancels except the direction -- which is exactly the quantity to be learned, and
exactly why a chemistry-blind descriptor cannot do it.

REVERSING A SEQUENCE IS NOT THE SAME AS REVERSING A POSE, and the difference is the point. We keep
the backbone fixed and write the sequence onto it backwards, so residue k sits where residue
(n+1-k) used to. That is precisely what our sampler does when it threads a groove the wrong way:
the chain occupies the same density, with the side chains in the wrong sockets.

Usage: sel_direction_data.py [workers] [--blocks 40]
Output: logs/sel_direction.jsonl  (two rows per complex: forward and reversed)
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
BLOCKS = ROOT / "data/sel_blocks.json"
OUT = ROOT / "logs/sel_direction.jsonl"

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF


def do_one(args: tuple) -> list[dict]:
    """Thread the sequence forwards and backwards onto the same backbone; score both."""
    import sel_features as SF
    pdb, receptor, backbone, seq = args
    out = []
    for direction, s in (("forward", seq), ("reversed", seq[::-1])):
        row = {"pdb": pdb, "direction": direction, "seq": s, "len": len(s),
               "label": int(direction == "forward")}
        t0 = time.time()
        try:
            r = SF.do_pair((0, 0, 0, s, receptor, backbone, backbone, pdb, pdb))
            if "error" in r:
                row["error"] = r["error"]
            else:
                row.update({k: v for k, v in r.items()
                            if isinstance(v, (int, float)) and k not in
                            ("block", "pep_idx", "rec_idx", "cognate", "seconds")})
            row["seconds"] = round(time.time() - t0, 1)
        except Exception as exc:  # noqa: BLE001 -- one bad complex must not stop the set
            row["error"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[sys.argv.index("--blocks") + 1]) if "--blocks" in sys.argv else 40

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["pdb"])

    jobs, seen = [], set()
    for b in json.loads(BLOCKS.read_text())[:nb]:
        for m in b["members"]:
            if m["pdb"] in seen or m["pdb"] in done:
                continue
            seen.add(m["pdb"])
            # a palindrome has no direction to learn, and a very short peptide barely has one
            if len(m["seq"]) >= 8 and m["seq"] != m["seq"][::-1]:
                jobs.append((m["pdb"], m["receptor"], m["peptide_pdb"], m["seq"]))
    print(f"{len(jobs)} complexes x 2 directions = {2 * len(jobs)} poses "
          f"({len(done)} done), {workers} workers", flush=True)
    if not jobs:
        return

    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        n = 0
        for rows in ex.map(do_one, jobs, chunksize=2):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
            if n % 40 == 0:
                rate = n / max(time.time() - t0, 1)
                print(f"  {n}/{2 * len(jobs)}  {rows[0]['pdb']}  "
                      f"eta {(2 * len(jobs) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    print("SEL_DIRECTION_DONE")


if __name__ == "__main__":
    main()
