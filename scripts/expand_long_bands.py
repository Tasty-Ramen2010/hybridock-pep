#!/usr/bin/env python
"""Fill the 13-16 and 17-25 aa bands of the balanced benchmark from an RCSB search.

The local manifests only yield 34 per long band. RCSB search (deposited >= 2020-09-21, X-ray
<= 3.0 A, 2-3 protein entities, one of length 13-30) returns ~1,400 entries with no chain
roles, so roles are discovered from the structure:
  peptide  = chain whose longest unbroken full-backbone stretch is 13-25 aa
  receptor = the >=50-residue protein chain with the most heavy atoms within 5 A of it
Then the SAME build_one() as the manifest build: authors' pocket crop, resolved-stretch target,
phi/psi SS label. Same exclusions (our finetune ids, RecentSet ids, already-benchmarked).

Usage: python scripts/expand_long_bands.py /tmp/claude-1000/rcsb_ids.txt [per_band]
Appends to data/bench_balanced_post2020.csv.
"""
from __future__ import annotations

import csv
import hashlib
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser, NeighborSearch  # noqa: E402

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]  # build_balanced_bench parses sys.argv at import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_balanced_bench as B  # noqa: E402

PER_BAND = int(ARGS[1]) if len(ARGS) > 1 else 60
LONG = ("13-16", "17-25")


def longest_stretch(chain) -> list:
    runs, cur = [], []
    for r in chain:
        if r.id[0] != " ":
            continue
        ok = r.get_resname() in B.AA3 and all(a in r for a in ("N", "CA", "C", "O"))
        if ok and cur and np.linalg.norm(cur[-1]["C"].coord - r["N"].coord) < 2.0:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = [r] if ok else []
    if cur:
        runs.append(cur)
    return max(runs, key=len) if runs else []


def discover(pid: str) -> dict | None:
    """Return a build_one-compatible candidate, or None."""
    path = B.fetch(pid)
    if path is None:
        return None
    try:
        model = next(iter(MMCIFParser(QUIET=True).get_structure(pid, str(path))))
        prot = {ch.id: [r for r in ch if r.id[0] == " " and r.get_resname() in B.AA3] for ch in model}
        peps = [(cid, longest_stretch(model[cid])) for cid in prot]
        peps = [(cid, s) for cid, s in peps if 13 <= len(s) <= 25 and len(prot[cid]) <= 30]
        if not peps:
            return None
        cid, stretch = peps[0]
        pep_atoms = [a for r in stretch for a in r if a.element != "H"]
        recs = [ch for ch in prot if ch != cid and len(prot[ch]) >= 50]
        if not recs:
            return None
        best, best_n = None, 0
        for rc in recs:
            ns = NeighborSearch([a for r in prot[rc] for a in r])
            n = sum(1 for a in pep_atoms if ns.search(a.coord, 5.0))
            if n > best_n:
                best, best_n = rc, n
        if best is None or best_n < 10:
            return None
        rec_seq = "".join(B.AA3[r.get_resname()] for r in prot[best])
        return dict(pdb=pid, pep_chain=cid, rec_chain=best, L=len(stretch), res=0.0,
                    date="rcsb>=2020-09-21", seq="".join(B.AA3[r.get_resname()] for r in stretch),
                    rec_key=hashlib.md5(rec_seq.encode()).hexdigest())
    except Exception:  # noqa: BLE001 -- one bad entry must not stop discovery
        return None


def main() -> None:
    ids = [l.strip().lower() for l in open(ARGS[0]) if l.strip()]
    skip = B.excluded_ids()
    rows = list(csv.DictReader(open(B.CSV_OUT)))
    have_ids = {r["name"].split("_")[0] for r in rows}
    seen_seq = {r["seq"] for r in rows}
    per_rec = Counter(r["rec_key"] for r in rows)
    need = {b: PER_BAND - sum(r["length_bucket"] == b for r in rows) for b in LONG}
    ids = [i for i in ids if i not in skip and i not in have_ids]
    print(f"search entries after exclusions: {len(ids)}; still needed: {need}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        cands = [c for c in ex.map(discover, ids) if c]
    print(f"entries with a 13-25 aa peptide bound to a >=50 aa receptor: {len(cands)}", flush=True)

    added, fails = [], Counter()
    for c in cands:
        b = B.band(c["L"])
        if b not in need or need[b] <= 0 or c["seq"] in seen_seq or per_rec[c["rec_key"]] >= 2:
            continue
        res = B.build_one(c)
        if not res or "fail" in res:
            fails[(res or {}).get("fail", "none")[:30]] += 1
            continue
        if res["length_bucket"] != b or need[res["length_bucket"]] <= 0:
            continue
        res["source"] = "rcsb_post2020"
        added.append(res)
        seen_seq.add(res["seq"])
        per_rec[res["rec_key"]] += 1
        need[res["length_bucket"]] -= 1
        if all(v <= 0 for v in need.values()):
            break

    flds = list(rows[0].keys())
    with open(B.CSV_OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        for r in added:
            w.writerow({k: r.get(k, "") for k in flds})
    rows = list(csv.DictReader(open(B.CSV_OUT)))
    print(f"added {len(added)}; benchmark now {len(rows)} complexes")
    print("by band:", dict(Counter(r["length_bucket"] for r in rows)))
    print("by SS  :", dict(Counter(r["ss_class"] for r in rows)))
    print("long band x SS:", dict(Counter((r["length_bucket"], r["ss_class"]) for r in rows
                                          if r["length_bucket"] in LONG)))
    print("build rejections:", dict(fails))


if __name__ == "__main__":
    main()
