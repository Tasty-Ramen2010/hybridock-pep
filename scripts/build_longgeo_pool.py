#!/usr/bin/env python
"""Build the long/geometry-emphasised training pool for the full-corpus finetune.

WHY THIS SET EXISTS.  The raw on-disk corpus (data/longft_train_full16k.csv, 18,135
complexes) is 87% peptides of 12 residues or fewer.  Training on it as-is would drown the
long-peptide signal our epoch-10 checkpoint was deliberately finetuned for.  But the corpus
is still worth using, because what it actually brings is RECEPTORS: 8,768 distinct PDB ids
against the 3,705 complexes we train on today.  So the pool is built to keep the receptor
diversity and throw away the redundancy, then re-weight length back toward long/very long.

THREE KNOBS, ALL DECLARED HERE RATHER THAN IN THE TRAINER.
  1. length repetition.  13-16 x3, 17-20 x5, 21-25 x5, short x1.  The trainer already
     expands rows by repetition (`weighted_train_indices.extend([i] * w)`), so encoding the
     weights as duplicate CSV rows is exactly equivalent and needs no code change -- and it
     composes with --no-source-weights, which we must keep on so the orthogonal per-source
     weights do not silently re-skew the mix.
  2. short-bucket dedupe by receptor.  In 05-08 and 09-12 we keep at most KEEP_PER_PDB rows
     per PDB id, choosing the ones with the most enclosed interface.  Many short entries are
     the same receptor with a different chain or a different MHC epitope; those teach one
     geometry many times over.  One or two per receptor keeps the geometry and drops the
     repetition.
  3. interface-enclosure bonus.  Every row gets contacts_per_res = receptor heavy atoms
     within CONTACT_CUT of the peptide, per peptide residue.  A peptide lying on a flat
     surface patch scores low; a peptide threading an enclosed groove scores high.  Rows in
     the top quartile get one extra copy.  This is the geometry class we measurably fail on
     (79-81% of our poses thread the groove backwards), and it is a receptor property, so it
     up-weights groove geometry without needing a curated solenoid list.

The 155 benchmark PDB ids were already excluded upstream when full16k was built; the 574
complexes unique to our current long-biased set (propedia_long, refpepdb, recent_2024_2026)
are unioned back in so this run is a superset of what epoch-10 already saw.

Usage: build_longgeo_pool.py [workers]
Writes: data/longft_train_longgeo.csv  (+ a .stats.json beside it)
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
FULL = ROOT / "data/longft_train_full16k.csv"
CUR = ROOT / "data/longft_train_scope25_authorsP.csv"
OUT = ROOT / "data/longft_train_longgeo.csv"

LEN_WEIGHT = {"05-08": 1, "09-12": 1, "13-16": 4, "17-20": 8, "21-25": 8}
KEEP_PER_PDB = {"05-08": 1, "09-12": 1}     # long buckets: keep everything
CONTACT_CUT = 4.5                            # A, heavy-atom
GEO_BONUS_Q = 0.75                           # top quartile of contacts_per_res gets +1 copy
# Short peptides are in this pool for ONE reason -- the receptors they come with -- so a short
# entry has to earn its slot by being an enclosed interface.  Anything below the pool median
# enclosure is a peptide lying on a flat surface patch: it teaches neither long-chain placement
# nor groove geometry, so it is dropped rather than diluting the epoch.  Long entries are kept
# unconditionally; they are the signal epoch-10 was finetuned for and we are not trading it away.
SHORT_MIN_ENCLOSURE_Q = 0.50
CACHE = ROOT / "data/.enclosure_cache_full.json"


def bucket(n: int) -> str:
    return ("05-08" if n <= 8 else "09-12" if n <= 12 else
            "13-16" if n <= 16 else "17-20" if n <= 20 else "21-25")


def pdb_id(cn: str) -> str:
    m = re.search(r"_([0-9][A-Za-z0-9]{3})(?:_|$)", cn)
    return m.group(1).upper() if m else cn


def _heavy(path: str):
    import numpy as np
    xs = []
    with open(path) as fh:
        for l in fh:
            if l.startswith(("ATOM", "HETATM")) and l[76:78].strip() != "H":
                xs.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return np.asarray(xs, dtype="f4")


def contacts(args):
    """Receptor heavy atoms within CONTACT_CUT of the peptide, normalised per residue."""
    idx, rec_p, pep_p, n = args
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        R, P = _heavy(rec_p), _heavy(pep_p)
        if len(R) == 0 or len(P) == 0:
            return idx, None
        hit = cKDTree(R).query_ball_point(P, CONTACT_CUT)
        return idx, len({j for h in hit for j in h}) / max(n, 1)
    except Exception:  # noqa: BLE001 -- one unreadable file must not kill the build
        return idx, None


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seen, pool = set(), []
    for src in (FULL, CUR):
        for r in csv.DictReader(open(src)):
            if r["complex_name"] in seen:
                continue
            seen.add(r["complex_name"])
            pool.append(r)
    print(f"union pool: {len(pool)} unique complexes "
          f"({len({pdb_id(r['complex_name']) for r in pool})} PDB ids)", flush=True)

    jobs = [(i, r["protein_description"], r["peptide_description"], int(r["pep_len"]))
            for i, r in enumerate(pool)]
    missing = [j for j in jobs if not (os.path.exists(j[1]) and os.path.exists(j[2]))]
    if missing:
        print(f"WARNING: {len(missing)} rows have missing files; dropping them")
        bad = {j[0] for j in missing}
        pool = [r for i, r in enumerate(pool) if i not in bad]
        jobs = [(i, r["protein_description"], r["peptide_description"], int(r["pep_len"]))
                for i, r in enumerate(pool)]

    cpr = [None] * len(pool)
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [j for j in jobs if pool[j[0]]["complex_name"] not in cached]
    for i, r in enumerate(pool):
        if r["complex_name"] in cached:
            cpr[i] = cached[r["complex_name"]]
    print(f"interface enclosure: {len(cached)} cached, computing {len(todo)} "
          f"with {workers} workers...", flush=True)
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for k, (i, v) in enumerate(ex.map(contacts, todo, chunksize=64), 1):
                cpr[i] = v
                if k % 4000 == 0:
                    print(f"  {k}/{len(todo)}", flush=True)
        CACHE.write_text(json.dumps({r["complex_name"]: cpr[i]
                                     for i, r in enumerate(pool) if cpr[i] is not None}))
    ok = [v for v in cpr if v is not None]
    print(f"  enclosure computed for {len(ok)}/{len(pool)}; "
          f"median {st.median(ok):.1f} receptor atoms per peptide residue", flush=True)
    qs = sorted(ok)
    thr = qs[int(GEO_BONUS_Q * (len(qs) - 1))]
    floor = qs[int(SHORT_MIN_ENCLOSURE_Q * (len(qs) - 1))]
    print(f"  geometry bonus threshold (q{GEO_BONUS_Q:.2f}) = {thr:.1f}; "
          f"short-bucket enclosure floor (q{SHORT_MIN_ENCLOSURE_Q:.2f}) = {floor:.1f}",
          flush=True)

    # --- short buckets: one row per receptor, flat interfaces dropped ----------
    by_pdb = defaultdict(list)
    kept, dropped, flat = [], Counter(), Counter()
    for i, r in enumerate(pool):
        b = bucket(int(r["pep_len"]))
        if b not in KEEP_PER_PDB:
            kept.append(i)                       # long: unconditional
        elif cpr[i] is not None and cpr[i] >= floor:
            by_pdb[(b, pdb_id(r["complex_name"]))].append(i)
        else:
            flat[b] += 1
    for (b, _), idxs in by_pdb.items():
        idxs.sort(key=lambda i: -cpr[i])
        kept.extend(idxs[:KEEP_PER_PDB[b]])
        dropped[b] += max(0, len(idxs) - KEEP_PER_PDB[b])
    kept.sort()
    print(f"short buckets: dropped {sum(flat.values())} flat-interface rows {dict(flat)}, "
          f"then {sum(dropped.values())} same-receptor duplicates {dict(dropped)}", flush=True)

    # --- expand by weight ------------------------------------------------------
    cols = list(pool[0].keys()) + ["contacts_per_res", "len_w", "geo_w"]
    eff = Counter()
    uniq = Counter()
    rows_out = []
    for i in kept:
        r = pool[i]
        b = bucket(int(r["pep_len"]))
        w = LEN_WEIGHT[b]
        g = 1 if (cpr[i] is not None and cpr[i] >= thr) else 0
        total = w + g
        o = dict(r)
        o["contacts_per_res"] = f"{cpr[i]:.2f}" if cpr[i] is not None else ""
        o["len_w"], o["geo_w"] = w, g
        rows_out.extend([o] * total)
        eff[b] += total
        uniq[b] += 1

    with OUT.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows_out)

    tot = sum(eff.values())
    print(f"\nwrote {OUT}")
    print(f"{'bucket':>8} {'unique':>8} {'effective':>10} {'share':>7}")
    for b in ("05-08", "09-12", "13-16", "17-20", "21-25"):
        print(f"{b:>8} {uniq[b]:>8d} {eff[b]:>10d} {100 * eff[b] / tot:>6.1f}%")
    longshare = 100 * sum(eff[b] for b in ("13-16", "17-20", "21-25")) / tot
    print(f"{'TOTAL':>8} {sum(uniq.values()):>8d} {tot:>10d}")
    print(f"peptides >=13 aa: {longshare:.1f}% of the epoch "
          f"(full16k raw: 13.1%, our current set: 76.8%)")
    print(f"distinct receptors: {len({pdb_id(rows_out[0]['complex_name']) for _ in [0]}) and ''}"
          f"{len({pdb_id(r['complex_name']) for r in rows_out})} PDB ids "
          f"(current set: 3,705 complexes)")
    (OUT.with_suffix(".stats.json")).write_text(json.dumps({
        "unique": dict(uniq), "effective": dict(eff), "rows": len(rows_out),
        "long_share_pct": round(longshare, 1), "len_weight": LEN_WEIGHT,
        "keep_per_pdb": KEEP_PER_PDB, "geo_threshold": round(thr, 2),
        "contact_cut": CONTACT_CUT}, indent=2))


if __name__ == "__main__":
    main()
