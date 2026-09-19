#!/usr/bin/env python
"""Featurise cropped PPI interfaces the same way we featurise peptide interfaces — and no repack.

TWO THINGS ARE BEING TESTED AT ONCE, deliberately, because they share all their machinery.

1. DOES PPI DATA TRANSFER? The interfaces are the same size as peptide interfaces (survey: 18 vs
   13 residues, 508 vs 568 contacts). If the same descriptors computed on both put them in the
   same feature space, then SKEMPI's 294 cropped complexes -- and the 5,595 mutant measurements
   sitting on them -- become training data for a peptide scorer, and the data starvation that has
   capped every model in this project eases by an order of magnitude.

2. IS COVENTRY RIGHT ABOUT REPACKING? At 9:05: "take the interface section. Again, don't touch it
   after that point, because you'll change it in a way that's probably bad." Our whole peptide
   pipeline repacks before scoring. We already have one piece of evidence he is right -- the
   soft-repulsive pre-pass cost AUC 0.646 -> 0.579 on docked poses -- but that was a pre-pass, not
   the repack itself. Here every interface is scored BOTH ways from identical atoms, so the repack
   is the only variable.

The no-repack arm is also roughly an order of magnitude faster, which matters at 294 complexes and
matters much more at 5,595 mutants.

Usage: ppi_interface_features.py [workers] [--repack]
Output: logs/ppi_interface_features.jsonl  (or _repack.jsonl)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
BENCH = ROOT / "data/ppi_interface_bench.csv"
REPACK = "--repack" in sys.argv
OUT = ROOT / ("logs/ppi_interface_features_repack.jsonl" if REPACK
              else "logs/ppi_interface_features.jsonl")

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["SEL_SOFT_PASS"] = "0"
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF
    import sel_coventry_features as SCF
    SCF._PR = _PR
    SCF._SF = _SF
    SCF.SOFT_PASS = False


def score_raw(receptor: str, peptide: str) -> dict:
    """ref2015 interface energy + our geometry features on the atoms AS GIVEN. No repack."""
    import sel_features as SF

    rec = _PR.pose_from_pdb(receptor)
    pep = _PR.pose_from_pdb(peptide)
    n_rec = rec.total_residue()
    pose = rec.clone()
    pose.append_pose_by_jump(pep, n_rec)
    _SF(pose)

    rec_only = rec.clone(); _SF(rec_only)
    pep_only = pep.clone(); _SF(pep_only)
    out = {"i_total": float(_SF(pose) - _SF(rec_only) - _SF(pep_only))}
    # per-term interface energies, same decomposition the peptide path uses
    tr, tp, tc = SF._terms_of(rec_only), SF._terms_of(pep_only), SF._terms_of(pose)
    for t in SF.TERMS:
        if t in tc:
            out[f"i_{t}"] = float(tc[t] - tr.get(t, 0.0) - tp.get(t, 0.0))
    try:
        out.update({k: float(v) for k, v in SF._geometry(pose, n_rec).items()
                    if isinstance(v, (int, float))})
    except Exception:  # noqa: BLE001 -- geometry is optional, the energy is not
        pass
    return out


def do_one(row: dict) -> dict:
    out = {k: row[k] for k in ("name", "seq", "pep_len", "pocket_res", "dG_kcal_mol",
                               "holdout_id", "n_mutants")}
    t0 = time.time()
    try:
        if REPACK:
            import sel_coventry_features as SCF
            f = SCF.refine_and_feature(row["receptor"], row["peptide_pdb"])
            out["feat"] = {k: float(v) for k, v in f.items()
                           if isinstance(v, (int, float))}
        else:
            out["feat"] = score_raw(row["receptor"], row["peptide_pdb"])
    except Exception as exc:  # noqa: BLE001 -- one bad complex must not stop the set
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    workers = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8)
    rows = list(csv.DictReader(open(BENCH)))
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["name"])
    todo = [r for r in rows if r["name"] not in done]
    print(f"{'repack' if REPACK else 'NO-repack'}: {len(todo)} interfaces, {workers} workers",
          flush=True)
    if not todo:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, r in enumerate(ex.map(do_one, todo, chunksize=1), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if k % 25 == 0 or k == 1:
                el = time.time() - t0
                v = r.get("feat", {}).get("i_total", float("nan"))
                print(f"  [{k}/{len(todo)}] {r['name']:<7} i_total {v:9.1f}  "
                      f"eta {(len(todo) - k) * el / k / 60:.0f} min", flush=True)
    print("PPI_FEATURES_DONE")


if __name__ == "__main__":
    main()
