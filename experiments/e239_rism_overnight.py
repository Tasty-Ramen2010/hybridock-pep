"""E239 — overnight PARALLEL 3D-RISM expansion. Generate hydration descriptors for the FULL
PDBbind-925 receptor universe (data/e228_manifest_all.json, 767 receptors) so the E230 baseline ML
has real n to learn from instead of n~49.

Reuses e230.run_one verbatim (pdb4amber -> tleap/ff14SB -> rism3d.snglpnt, cSPCE/KH). Each receptor
runs in its OWN runs/e230_rism/{pdb}/ dir (cwd-isolated, no temp races), so we shard across K worker
processes. Single-writer: workers return rows, the parent appends to the cache as they land (restart-safe,
skips anything already in ANY rism cache: e230_rism.jsonl / e230_t100_rism.jsonl / e230_rism_all.jsonl).

Cheapest receptors first (shortest chain) so the count is maximized by morning.

Run (overnight, in tmux):
  python3 experiments/e239_rism_overnight.py --workers 8 --omp 2
  python3 experiments/e239_rism_overnight.py --eval-only      # merged correlation over every cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e230_rism_pilot as e230  # noqa: E402  (run_one + eval helpers)

MANIFEST = ROOT / "data" / "e228_manifest_all.json"
OUT = ROOT / "data" / "e230_rism_all.jsonl"
# every rism cache is read for de-dup (a receptor done in ANY run is skipped). OUT is set per-run.
ALL_CACHES = [ROOT / "data" / "e230_rism.jsonl", ROOT / "data" / "e230_t100_rism.jsonl",
              ROOT / "data" / "e230_rism_all.jsonl", ROOT / "data" / "e240_ppikb_rism.jsonl"]


def done_set():
    done = set()
    for c in set(ALL_CACHES + [OUT]):
        if c.exists():
            done |= {json.loads(l)["rep_pdb"] for l in c.read_text().splitlines() if l.strip()}
    return done


def _init(omp):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(omp)
    # rebuild e230's subprocess ENV so RISM picks up the per-worker thread count
    e230.ENV = {**os.environ, "AMBERHOME": str(e230.AMBER),
                "PATH": f"{e230.AMBER/'bin'}:{os.environ.get('PATH','')}"}


def _work(rc):
    rep = rc["peptides"][0]
    pdb = rep["pdb"]
    t0 = time.time()
    try:
        d = e230.run_one(pdb, rep["seq"], rep["pep_ch"], smoke=False)
        row = {"rep_pdb": pdb, "n_pep": rc["n_pep"], "y_mean": rc["y_mean"],
               "y_std": rc["y_std"], **d}
        return ("ok", pdb, row, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return ("fail", pdb, str(e)[:160], time.time() - t0)


def eval_only(caches=None):
    rows = []
    for c in (caches or ALL_CACHES):
        if c.exists():
            rows += [json.loads(l) for l in c.read_text().splitlines() if l.strip()]
    seen, uniq = set(), []
    for r in rows:                      # dedupe by receptor, keep first
        if r["rep_pdb"] not in seen:
            seen.add(r["rep_pdb"]); uniq.append(r)
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    feats = ["n_pocket", "n_sites", "max_g", "mean_g", "exchem"]
    y = np.array([r["y_mean"] for r in uniq])
    print(f"=== MERGED 3D-RISM -> receptor baseline (n={len(uniq)}, std={y.std():.2f}) ===")
    for f in feats:
        x = np.array([r.get(f, np.nan) for r in uniq], float)
        ok = ~np.isnan(x)
        if ok.sum() >= 5 and np.nanstd(x[ok]) > 1e-9:
            print(f"  {f:<10} r={np.corrcoef(x[ok], y[ok])[0,1]:+.3f}")
    X = np.array([[r.get(f, np.nan) for f in feats] for r in uniq], float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    pred = np.empty(len(uniq))
    for i in range(len(uniq)):
        tr = np.arange(len(uniq)) != i
        sc = StandardScaler().fit(X[tr])
        pred[i] = Ridge(alpha=2.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[i:i+1]))[0]
    print(f"\n  LOO-Ridge multivariate r = {np.corrcoef(pred, y)[0,1]:+.3f}  (n={len(uniq)})")
    # split: multi-binder (real averaged baseline) vs single-binder
    multi = [i for i, r in enumerate(uniq) if r.get("n_pep", 1) >= 2]
    if 5 <= len(multi) < len(uniq):
        ym = y[multi]; pm = pred[multi]
        print(f"  ... multi-binder subset (n_pep>=2, n={len(multi)}) r = {np.corrcoef(pm, ym)[0,1]:+.3f}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--omp", type=int, default=2)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT), help="output cache (use a distinct file per concurrent run)")
    ap.add_argument("--no-t100-guard", action="store_true",
                    help="skip the t100 work-dir partition (use for a non-PDBbind pdb universe like PPIKB)")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.eval_only:
        return eval_only()

    recs = json.load(open(a.manifest))["receptors"]
    done = done_set()
    # partition away from the still-running t100 driver: it owns runs/e230_rism/{pdb}/ for
    # every pdb in its manifest, so e239 must not touch those (shared work-dir = corruption).
    t100_man = ROOT / "data" / "e228_manifest_t100.json"
    if t100_man.exists() and not a.no_t100_guard:
        done |= {r["peptides"][0]["pdb"] for r in json.load(open(t100_man))["receptors"]}
    todo = [r for r in recs if r["peptides"][0]["pdb"] not in done]
    todo.sort(key=lambda d: d.get("receptor_len", 9999))   # cheapest first
    if a.limit:
        todo = todo[: a.limit]
    print(f"=== E239 overnight RISM: {len(recs)} in manifest, {len(done)} already done, "
          f"{len(todo)} TODO | {a.workers} workers x OMP {a.omp} ===", flush=True)
    if not todo:
        return eval_only()

    n_ok = n_fail = 0
    t_start = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.omp,)) as pool, open(OUT, "a") as fh:
        for k, (status, pdb, payload, dt) in enumerate(
                pool.imap_unordered(_work, todo), 1):
            if status == "ok":
                fh.write(json.dumps(payload) + "\n"); fh.flush()
                n_ok += 1
                tag = (f"n_pocket={payload['n_pocket']:.1f} n_sites={payload['n_sites']} "
                       f"max_g={payload['max_g']:.1f}")
            else:
                n_fail += 1
                tag = f"FAILED: {payload}"
            rate = (time.time() - t_start) / k
            eta = rate * (len(todo) - k) / max(a.workers, 1) / 60
            print(f"  [{k}/{len(todo)}] {pdb} {tag}  ({dt:.0f}s) | ok={n_ok} fail={n_fail} "
                  f"~ETA {eta:.0f}m", flush=True)
    print(f"\n=== DONE: {n_ok} ok, {n_fail} failed in {(time.time()-t_start)/3600:.1f}h ===", flush=True)
    eval_only()


if __name__ == "__main__":
    main()
