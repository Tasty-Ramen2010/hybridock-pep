"""Build the unbiased pooled 65+98 train/test benchmark (Ram's repeated ask).

Combines crystal-65 + the-98 into ONE dataset, then makes a STRATIFIED train/test split so train and test
have MATCHED composition across every confound the campaign identified:
  - dataset source (cr65 vs the98)        -> avoid leave-dataset-out collapse leaking into the split
  - affinity type  (Kd vs Ki)             -> Ki/Kd are not directly comparable
  - net charge class (low |Q|<2 vs charged)-> charged binders are the hard subset
  - ΔG tercile (strong/mid/weak)          -> avoid range imbalance that fakes/breaks correlation
Each stratum is split ~75/25 independently, so the test set is a faithful miniature of the whole.
Writes data/pooled_benchmark_{train,test}.csv (features + y + metadata) and a manifest with the
balance-check table proving train/test are distributionally matched.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROD = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
        "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L", "org_density",
        "cys_frac"]
SEED = 20260612
TEST_FRAC = 0.25


def load():
    g = json.loads(Path("/tmp/e69_geom_all.json").read_text())
    e78 = json.loads(Path("/tmp/e78_dewet.json").read_text())
    # affinity type for cr65 from benchmark_crystal; the98 are all Kd
    bench = {r["pdb"]: r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    rows = []
    for r in g:
        k = ("cr_" + r["pdb"]) if r["ds"] == "cr65" else ("98_" + r["pdb"])
        e = e78.get(k, {})
        nc = e.get("net_charge", 0)
        seq = e.get("seq", "")
        atype = bench.get(r["pdb"], {}).get("affinity_type", "Kd") if r["ds"] == "cr65" else "Kd"
        rr = {f: r.get(f, np.nan) for f in PROD}
        rr.update(id=k, pdb=r["pdb"], dataset=r["ds"], affinity_type=str(atype),
                  net_charge=int(nc), length=len(seq), seq=seq, y=float(r["y"]),
                  net_dewet=e78.get(k, {}).get("net_dewet", np.nan),
                  polar_desolv=e78.get(k, {}).get("polar_desolv", np.nan))
        rows.append(rr)
    return rows


def stratum(r, y_terciles):
    chg = "chg" if abs(r["net_charge"]) >= 2 else "low"
    at = "Ki" if str(r["affinity_type"]).lower().startswith("ki") else "Kd"
    t = "strong" if r["y"] <= y_terciles[0] else ("weak" if r["y"] > y_terciles[1] else "mid")
    return f"{r['dataset']}|{at}|{chg}|{t}"


def main():
    rows = load()
    y = np.array([r["y"] for r in rows])
    yt = (np.percentile(y, 33), np.percentile(y, 67))
    rng = np.random.RandomState(SEED)
    strata: dict = {}
    for r in rows:
        strata.setdefault(stratum(r, yt), []).append(r)
    train, test = [], []
    for s, items in strata.items():
        idx = np.arange(len(items)); rng.shuffle(idx)
        ntest = max(1, round(len(items) * TEST_FRAC)) if len(items) >= 2 else 0
        for j, i in enumerate(idx):
            (test if j < ntest else train).append(items[i])

    cols = ["id", "pdb", "dataset", "affinity_type", "net_charge", "length", "y"] + PROD + \
           ["net_dewet", "polar_desolv", "seq"]
    for name, part in [("train", train), ("test", test)]:
        p = ROOT / f"data/pooled_benchmark_{name}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in part:
                w.writerow(r)
        print(f"  wrote {p.name}: {len(part)} rows")

    # balance check
    def dist(part):
        n = len(part)
        return dict(
            n=n,
            cr65=sum(r["dataset"] == "cr65" for r in part) / n,
            the98=sum(r["dataset"] == "the98" for r in part) / n,
            charged=sum(abs(r["net_charge"]) >= 2 for r in part) / n,
            Ki=sum(str(r["affinity_type"]).lower().startswith("ki") for r in part) / n,
            y_mean=float(np.mean([r["y"] for r in part])),
            y_std=float(np.std([r["y"] for r in part])),
            len_mean=float(np.mean([r["length"] for r in part])),
        )
    dtr, dte = dist(train), dist(test)
    print("\n=== balance check (train vs test should match) ===")
    print(f"{'metric':<10}{'train':>10}{'test':>10}")
    for k in ["n", "cr65", "the98", "charged", "Ki", "y_mean", "y_std", "len_mean"]:
        print(f"  {k:<8}{dtr[k]:>10.3f}{dte[k]:>10.3f}")

    manifest = dict(seed=SEED, test_frac=TEST_FRAC, n_total=len(rows),
                    n_train=len(train), n_test=len(test), features=PROD,
                    strata_keys=sorted(strata), train_balance=dtr, test_balance=dte,
                    note="Stratified by dataset|affinity_type|charge_class|dG_tercile. "
                         "Combined crystal-65 + the-98. Use for pooled calibration + held-out eval.")
    (ROOT / "data/pooled_benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n  wrote data/pooled_benchmark_manifest.json")

    # sanity: fit on train, eval on test (production PROD features)
    from scipy.stats import pearsonr

    def mat(part, cols):
        return (np.array([[r[c] for c in cols] for r in part], float),
                np.array([r["y"] for r in part], float))
    Xtr, ytr = mat(train, PROD); Xte, yte = mat(test, PROD)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + 1.0 * R, A.T @ ytr)
    pred = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ w
    r = pearsonr(pred, yte)[0]; rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    chm = np.array([abs(r_["net_charge"]) >= 2 for r_ in test])
    print(f"\n=== held-out test performance (PROD, train->test) ===")
    print(f"  ALL   r={r:+.3f}  RMSE={rmse:.2f}  (n={len(test)})")
    if chm.sum() >= 4:
        print(f"  charged r={pearsonr(pred[chm], yte[chm])[0]:+.3f} (n={chm.sum()})  "
              f"low-charge r={pearsonr(pred[~chm], yte[~chm])[0]:+.3f} (n={(~chm).sum()})")


if __name__ == "__main__":
    main()
