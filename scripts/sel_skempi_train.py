#!/usr/bin/env python
"""Does a repacked mutant interface predict ddG better than the structure-free head?

THE COMPARISON HAS TO BE LIKE FOR LIKE.  The shipped head (scoring/mutation_ddg.py) reports
r = 0.385 leave-complex-out over all 4,956 SKEMPI single mutations.  This experiment runs on a
different subset -- 1,013 mutations on 53 complexes whose smaller partner is at most 60 residues,
i.e. the peptide-and-small-domain regime our tool is actually for -- so 0.385 is NOT the number to
beat here.  The shipped head is therefore re-run on THIS exact subset and that is the baseline.

Three models, same folds, same rows:
  Rosetta ddG alone   d_i_total, the plain interface-energy difference between mutant and
                      wild-type. This is what "just use Rosetta" would give you.
  structure-free      the shipped property head: two residue identities + a location class.
  structure-based     the full difference vector -- per-term interface energies, typed contact
                      chemistry, complementarity and burial shape, mutant minus wild-type.

WHY THE DIFFERENCE FORM MATTERS.  Wild-type and mutant are the same structure under the same
protocol, so every complex-level constant -- interface size, the receptor's own stickiness, any
protocol artefact -- cancels in the subtraction. That is the same main-effect cancellation the
selectivity model relies on, and it is why this task is a fairer test of the pair features than
the Coventry grid, where the poses themselves differ.

Leave-COMPLEX-out: all mutations of a complex are held out together. A random split lets
mutations of the same complex sit on both sides and inflates r substantially; published figures in
the 0.40-0.50 band are generally not grouped this way and are not comparable.

Usage: sel_skempi_train.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
FEAT = ROOT / "logs/sel_skempi_features.jsonl"
REPORT = ROOT / "logs/sel_skempi_report.json"


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    def rk(v):
        o = np.argsort(np.argsort(v))
        return o.astype(float)
    return pearson(rk(np.asarray(a)), rk(np.asarray(b)))


def main() -> None:
    rows = []
    for line in FEAT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r and "d_i_total" in r and r.get("ddg") is not None:
                rows.append(r)
    feats = sorted({k for r in rows for k in r if k.startswith("d_")})
    by_pdb = defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)
    print(f"{len(rows)} mutations over {len(by_pdb)} complexes, {len(feats)} difference features")
    print(f"ddG: mean {st.mean([r['ddg'] for r in rows]):+.2f}  "
          f"sd {st.pstdev([r['ddg'] for r in rows]):.2f} kcal/mol  "
          f"on the short chain: {sum(r['on_small'] for r in rows)}/{len(rows)}\n")
    if len(by_pdb) < 5:
        print("too few complexes for leave-complex-out yet")
        return

    y = np.array([r["ddg"] for r in rows])
    groups = np.array([r["pdb"] for r in rows])
    X = np.array([[float(r.get(f, 0.0)) for f in feats] for r in rows])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    results = {}

    # ---- 1. Rosetta ddG alone -----------------------------------------------
    d_tot = X[:, feats.index("d_i_total")]
    r0, s0 = pearson(d_tot, y), spearman(d_tot, y)
    mae0 = float(np.abs(d_tot - y).mean())
    print(f"{'Rosetta ddG alone (d_i_total)':34s} r {r0:+.3f}  rho {s0:+.3f}  "
          f"MAE {mae0:.3f} (uncalibrated units)")
    results["rosetta_ddg_alone"] = {"r": round(r0, 4), "rho": round(s0, 4)}

    # ---- 2. the shipped structure-free head, on THESE rows -------------------
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from hybridock_pep.scoring.mutation_ddg import predict_ddg
        pred = []
        for r in rows:
            try:
                p = predict_ddg(r["wt_aa"], r["mut_aa"], r.get("location", "COR") or "COR")
                pred.append(float(p.ddg if hasattr(p, "ddg") else p))
            except Exception:  # noqa: BLE001
                pred.append(float("nan"))
        pred = np.array(pred)
        ok = ~np.isnan(pred)
        rf, sf = pearson(pred[ok], y[ok]), spearman(pred[ok], y[ok])
        print(f"{'shipped structure-FREE head':34s} r {rf:+.3f}  rho {sf:+.3f}  "
              f"MAE {np.abs(pred[ok] - y[ok]).mean():.3f}  (n={ok.sum()})")
        results["structure_free_head"] = {"r": round(rf, 4), "rho": round(sf, 4),
                                          "n": int(ok.sum())}
    except Exception as exc:  # noqa: BLE001
        print(f"  (shipped head unavailable: {type(exc).__name__}: {exc})")

    # ---- 2b. the structure-free FEATURES, refitted on these folds -------------
    # The shipped head above was trained on all 4,956 SKEMPI single mutations, which INCLUDES
    # these rows, so its correlation here is partly in-sample and is not a fair baseline.
    # Refitting its own feature representation under the identical leave-complex-out folds is.
    try:
        from hybridock_pep.scoring.mutation_ddg import mutation_features
        Xf = np.array([mutation_features(r["wt_aa"], r["mut_aa"],
                                         r.get("location", "COR") or "COR") for r in rows])
        Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
        results["structure_free_refit"] = {}
    except Exception as exc:  # noqa: BLE001
        Xf = None
        print(f"  (structure-free features unavailable: {exc})")

    # ---- 3. structure-based, leave-complex-out -------------------------------
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    def cv(mat, make):
        o = np.zeros(len(y))
        for pdb_ in by_pdb:
            te = groups == pdb_
            tr = ~te
            if tr.sum() < 50:
                continue
            s2 = StandardScaler().fit(mat[tr])
            mm = make()
            mm.fit(s2.transform(mat[tr]), y[tr])
            o[te] = mm.predict(s2.transform(mat[te]))
        return o

    for label, make in (("structure-based RIDGE", lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))),
                        ("structure-based GBT",
                         lambda: GradientBoostingRegressor(random_state=0, n_estimators=300,
                                                           max_depth=3, learning_rate=0.05,
                                                           subsample=0.8))):
        oof = np.zeros(len(y))
        for pdb in by_pdb:
            te = groups == pdb
            tr = ~te
            if tr.sum() < 50:
                continue
            sc = StandardScaler().fit(X[tr])
            m = make()
            m.fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict(sc.transform(X[te]))
        r, s = pearson(oof, y), spearman(oof, y)
        mae = float(np.abs(oof - y).mean())
        print(f"{label:34s} r {r:+.3f}  rho {s:+.3f}  MAE {mae:.3f}")
        results[label] = {"r": round(r, 4), "rho": round(s, 4), "mae": round(mae, 4)}

        # permutation null: shuffle ddG WITHIN each complex, so the null keeps the group
        # structure and only destroys the mutation-to-ddG link
        rng = np.random.default_rng(0)
        nulls = []
        for _ in range(5):
            yp = y.copy()
            for pdb in by_pdb:
                idx = np.flatnonzero(groups == pdb)
                yp[idx] = rng.permutation(yp[idx])
            o = np.zeros(len(y))
            for pdb in by_pdb:
                te = groups == pdb
                tr = ~te
                if tr.sum() < 50:
                    continue
                sc = StandardScaler().fit(X[tr])
                m = make()
                m.fit(sc.transform(X[tr]), yp[tr])
                o[te] = m.predict(sc.transform(X[te]))
            nulls.append(pearson(o, yp))
        print(f"{'':34s} permutation null r {st.mean(nulls):+.3f} +- {st.pstdev(nulls):.3f}")
        results[label]["perm_null"] = round(st.mean(nulls), 4)

    # fair head-to-head: same folds, same model class, different feature sets
    if Xf is not None:
        mk = lambda: GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3,
                                               learning_rate=0.05, subsample=0.8)
        o_free = cv(Xf, mk)
        o_str = cv(X, mk)
        o_both = cv(np.hstack([Xf, X]), mk)
        print("\nSAME FOLDS, SAME MODEL CLASS (GBT), different features:")
        for lab, o in (("structure-FREE features (refit)", o_free),
                       ("structure-BASED features", o_str),
                       ("BOTH", o_both)):
            print(f"   {lab:34s} r {pearson(o, y):+.3f}  rho {spearman(o, y):+.3f}  "
                  f"MAE {np.abs(o - y).mean():.3f}")
            results[lab] = {"r": round(pearson(o, y), 4), "rho": round(spearman(o, y), 4)}

    # ---- by location and by chain -------------------------------------------
    sc = StandardScaler().fit(X)
    oof = np.zeros(len(y))
    for pdb in by_pdb:
        te = groups == pdb
        tr = ~te
        if tr.sum() < 50:
            continue
        s2 = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(s2.transform(X[tr]), y[tr])
        oof[te] = m.predict(s2.transform(X[te]))
    print("\nby interface location (ridge out-of-fold):")
    for loc in ("COR", "SUP", "RIM", "INT", "SUR"):
        m = np.array([r.get("location") == loc for r in rows])
        if m.sum() >= 20:
            print(f"   {loc}  n={m.sum():4d}  r {pearson(oof[m], y[m]):+.3f}  "
                  f"MAE {np.abs(oof[m] - y[m]).mean():.3f}")
    m = np.array([bool(r["on_small"]) for r in rows])
    if m.sum() >= 20 and (~m).sum() >= 20:
        print(f"\n   mutation on the SHORT partner  n={m.sum():4d}  r {pearson(oof[m], y[m]):+.3f}")
        print(f"   mutation on the receptor       n={(~m).sum():4d}  "
              f"r {pearson(oof[~m], y[~m]):+.3f}")
    REPORT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
