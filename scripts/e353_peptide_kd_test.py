"""E353 — THE test: does PRISM's charged correction improve ABSOLUTE ΔG-vs-Kd (the r/MAE/RMSE we publish)?

Everything before was ΔΔG (relative). This is absolute affinity on real charged peptides. LRA decomposition:
  ΔG_total = ΔG_scorer(peptide features)  +  Σ_i charged_contribution(residue i)
where charged_contribution = the L5-predicted ISOSTERIC neutralization ΔΔG of each charged peptide residue
(D→N, E→Q, K→M, R→M — removes the charge, keeps the size → the PURE charge term, orthogonal to the scorer's
packing/burial features). We then ask the honest question (Pearson r is invariant to global rescale, so this is
fair despite the scorer already absorbing average charge):

  Model A (baseline "scorer"): exp_ΔG ~ peptide physics features         [GBT, leave-one-out grouped by pocket]
  Model B (+PRISM):            exp_ΔG ~ peptide physics features + charged_sum
  → report r / MAE / RMSE for A and B, OVERALL and on the CHARGED subset (the scorer's known weak spot).

If B beats A on the charged subset, PRISM breaks the charged floor on absolute affinity — the publishable result.

Stage 1 (this file, --extract): compute charged_sum per complex → data/e353_charged_sum.jsonl (slow, structures).
Stage 2 (--eval): train A vs B, report metrics.

Run: OMP_NUM_THREADS=2 python scripts/e353_peptide_kd_test.py --extract
     OMP_NUM_THREADS=1 python scripts/e353_peptide_kd_test.py --eval
"""
from __future__ import annotations
import sys, json, argparse, time
import numpy as np
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from e334_skempi_validation import fetch
import e351_l5_features as l5

PEP = "/home/igem/unknown_software/data/pdbbind_peptides.jsonl"
SUM_OUT = "/home/igem/unknown_software/data/e353_charged_sum.jsonl"
L5_FEATS = "/home/igem/unknown_software/data/e351_l5_features.jsonl"
NEUTRALIZE = {"D": "N", "E": "Q", "K": "M", "R": "M"}     # isosteric charge removal
SCORER_FEATS = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
                "arom_cc", "hb_count", "mj_contact", "strength_bur", "rg_per_L", "org_density", "cys_frac",
                "mean_burial"]
_P = PDBParser(QUIET=True)


def train_l5():
    """Train the L5 charged-ΔΔG model on the SKEMPI feature set (same as e352)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = [json.loads(x) for x in open(L5_FEATS)]
    X = np.array([[r[k] for k in _L5KEYS] for r in rows], float)
    y = np.array([r["exp"] for r in rows], float)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                      l2_regularization=1.0, min_samples_leaf=20, random_state=0)
    m.fit(X, y)
    return m


_L5KEYS = ["wt_charge", "mut_charge", "dq", "d_volume", "d_hydropathy", "is_alanine", "is_isosteric",
           "buried_frac", "n_contacts", "opp_charge_dist", "same_charge_dist", "n_aromatic",
           "n_polar_neutral", "n_hydrophobic", "metal_near", "complex_atoms"]


def find_chains(pdb, seq):
    """Return (peptide_chain_id, receptor_chain_ids) by matching the peptide sequence to a chain."""
    st = _P.get_structure(pdb, fetch(pdb))[0]
    seq = seq.upper(); pep = None
    for ch in st:
        res = [r for r in ch if r.id[0] == " "]
        try:
            cs = "".join(seq1(r.get_resname()) for r in res)
        except Exception:
            continue
        if cs and (seq in cs or cs in seq or _sim(cs, seq) > 0.7):
            pep = ch.id
    if pep is None:
        return None
    rec = "".join(sorted(ch.id for ch in st if ch.id != pep and any(r.id[0] == " " for r in ch)))
    return pep, rec


def _sim(a, b):
    n = min(len(a), len(b))
    return (sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))) if n else 0.0


def charged_sum(pdb, seq, model):
    """Σ L5-predicted isosteric neutralization ΔΔG over the peptide's charged residues (the PRISM charge term)."""
    ch = find_chains(pdb, seq)
    if ch is None:
        raise RuntimeError("no peptide chain")
    pep, rec = ch
    if not rec:
        raise RuntimeError("no receptor chain")
    st = l5.get_struct(pdb)                      # cached, SASA computed
    # map peptide chain residues (author numbering) to the peptide seq positions we know are charged
    pep_res = [r for r in st[pep] if r.id[0] == " "]
    total, n = 0.0, 0
    for r in pep_res:
        try:
            aa = seq1(r.get_resname())
        except Exception:
            continue
        if aa not in NEUTRALIZE:
            continue
        mut = f"{aa}{pep}{r.id[1]}{NEUTRALIZE[aa]}"
        tag = f"{pdb}_{pep}_{rec}"
        f = l5.features(tag, mut, 0.0)
        if f is None:
            continue
        x = np.array([[f[k] for k in _L5KEYS]], float)
        total += float(model.predict(x)[0]); n += 1
    return total, n


def extract():
    model = train_l5()
    rows = [json.loads(x) for x in open(PEP)]
    done = set()
    try:
        for x in open(SUM_OUT):
            done.add(json.loads(x)["pdb"])
    except FileNotFoundError:
        pass
    print(f"=== E353 extract charged_sum: {len(rows)} complexes, {len(done)} done ===", flush=True)
    t0 = time.time(); k = 0
    with open(SUM_OUT, "a") as fh:
        for r in rows:
            if r["pdb"] in done:
                continue
            rec = {"pdb": r["pdb"], "y": float(r["y"]), "seq": r["seq"]}
            try:
                cs, n = charged_sum(r["pdb"], r["seq"], model)
                rec["charged_sum"] = cs; rec["n_charged"] = n
            except Exception as e:
                rec["charged_sum"] = None; rec["err"] = f"{type(e).__name__}:{str(e)[:40]}"
            fh.write(json.dumps(rec) + "\n"); fh.flush(); k += 1
            if k % 50 == 0:
                print(f"  {k} done ({(time.time()-t0)/60:.0f}min, {len(l5._struct)} PDBs cached)", flush=True)
    print(f"wrote {SUM_OUT} (+{k}, {(time.time()-t0)/60:.0f}min)")


def _metrics(pred, y):
    from scipy.stats import pearsonr
    return pearsonr(pred, y)[0], np.mean(np.abs(pred - y)), np.sqrt(np.mean((pred - y) ** 2))


def evaluate():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    pep = {json.loads(x)["pdb"]: json.loads(x) for x in open(PEP)}
    cs = {json.loads(x)["pdb"]: json.loads(x) for x in open(SUM_OUT)}
    data = []
    for pdb, c in cs.items():
        if c.get("charged_sum") is None or pdb not in pep:
            continue
        p = pep[pdb]
        feats = [float(p[f]) for f in SCORER_FEATS]
        data.append((pdb, feats, c["charged_sum"], c.get("n_charged", 0), float(p["y"]), p["seq"]))
    pdbs = [d[0] for d in data]
    Xs = np.array([d[1] for d in data]); cvec = np.array([d[2] for d in data]).reshape(-1, 1)
    nch = np.array([d[3] for d in data]); y = np.array([d[4] for d in data])
    groups = np.array([hash(d[5][:4]) % 100000 for d in data])   # loose grouping to reduce trivial leakage
    print(f"=== E353 eval: n={len(y)} charged-peptide complexes ===")

    def loo(X):
        oof = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=8).split(X, y, groups):
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=15, l2_regularization=1.0, random_state=0)
            m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        return oof

    A = loo(Xs)                          # baseline scorer
    B = loo(np.hstack([Xs, cvec]))       # + PRISM charged_sum
    for name, pred in (("A baseline", A), ("B +PRISM ", B)):
        r, mae, rmse = _metrics(pred, y)
        print(f"{name}  ALL   n={len(y)}  r={r:+.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    # charged subset = peptides with >=2 charged residues (where the scorer should struggle most)
    mask = nch >= 2
    for name, pred in (("A baseline", A), ("B +PRISM ", B)):
        r, mae, rmse = _metrics(pred[mask], y[mask])
        print(f"{name}  CHARGED(>=2) n={mask.sum():3d}  r={r:+.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    print("\nVERDICT: if B beats A on the CHARGED subset, PRISM's charge term breaks the charged floor on absolute "
          "Kd — the publishable r/MAE/RMSE win.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true"); ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.eval:
        evaluate()


if __name__ == "__main__":
    main()
