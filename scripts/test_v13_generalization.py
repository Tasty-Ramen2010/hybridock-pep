"""Test v1.3 per-family calibration's generalization to unseen complexes.

The +0.731 Pearson r reported in calibration_notes.md is LOO within the 146
complexes that landed in the 8 big clusters. The honest test: route a *new*
receptor to its nearest big family via k-mer Jaccard, apply that family's
ridge, compare to experiment.

Three test sets:
  1. 94 "fallback" complexes — in the 240-complex audit but not in any big
     cluster (didn't have ≥10 neighbors). Each will be routed to its nearest
     big family.
  2. PEPBI 45 scored entries (truly out-of-distribution).
  3. Wang 27 entries (truly out-of-distribution).

Reports Pearson r + RMSE per test set, and a stratification by which big
family the dispatcher assigned each test point to.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw_pdbs"
K = 6  # k-mer size, matches cluster_and_fit_per_family.py

AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q",
          "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
          "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def chain_seq(lines):
    seen, last = [], None
    for l in lines:
        try: key = (int(l[22:26].strip()), l[26])
        except ValueError: continue
        if key == last: continue
        last = key
        seen.append(AA3to1.get(l[17:20].strip(),'X'))
    return ''.join(seen)


def receptor_sequence(pdb_path: Path, peptide_seq: str, receptor_chain_hint: str | None) -> str | None:
    if not pdb_path.exists():
        return None
    chains: dict[str, list[str]] = defaultdict(list)
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM"):
            chains[line[21]].append(line)
    if receptor_chain_hint and receptor_chain_hint.strip() in chains:
        return chain_seq(chains[receptor_chain_hint.strip()])
    # Fallback: longest chain that doesn't look like the peptide
    pep = peptide_seq.upper() if peptide_seq else ""
    cand = sorted(((c, chain_seq(ls)) for c, ls in chains.items()),
                  key=lambda x: -len(x[1]))
    for c, s in cand:
        if pep and pep in s and len(s) <= len(pep) + 5:
            continue  # this is the peptide
        return s
    return cand[0][1] if cand else None


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def dispatch_family(receptor_seq: str, family_members_kmers: dict[str, list[set]]) -> tuple[str, float]:
    """Route to nearest family by MAX Jaccard to any member of that family.

    Returns (family_id, similarity).
    """
    q = {receptor_seq[i:i+K] for i in range(len(receptor_seq) - K + 1)}
    best_fam, best_sim = None, -1.0
    for fam, member_kmers in family_members_kmers.items():
        sim = max((jaccard(q, m) for m in member_kmers), default=0.0)
        if sim > best_sim:
            best_sim = sim
            best_fam = fam
    return best_fam, best_sim


def predict_with_v13(family_fit: dict, vina: float, n_contact: int, s_ss: float) -> float:
    return (family_fit["w_vina"] * vina
            + family_fit["w_contact"] * n_contact
            + family_fit["w_s_ss_weighted"] * s_ss
            + family_fit["intercept"])


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y):
    if not x:
        return float("nan")
    a = np.array(x); b = np.array(y)
    return float(math.sqrt(((a - b) ** 2).mean()))


def main() -> None:
    # Load v1.3
    pf = json.loads((ROOT / "data" / "calibration_per_family.json").read_text())
    families = pf["families"]
    fallback = pf["fallback"]

    # Pre-compute k-mer sets for each big family's member PDBs
    family_members_kmers: dict[str, list[set]] = {}
    family_pdbs: dict[str, set[str]] = {}
    eval_data = {r["pdb"]: r for r in json.loads(
        (ROOT / "data" / "eval_holdout_calibrations.json").read_text())}
    with (ROOT / "data" / "training_complexes_full.csv").open() as f:
        full_meta = {r["pdb_id"].lower(): r for r in csv.DictReader(f) if r.get("experimental_pkd")}
    for fam_id, fit in families.items():
        member_kmers = []
        family_pdbs[fam_id] = set(p.lower() for p in fit["pdbs"])
        for pdb in fit["pdbs"]:
            r = full_meta.get(pdb.lower())
            if not r:
                continue
            seq = receptor_sequence(
                RAW / f"{pdb.upper()}.pdb",
                r["peptide_sequence"],
                r.get("receptor_chain"),
            )
            if seq:
                member_kmers.append({seq[i:i+K] for i in range(len(seq)-K+1)})
        family_members_kmers[fam_id] = member_kmers

    print("=" * 76)
    print("v1.3 per-family generalization test")
    print("=" * 76)

    # === Test 1: 94 fallback complexes from same 240-complex audit ===
    print("\n--- Test 1: 94 fallback complexes (audit-set, not in any big family) ---")
    all_in_big = set()
    for fam in family_pdbs.values():
        all_in_big.update(fam)
    fallback_pdbs = [p for p in eval_data if p.lower() not in all_in_big]
    print(f"  candidates: {len(fallback_pdbs)}")

    dispatched = []
    family_counts = Counter()
    for pdb in fallback_pdbs:
        r = full_meta.get(pdb.lower())
        if not r:
            continue
        rec_seq = receptor_sequence(
            RAW / f"{pdb.upper()}.pdb",
            r["peptide_sequence"],
            r.get("receptor_chain"),
        )
        if not rec_seq or len(rec_seq) < K:
            continue
        fam, sim = dispatch_family(rec_seq, family_members_kmers)
        if fam is None:
            continue
        family_counts[fam] += 1
        eval_row = eval_data[pdb]
        pred = predict_with_v13(
            families[fam], eval_row["vina"], eval_row["n_contact"], eval_row["s_ss_weighted"],
        )
        dispatched.append({
            "pdb": pdb,
            "family": fam,
            "similarity": round(sim, 3),
            "dg_exp": eval_row["dg_exp"],
            "dg_pred": pred,
            "vina": eval_row["vina"],
        })

    if dispatched:
        dg_e = [d["dg_exp"] for d in dispatched]
        dg_p = [d["dg_pred"] for d in dispatched]
        r = pearson(dg_e, dg_p)
        rms = rmse(dg_e, dg_p)
        print(f"  Routed: {len(dispatched)} of {len(fallback_pdbs)}")
        print(f"  Family assignment distribution: {dict(family_counts.most_common())}")
        print(f"  Pearson r = {r:+.3f}")
        print(f"  RMSE      = {rms:.2f} kcal/mol")
        sims = [d["similarity"] for d in dispatched]
        print(f"  Median dispatcher similarity: {np.median(sims):.3f}")
        print(f"  Fraction with sim ≥ 0.10: {sum(1 for s in sims if s>=0.10)}/{len(sims)}")
        # Per-similarity-bucket
        hi = [d for d in dispatched if d["similarity"] >= 0.10]
        lo = [d for d in dispatched if d["similarity"] < 0.10]
        if hi:
            r_hi = pearson([d["dg_exp"] for d in hi], [d["dg_pred"] for d in hi])
            print(f"  High-similarity subset (sim ≥ 0.10, n={len(hi)}): r={r_hi:+.3f}")
        if lo:
            r_lo = pearson([d["dg_exp"] for d in lo], [d["dg_pred"] for d in lo])
            print(f"  Low-similarity subset  (sim < 0.10, n={len(lo)}): r={r_lo:+.3f}")
    else:
        print("  No dispatched complexes — investigate")

    # Save dispatch results
    out = ROOT / "data" / "v13_generalization_results.json"
    out.write_text(json.dumps(dispatched, indent=2))
    print(f"\nWrote {out}")

    # === Vs. naive baselines ===
    if dispatched:
        # Baseline 1: use fallback ridge for everyone
        dg_fb = [fallback["w_vina"] * d["vina"]
                 + fallback["w_contact"] * eval_data[d["pdb"]]["n_contact"]
                 + fallback["w_s_ss_weighted"] * eval_data[d["pdb"]]["s_ss_weighted"]
                 + fallback["intercept"] for d in dispatched]
        r_fb = pearson(dg_e, dg_fb)
        rms_fb = rmse(dg_e, dg_fb)
        # Baseline 2: Vina only
        v_only = [d["vina"] for d in dispatched]
        r_v = pearson(dg_e, v_only)

        print("\n--- Comparison on same fallback set ---")
        print(f"  v1.3 dispatch  : r={r:+.3f}  RMSE={rms:.2f}")
        print(f"  fallback ridge : r={r_fb:+.3f}  RMSE={rms_fb:.2f}")
        print(f"  Vina only      : r={r_v:+.3f}")


if __name__ == "__main__":
    main()
