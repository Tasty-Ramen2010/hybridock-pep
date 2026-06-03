"""Evaluate v1.0 (legacy), v1.1 (ridge), v1.2 (entropy) calibrations on the
held-out crystal-pose set.

Pulls from data/training_scores_full.json (272 crystal-pose Vina/AD4 scores)
and pairs against training_complexes_full.csv for experimental pKd. Peptide
chains and receptor coords are extracted from datasets/raw_pdbs/{PDB}.pdb by
matching the sequence column to whichever chain produces the matching
single-letter sequence.

Writes per-complex predictions to data/eval_holdout_calibrations.json and
prints summary tables on ALL, HELD-OUT (excluding the 6 PepSet-6 training
overlap), and Kd-only subsets.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from hybridock_pep.scoring.per_residue_entropy import compute_entropy_sums

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw_pdbs"
SCORES = ROOT / "data" / "training_scores_full.json"
CSV_PATH = ROOT / "data" / "training_complexes_full.csv"
TRAIN_OVERLAP = {"1a0n", "1ddv", "1l2z", "1nrl", "1ywi", "2hwn"}

CAL_LEGACY = json.loads((ROOT / "data" / "calibration.json").read_text())
CAL_RIDGE = json.loads((ROOT / "data" / "calibration_v1_1_production_ridge.json").read_text())
CAL_ENT = json.loads((ROOT / "data" / "calibration_v1_2_production_entropy.json").read_text())

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def kd_to_dg(pkd: float) -> float:
    return -1.3633 * pkd  # 298 K


def predict_legacy(vina, ad4, nc, cal):
    a = cal.get("ensemble_ad4_weight", 0.3)
    blend = (1 - a) * vina + a * (ad4 if ad4 is not None else vina)
    return blend + cal["alpha"] * nc


def predict_ridge(vina, ad4, nc, cal):
    ad4 = ad4 if ad4 is not None else 0.0
    return cal["w_vina"] * vina + cal["w_ad4"] * ad4 + cal["w_contact"] * nc + cal["intercept"]


def predict_entropy(vina, s_ss, cal):
    return cal["w_vina"] * vina + cal["w_s_ss_weighted"] * s_ss + cal["intercept"]


def split_chains(pdb_path: Path):
    """Return dict chain_id → list of ATOM lines (peptide & receptor candidates)."""
    chains: dict[str, list[str]] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        cid = line[21]
        chains.setdefault(cid, []).append(line)
    return chains


def chain_sequence(lines: list[str]) -> str:
    seen = []
    last = None
    for line in lines:
        try:
            res_seq = int(line[22:26].strip())
            i_code = line[26]
        except ValueError:
            continue
        key = (res_seq, i_code)
        if key == last:
            continue
        last = key
        resn = line[17:20].strip()
        seen.append(AA3to1.get(resn, "X"))
    return "".join(seen)


def find_peptide_chain(chains: dict[str, list[str]], target_seq: str) -> str | None:
    """Find chain whose sequence contains target_seq as a contiguous substring."""
    ts = target_seq.upper()
    best = None
    for cid, lines in chains.items():
        seq = chain_sequence(lines)
        if ts in seq:
            # Prefer the shortest chain that contains it (likely the peptide).
            if best is None or len(seq) < len(chain_sequence(chains[best])):
                best = cid
    return best


def heavy_coords(lines: list[str]) -> np.ndarray:
    out = []
    for line in lines:
        atom = line[12:16].strip()
        if atom.startswith("H") or atom in {"H"}:
            continue
        try:
            out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.array(out) if out else np.zeros((0, 3))


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\nEND\n")


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(np.array(x), np.array(y))[0, 1])


def rmse(x, y):
    if not x:
        return float("nan")
    return float(math.sqrt(((np.array(x) - np.array(y)) ** 2).mean()))


def main() -> None:
    import tempfile

    scores = json.loads(SCORES.read_text())
    with CSV_PATH.open() as f:
        meta = {r["pdb_id"].lower(): r for r in csv.DictReader(f) if r.get("experimental_pkd")}

    tmp = Path(tempfile.mkdtemp(prefix="ent_eval_"))
    results = []
    skipped = {"no_meta": 0, "no_pdb": 0, "no_chain": 0, "ent_err": 0}

    for pdb, sc in scores.items():
        pdb_lc = pdb.lower()
        if pdb_lc not in meta:
            skipped["no_meta"] += 1
            continue
        raw_pdb = RAW / f"{pdb_lc.upper()}.pdb"
        if not raw_pdb.exists():
            skipped["no_pdb"] += 1
            continue
        seq = meta[pdb_lc]["peptide_sequence"]
        chains = split_chains(raw_pdb)
        pep_chain = find_peptide_chain(chains, seq)
        if pep_chain is None:
            skipped["no_chain"] += 1
            continue
        pep_lines = chains[pep_chain]
        rec_lines = [l for c, ls in chains.items() if c != pep_chain for l in ls]
        pep_path = tmp / f"{pdb_lc}_pep.pdb"
        write_lines(pep_path, pep_lines)
        rec_coords = heavy_coords(rec_lines)
        if rec_coords.size == 0:
            skipped["no_chain"] += 1
            continue
        try:
            ent = compute_entropy_sums(pep_path, seq, receptor_coords=rec_coords, cutoff=4.5)
        except Exception as e:  # noqa: BLE001
            skipped["ent_err"] += 1
            continue

        vina = sc.get("vina_score")
        ad4 = sc.get("ad4_score")
        nc = sc.get("n_contact_residues") or ent["n_contact"]
        pkd = float(meta[pdb_lc]["experimental_pkd"])
        aff_type = meta[pdb_lc].get("affinity_type", "Kd")

        if vina is None or vina == 0.0:
            continue

        pred_leg = predict_legacy(vina, ad4, nc, CAL_LEGACY)
        pred_rid = predict_ridge(vina, ad4, nc, CAL_RIDGE)
        pred_ent = predict_entropy(vina, ent["s_ss_weighted"], CAL_ENT)
        results.append({
            "pdb": pdb_lc,
            "affinity_type": aff_type,
            "pkd": pkd,
            "dg_exp": kd_to_dg(pkd),
            "vina": vina,
            "ad4": ad4,
            "n_contact": nc,
            "s_ss_weighted": ent["s_ss_weighted"],
            "s_sc_sum": ent["s_sc_sum"],
            "s_bb_sum": ent["s_bb_sum"],
            "pred_legacy": pred_leg,
            "pred_ridge": pred_rid,
            "pred_entropy": pred_ent,
            "in_training": pdb_lc in TRAIN_OVERLAP,
        })

    out = ROOT / "data" / "eval_holdout_calibrations.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}  ({len(results)} complexes; skipped {skipped})")

    for subset_name, predicate in [
        ("ALL", lambda r: True),
        ("HELD-OUT (excl. PepSet-6 train)", lambda r: not r["in_training"]),
        ("HELD-OUT, Kd-only", lambda r: not r["in_training"] and r["affinity_type"] == "Kd"),
        ("TRAINING-OVERLAP only", lambda r: r["in_training"]),
    ]:
        rows = [r for r in results if predicate(r)]
        if not rows:
            continue
        dg = [r["dg_exp"] for r in rows]
        leg = [r["pred_legacy"] for r in rows]
        rid = [r["pred_ridge"] for r in rows]
        ent = [r["pred_entropy"] for r in rows]
        vina = [r["vina"] for r in rows]
        print(f"\n=== {subset_name} (n={len(rows)}) ===")
        print(f"  v1.0 legacy   r={pearson(dg, leg):+.3f}  RMSE={rmse(dg, leg):.2f}")
        print(f"  v1.1 ridge    r={pearson(dg, rid):+.3f}  RMSE={rmse(dg, rid):.2f}")
        print(f"  v1.2 entropy  r={pearson(dg, ent):+.3f}  RMSE={rmse(dg, ent):.2f}")
        print(f"  Vina-only     r={pearson(dg, vina):+.3f}")


if __name__ == "__main__":
    main()
