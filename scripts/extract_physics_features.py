#!/usr/bin/env python3
"""
extract_physics_features.py — Extract per-term PyRosetta ref2015 physics features
for all benchmark poses and save as pkl compatible with the confidence head pipeline.

Key design decisions:
  NO relaxation — FastRelax compresses score ranges 6×, destroying discrimination
  signal on diffusion model poses. Raw ref2015 score-only is confirmed better
  (τ=0.174) than FastRelax (τ=0.163, relax20 τ=0.139). Rank on raw, relax top-1
  for delivery.

  Interface ΔΔG computed by: score(complex) − score(receptor alone).
  This removes receptor self-energy (constant across poses, uninformative for ranking).

Feature vector per pose (16 dims):
  fa_atr, fa_rep, fa_sol, fa_intra_rep, fa_elec,
  hbond_bb_sc, hbond_sc, hbond_lr_bb, hbond_sr_bb,
  rama_prepro, fa_dun, p_aa_pp,
  interface_ddG,           ← E(complex) − E(receptor)
  total_score,             ← total ref2015 (raw, score-only)
  resp_delta_e,            ← E(raw) − E(after restrained peptide min)  [response feature]
  resp_ca_disp,            ← peptide Cα displacement under restrained min (Å)
  ref_rmsds ARE in benchmark_results.json — used as labels

Response features (resp_*) test the hypothesis that "how much a pose relaxes"
discriminates near-native from decoy WITHOUT committing to the flattened
minimized geometry (which empirically drops τ: FastRelax 0.174→0.139).
The 14 static terms stay RAW score-only; only the 2-dim relaxation *response*
is added.  Ablation in train_physics_head.py: feats[:14] vs feats[:16].
Mechanism: PyRosetta MinMover, peptide bb+chi free, receptor fixed, lbfgs.

Usage (base env with PyRosetta OR score-env):
  python3 scripts/extract_physics_features.py \
      --bench    [default] \
      --gen      [optional] \
      --out-dir  logs/diagnosis
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pickle

warnings.filterwarnings("ignore")

REPO      = Path(__file__).resolve().parent.parent
BENCH_JSON = REPO / "logs"  / "analysis_bench300" / "benchmark_results.json"
GEN_JSON   = REPO / "logs"  / "confidence_training_data" / "benchmark_results.json"
BENCH_CSV  = REPO / "data"  / "benchmark300.csv"
GEN_CSV    = REPO / "data"  / "confidence_training_500.csv"
OUT_DIR    = REPO / "logs"  / "diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("physics_feats")

SCORE_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec",
    "hbond_bb_sc", "hbond_sc", "hbond_lr_bb", "hbond_sr_bb",
    "rama_prepro", "fa_dun", "p_aa_pp",
]
# Static dim: len(SCORE_TERMS) + 2 (interface_ddG, total_score) = 14
# + response dim 2 (resp_delta_e, resp_ca_disp) = 16 total
_MIN_MAX_ITER = 50          # MinMover lbfgs iterations (cheap restrained relax)
_RECEPTOR_CACHE: dict[str, tuple[float, int]] = {}   # rec_pdb -> (e_rec, n_rec)


def _peptide_ca_coords(pose, n_rec: int, n_total: int) -> np.ndarray:
    """Cα coordinates of the peptide residues (n_rec+1 .. n_total) as (k,3) array."""
    coords = []
    for i in range(n_rec + 1, n_total + 1):
        res = pose.residue(i)
        if res.has("CA"):
            xyz = res.xyz("CA")
            coords.append([xyz.x, xyz.y, xyz.z])
    return np.asarray(coords, dtype=np.float64)


def _minimization_response(pose_complex, n_rec: int, sfxn, pyrosetta) -> tuple[float, float]:
    """Restrained relaxation of the peptide against a fixed receptor.

    Frees only peptide backbone+sidechain DOFs, holds the receptor rigid, runs
    a short lbfgs minimization, and reports how far the pose fell:

      resp_delta_e = E(raw) − E(min)   (energy shed; large for strained decoys)
      resp_ca_disp = peptide Cα RMSD between raw and minimized (no superposition;
                     receptor is fixed so the frame is shared)

    Never raises — returns (0.0, 0.0) on any failure so extraction continues.
    """
    try:
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover

        n_total = pose_complex.total_residue()
        if n_rec >= n_total:                       # no peptide residues resolved
            return 0.0, 0.0

        e_raw = sfxn(pose_complex)
        ca_pre = _peptide_ca_coords(pose_complex, n_rec, n_total)

        mm = MoveMap()
        mm.set_bb(False)
        mm.set_chi(False)
        mm.set_jump(False)
        for i in range(n_rec + 1, n_total + 1):
            mm.set_bb(i, True)
            mm.set_chi(i, True)

        minmover = MinMover(mm, sfxn, "lbfgs_armijo_nonmonotone", 0.01, True)
        minmover.max_iter(_MIN_MAX_ITER)
        minmover.apply(pose_complex)               # mutates a throwaway clone (see caller)

        e_min = sfxn(pose_complex)
        ca_post = _peptide_ca_coords(pose_complex, n_rec, n_total)

        delta_e = float(e_raw - e_min)
        if ca_pre.shape == ca_post.shape and ca_pre.size:
            ca_disp = float(np.sqrt(np.mean(np.sum((ca_post - ca_pre) ** 2, axis=1))))
        else:
            ca_disp = 0.0
        return delta_e, ca_disp

    except Exception as exc:                       # noqa: BLE001 — best-effort feature
        log.debug("min response failed: %s", exc)
        return 0.0, 0.0


def init_pyrosetta():
    import pyrosetta
    pyrosetta.init(
        " ".join(["-mute", "all",
                  "-use_input_sc",
                  "-ignore_unrecognized_res",
                  "-ignore_zero_occupancy", "false",
                  "-ex1", "-ex2aro",
                  "-no_his_his_pairE"]),
        silent=True
    )
    sfxn = pyrosetta.create_score_function("ref2015")
    return pyrosetta, sfxn


def score_pose_file(pdb_path: str, receptor_pdb: str,
                    pyrosetta, sfxn) -> dict | None:
    """
    Score a single pose combined with its receptor.
    Returns dict of physics features + interface_ddG + total_score.
    NO relaxation — raw score only.
    """
    from pyrosetta.rosetta.core.scoring import ScoreType

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
            tmp.write(open(receptor_pdb).read().rstrip())
            tmp.write("\nTER\n")
            tmp.write(open(pdb_path).read())
            tmp_path = tmp.name

        pose_complex = pyrosetta.pose_from_pdb(tmp_path)
        os.unlink(tmp_path)

        total = sfxn(pose_complex)
        e = pose_complex.energies().total_energies()

        # Receptor-only score for interface ΔΔG (cached per receptor PDB:
        # constant across all poses of a complex, and reloading + rescoring
        # the receptor every pose is the dominant cost otherwise).
        if receptor_pdb in _RECEPTOR_CACHE:
            e_rec, n_rec = _RECEPTOR_CACHE[receptor_pdb]
        else:
            receptor_pose = pyrosetta.pose_from_pdb(receptor_pdb)
            e_rec = sfxn(receptor_pose)
            n_rec = receptor_pose.total_residue()
            _RECEPTOR_CACHE[receptor_pdb] = (e_rec, n_rec)
        interface_ddG = total - e_rec

        feats = []
        for term in SCORE_TERMS:
            try:
                st = getattr(ScoreType, term)
                feats.append(float(e[st]))
            except Exception:
                feats.append(0.0)
        feats.append(float(interface_ddG))
        feats.append(float(total))

        # Response features: relax a CLONE (keep the 14 static feats raw).
        # Receptor is residues 1..n_rec in the merged complex (written first),
        # peptide is n_rec+1..end — so freeing residues > n_rec frees only the
        # peptide while holding the receptor rigid.
        resp_delta_e, resp_ca_disp = _minimization_response(
            pose_complex.clone(), n_rec, sfxn, pyrosetta
        )
        feats.append(float(resp_delta_e))
        feats.append(float(resp_ca_disp))

        return np.array(feats, dtype=np.float32)

    except Exception as e_exc:
        log.debug("score failed for %s: %s", pdb_path, e_exc)
        return None


def extract_features_for_dataset(json_path: str, label: str,
                                  out_pkl: Path, pyrosetta, sfxn,
                                  force_rerun: bool = False,
                                  receptor_map: dict[str, str] | None = None) -> dict:
    """
    Extract physics features for all poses in a benchmark_results.json.
    Saves incrementally to out_pkl; restores progress on restart.

    receptor_map: optional {complex_name: receptor_pdb_path}. Used when the JSON
    does not carry receptor_pdb and there is no scoring/ crop on disk (e.g. the
    n=100 gen_subset, whose receptors live in datasets/training_formatted_peppc/).
    """
    if out_pkl.exists() and not force_rerun:
        with open(out_pkl, "rb") as f:
            feat_map = pickle.load(f)
        log.info("%s: loaded %d existing features from %s", label, len(feat_map), out_pkl)
    else:
        feat_map = {}

    data = json.load(open(json_path))
    n_total = sum(len(res.get("ref_rmsds", []))
                  for cx in data.values()
                  for res in cx.values())
    n_done  = len(feat_map)
    n_ok = 0; n_skip = 0; t0 = time.time()

    for cx_i, (cname, model_results) in enumerate(data.items()):
        for mkey, res in model_results.items():
            poses_dir = Path(res["poses_dir"])
            rmsds     = res.get("ref_rmsds", [])
            rec_pdb   = (res.get("receptor_pdb")
                         or (receptor_map.get(cname) if receptor_map else None)
                         or _find_receptor(poses_dir))

            if rec_pdb is None:
                n_skip += len(rmsds); continue

            for pose_idx, rmsd in enumerate(rmsds):
                key = (cname, mkey, pose_idx)
                if key in feat_map:
                    n_ok += 1; continue

                pdb = poses_dir / f"pose_{pose_idx}.pdb"
                if not pdb.exists():
                    n_skip += 1; continue

                feat = score_pose_file(str(pdb), rec_pdb, pyrosetta, sfxn)
                if feat is None:
                    n_skip += 1; continue

                feat_map[key] = feat
                n_ok += 1

        # Save checkpoint every 10 complexes
        if (cx_i + 1) % 10 == 0:
            with open(out_pkl, "wb") as f: pickle.dump(feat_map, f)
            elapsed = time.time() - t0
            rate    = (n_ok + n_skip) / max(elapsed, 1)
            remain  = (n_total - len(feat_map)) / max(rate, 1e-6)
            log.info("  %s [%d/%d cx] %d ok %d skip  ETA=%.0f min",
                     label, cx_i+1, len(data), n_ok, n_skip, remain/60)

    with open(out_pkl, "wb") as f: pickle.dump(feat_map, f)
    log.info("%s: saved %d features → %s", label, len(feat_map), out_pkl)
    return feat_map


def _find_receptor(poses_dir: Path) -> str | None:
    """Heuristic: find the cropped receptor PDB near a poses_dir.

    Layouts seen in practice:
      bench300:  <complex>/<model>/poses/  → receptor at <complex>/scoring/receptor_cropped.pdb
      gen_ood:   <complex>/poses/          → receptor at <complex>/scoring/receptor_cropped.pdb
    The scoring/receptor_cropped.pdb is the exact pocket crop the production
    Vina/AD4 scorer uses, so it is also the right receptor for train/serve
    consistency.
    """
    # <complex> is one or two levels up depending on whether there is a model dir.
    complex_dirs = [poses_dir.parent, poses_dir.parent.parent]
    candidates: list[Path] = []
    for cd in complex_dirs:
        candidates.append(cd / "scoring" / "receptor_cropped.pdb")
        candidates.append(cd / "receptor.pdb")
        candidates.append(cd / "receptor_pocket.pdb")
    # Globs as a fallback (scoring dir first, then complex dir).
    for cd in complex_dirs:
        for sub in (cd / "scoring", cd):
            if sub.is_dir():
                for pat in ("*receptor*.pdb", "*pocket*.pdb", "*protein*.pdb"):
                    candidates.extend(sorted(sub.glob(pat)))
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _receptor_map_from_csv(csv_path: str) -> dict[str, str]:
    """Build {complex_name: receptor_pdb} from a training CSV (name,receptor,...)."""
    import csv as _csv
    rmap: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("name") and row.get("receptor"):
                rmap[row["name"]] = row["receptor"]
    return rmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",     action="store_true", default=True)
    ap.add_argument("--gen",       action="store_true")
    ap.add_argument("--out-dir",   default=str(OUT_DIR))
    ap.add_argument("--rerun",     action="store_true", help="Force re-extraction")
    # Custom dataset (e.g. n=100 gen subset): supply JSON + receptor CSV + out path.
    ap.add_argument("--json",      help="Custom benchmark_results.json to score")
    ap.add_argument("--csv",       help="CSV mapping complex name → receptor PDB")
    ap.add_argument("--out-pkl",   help="Output pkl for the custom dataset")
    ap.add_argument("--label",     default="custom")
    args = ap.parse_args()
    out = Path(args.out_dir)

    log.info("Initialising PyRosetta (score-only, no relaxation)...")
    pr, sfxn = init_pyrosetta()
    log.info("ref2015 ready. Extracting physics features (raw poses, NO relaxation).")
    log.info("Rationale: FastRelax compresses score range 6×, τ drops 0.174→0.139.")

    # Custom dataset path takes priority and runs alone.
    if args.json:
        if not (args.csv and args.out_pkl):
            ap.error("--json requires --csv (receptor map) and --out-pkl")
        rmap = _receptor_map_from_csv(args.csv)
        log.info("=== %s (custom) === receptor_map=%d entries", args.label, len(rmap))
        extract_features_for_dataset(
            args.json, args.label, Path(args.out_pkl),
            pr, sfxn, force_rerun=args.rerun, receptor_map=rmap)
        log.info("Custom extraction done → %s", args.out_pkl)
        return

    if args.bench:
        log.info("=== bench300 ===")
        extract_features_for_dataset(
            str(BENCH_JSON), "bench300",
            out / "feats_bench300_physics.pkl",
            pr, sfxn, force_rerun=args.rerun)

    if args.gen:
        log.info("=== gen_ood ===")
        extract_features_for_dataset(
            str(GEN_JSON), "gen_ood",
            out / "feats_gen_ood_physics.pkl",
            pr, sfxn, force_rerun=args.rerun)

    log.info("Done. Feature dim=%d  terms: %s + interface_ddG + total_score "
             "+ resp_delta_e + resp_ca_disp",
             len(SCORE_TERMS)+4, SCORE_TERMS)


if __name__ == "__main__":
    main()
