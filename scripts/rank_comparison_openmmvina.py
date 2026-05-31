#!/usr/bin/env python3
"""Ranking benchmark: OpenMM short minimization → Vina re-score.

Hypothesis: steric clashes in raw diffusion poses (~32% have Vina > 0) kill
the Vina signal.  A brief gradient-descent minimization (peptide free,
receptor fixed) should remove clashes and let Vina discriminate by actual
interaction quality.

Protocol per complex:
  1. Load receptor + pose into OpenMM (AMBER ff14SB, no solvent).
  2. Fix all receptor heavy atoms; minimize peptide (+ receptor H if any)
     for up to MAX_STEPS steps or until |ΔE| < E_TOL.
  3. Write minimized pose as PDB.
  4. Convert to PDBQT (obabel → strip protein tags → rigid-ligand format).
  5. Vina --score_only.
  6. Rank 5 poses by Vina score → compute ranking metrics vs RMSD labels.

Usage (score-env):
    conda run -n score-env python3 scripts/rank_comparison_openmmvina.py \
        --n-per-bucket 15 --out-dir logs/openmmvina_ranking --seed 42

Compares against plain-Vina results from logs/vina_ranking/ranking_results.json
if it exists.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
log = logging.getLogger("openmmvina")

REPO     = Path(__file__).resolve().parent.parent
BENCH300 = REPO / "logs" / "analysis_bench300" / "benchmark_results.json"
CSV300   = REPO / "data" / "benchmark300.csv"

# Minimisation settings
MAX_STEPS  = 500          # gradient descent steps (enough to relieve clashes)
E_TOL      = 10.0         # kJ/mol convergence criterion
RESTRAIN_K = 1000.0       # kJ/mol/nm² force constant on receptor heavy atoms
CLASH_THRESHOLD_RAW = 50.0  # kcal/mol; above this → raw Vina considered clash


# ── OpenMM minimisation ───────────────────────────────────────────────────────

def minimize_pose_openmm(receptor_pdb: Path, pose_pdb: Path, out_pdb: Path,
                          max_steps: int = MAX_STEPS,
                          restrain_k: float = RESTRAIN_K,
                          e_tol: float = E_TOL) -> bool:
    """Minimize pose with AMBER ff14SB + GBn2 implicit solvent (peptide only).

    Receptor pocket PDBs are truncated chains → AMBER FF can't handle them.
    We minimize the peptide alone (removes intra-peptide clashes / bad geometry).
    Then Vina scores the minimised peptide against the original receptor.

    Returns True on success, False on failure (script falls back to raw pose).
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError:
        log.error("OpenMM not installed")
        return False

    try:
        pep = app.PDBFile(str(pose_pdb))
        ff  = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")

        modeller = app.Modeller(pep.topology, pep.positions)
        modeller.addHydrogens(ff, pH=7.0)

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        platform = mm.Platform.getPlatformByName("CPU")
        sim = app.Simulation(modeller.topology, system, integrator, platform)
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy(
            tolerance=e_tol * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_steps,
        )

        state   = sim.context.getState(getPositions=True)
        min_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # nm, plain numpy

        # Write minimized peptide PDB — keep only heavy atoms (no H for PDBQT compat)
        with open(str(out_pdb), "w") as fh:
            n = 0
            for i, atom in enumerate(modeller.topology.atoms()):
                if atom.element is not None and atom.element.symbol == "H":
                    continue  # skip H — obabel handles re-addition for PDBQT
                n += 1
                res   = atom.residue
                resid = int(res.id) if res.id.isdigit() else n
                x = float(min_pos[i, 0]) * 10.0  # nm → Å
                y = float(min_pos[i, 1]) * 10.0
                z = float(min_pos[i, 2]) * 10.0
                elem = atom.element.symbol if atom.element else atom.name[0]
                fh.write(
                    f"ATOM  {n:5d} {atom.name:<4s} {res.name:<3s} A{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            fh.write("END\n")
        return True

    except Exception as e:
        log.debug("OpenMM minimization failed (%s): %s", pose_pdb.name, e)
        return False


# ── Vina scoring (same as rank_comparison_vina.py) ────────────────────────────

ADFR    = Path("/home/igem/ADFRsuite_x86_64Linux_1.0/bin")
PREP_REC = ADFR / "prepare_receptor"
OBABEL   = ADFR / "obabel"


def prepare_receptor_pdbqt(pdb: Path, out_dir: Path) -> Path:
    out = out_dir / "receptor.pdbqt"
    if out.exists() and out.stat().st_size > 200:
        return out
    r = subprocess.run(
        [str(PREP_REC), "-r", str(pdb), "-o", str(out), "-A", "checkhydrogens"],
        capture_output=True, text=True, timeout=30,
    )
    if not out.exists() or out.stat().st_size < 50:
        raise RuntimeError(f"prepare_receptor failed: {r.stderr[:200]}")
    return out


def prepare_pose_pdbqt(pose_pdb: Path, out_dir: Path, label: str) -> Path:
    out = out_dir / f"{label}.pdbqt"
    if out.exists() and out.stat().st_size > 50:
        return out
    subprocess.run(
        [str(OBABEL), "-i", "pdb", str(pose_pdb), "-o", "pdbqt", "-O", str(out)],
        capture_output=True, text=True, check=False, timeout=20,
    )
    if not out.exists() or out.stat().st_size < 20:
        return None
    # Strip protein-format tags (BEGIN_RES/END_RES) → rigid ligand format
    lines = out.read_text().splitlines()
    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        return None
    out.write_text("ROOT\n" + "\n".join(atom_lines) + "\nENDROOT\nTORSDOF 0\n")
    return out


def _pdb_coords(p: Path) -> np.ndarray:
    """Read heavy-atom coordinates from PDB/PDBQT by parsing text."""
    pts = []
    for ln in p.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        try:
            pts.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except ValueError:
            continue
    return np.array(pts) if pts else np.zeros((0, 3))


def box_from_poses(pose_pdbs: list[Path], crystal_pdb: Path, margin: float = 8.0):
    """Compute box center + size from all poses + crystal."""
    all_pts = []
    for p in pose_pdbs + [crystal_pdb]:
        if not p or not p.exists():
            continue
        pts = _pdb_coords(p)
        if pts.size:
            all_pts.append(pts)
    if not all_pts:
        raise RuntimeError("No valid atoms found for box computation")
    combined = np.vstack(all_pts)
    center = combined.mean(axis=0).tolist()
    half_extents = (combined.max(axis=0) - combined.min(axis=0)) / 2 + margin
    box_size = float(max(half_extents) * 2)
    box_size = max(30.0, min(box_size, 70.0))
    return center, box_size


def vina_score(rec_pdbqt: Path, pose_pdbqt: Path, center, box_size, optimize=True) -> float:
    """Score one pose with Vina. Returns kcal/mol (raw, no minimize)."""
    from vina import Vina
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(pose_pdbqt))
    v.compute_vina_maps(
        center=center,
        box_size=[box_size, box_size, box_size],
    )
    score = v.score()[0]
    if optimize and score > CLASH_THRESHOLD_RAW:
        try:
            v.optimize()
            score = v.score()[0]
        except Exception:
            pass
    return float(score)


def ranking_metrics(scores: list[float], rmsds: list[float]) -> dict:
    """Compute ranking quality metrics."""
    scores = np.array(scores)
    rmsds  = np.array(rmsds)
    valid  = ~np.isnan(scores)
    if valid.sum() < 2:
        return {k: float("nan") for k in
                ["top1_rmsd", "random_mean_rmsd", "best_rmsd", "oracle_gap",
                 "kendall_tau", "spearman_r", "p_select_best", "gap_recovered_frac"]}

    scores_v = scores[valid]
    rmsds_v  = rmsds[valid]

    # best Vina score = most negative → lowest index after argsort
    ranked_idx = np.argsort(scores_v)
    top1_rmsd  = float(rmsds_v[ranked_idx[0]])
    best_rmsd  = float(rmsds_v.min())
    random_mean = float(rmsds_v.mean())
    oracle_gap  = random_mean - best_rmsd

    p_best = float(rmsds_v[ranked_idx[0]] == best_rmsd)
    tau, _  = scipy_stats.kendalltau(scores_v, rmsds_v)
    rho, _  = scipy_stats.spearmanr(scores_v, rmsds_v)
    gap_rec = (random_mean - top1_rmsd) / oracle_gap if oracle_gap > 0.01 else float("nan")

    return {
        "top1_rmsd":          top1_rmsd,
        "random_mean_rmsd":   random_mean,
        "best_rmsd":          best_rmsd,
        "oracle_gap":         oracle_gap,
        "kendall_tau":        float(tau),
        "spearman_r":         float(rho),
        "p_select_best":      p_best,
        "gap_recovered_frac": gap_rec,
        "n_valid":            int(valid.sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="OpenMM-minimize → Vina ranking benchmark")
    ap.add_argument("--n-per-bucket", type=int, default=15)
    ap.add_argument("--out-dir",      default="logs/openmmvina_ranking")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--no-minimize",  action="store_true",
                    help="Skip OpenMM step (debug mode: just Vina on raw poses)")
    ap.add_argument("--min-steps",    type=int, default=MAX_STEPS)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/openmmvina_ranking.log"),
        ],
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    # Load data
    with open(BENCH300) as f:
        bench300 = json.load(f)
    df300 = pd.read_csv(CSV300)
    name_to_row = {r["name"]: r for _, r in df300.iterrows()}

    # Stratified sample: same as rank_comparison_vina.py
    buckets = {"short": [], "medium": [], "long": [], "very_long": []}
    for cname in bench300:
        row = name_to_row.get(cname)
        if row is None:
            continue
        lb = row.get("length_bucket", "medium")
        if lb in buckets and "pretrained" in bench300[cname]:
            poses_dir = Path(bench300[cname]["pretrained"]["poses_dir"])
            if poses_dir.exists() and len(list(poses_dir.glob("pose_*.pdb"))) == 5:
                buckets[lb].append(cname)

    selected = []
    for lb, names in buckets.items():
        rng.shuffle(names)
        selected.extend([(n, lb) for n in names[:args.n_per_bucket]])

    log.info("Running OpenMM-minimize → Vina on %d complexes", len(selected))
    results = {}
    n_ok = 0
    n_fail = 0

    for idx, (cname, lb) in enumerate(selected):
        row  = name_to_row[cname]
        res0 = bench300[cname]["pretrained"]
        ref_rmsds = res0["ref_rmsds"]
        poses_dir = Path(res0["poses_dir"])
        receptor_pdb = Path(row["receptor"])
        crystal_pdb  = Path(row["peptide_pdb"])
        ss = row.get("ss_class", "HELIX")

        cdir = out_root / cname
        cdir.mkdir(exist_ok=True)

        # Step 1: prepare receptor PDBQT (shared across all poses)
        try:
            rec_pdbqt = prepare_receptor_pdbqt(receptor_pdb, cdir)
        except Exception as e:
            log.warning("[%d/%d] %s receptor prep failed: %s", idx+1, len(selected), cname, e)
            n_fail += 1
            continue

        pose_pdbs = [poses_dir / f"pose_{i}.pdb" for i in range(5)]

        # Step 2: OpenMM minimise each pose
        min_pdbs = []
        for i, ppdb in enumerate(pose_pdbs):
            if args.no_minimize:
                min_pdbs.append(ppdb)
                continue
            out_min = cdir / f"pose_{i}_min.pdb"
            if not out_min.exists():
                ok = minimize_pose_openmm(
                    receptor_pdb, ppdb, out_min, max_steps=args.min_steps)
                if not ok or not out_min.exists():
                    log.debug("%s pose %d minimization failed — using raw", cname, i)
                    out_min = ppdb  # fallback to raw
            min_pdbs.append(out_min)

        # Step 3: Vina score each minimised pose
        try:
            center, box_size = box_from_poses(pose_pdbs, crystal_pdb)
        except Exception as e:
            log.warning("Box computation failed for %s: %s", cname, e)
            n_fail += 1
            continue

        scores = []
        for i, mpdb in enumerate(min_pdbs):
            pdbqt = prepare_pose_pdbqt(mpdb, cdir, f"pose_{i}_min")
            if pdbqt is None:
                scores.append(float("nan"))
                continue
            try:
                s = vina_score(rec_pdbqt, pdbqt, center, box_size, optimize=False)
                scores.append(s)
            except Exception as e:
                log.debug("Vina failed on %s pose %d: %s", cname, i, e)
                scores.append(float("nan"))

        metrics = ranking_metrics(scores, list(ref_rmsds))
        metrics.update({"lb": lb, "ss": ss, "vina_scores_min": scores})
        results[cname] = metrics
        n_ok += 1

        log.info("[%d/%d] %s [%s/%s] box=%.0fÅ: rand=%.2f vina_top1=%.2f best=%.2f "
                 "τ=%.3f ρ=%.3f P(best)=%.0f%% rec=%.0f%%",
                 idx+1, len(selected), cname, lb, ss, box_size,
                 metrics["random_mean_rmsd"], metrics["top1_rmsd"],
                 metrics["best_rmsd"],
                 metrics["kendall_tau"] if not np.isnan(metrics["kendall_tau"]) else -99,
                 metrics["spearman_r"]  if not np.isnan(metrics["spearman_r"])  else -99,
                 metrics["p_select_best"] * 100 if not np.isnan(metrics["p_select_best"]) else -99,
                 metrics["gap_recovered_frac"] * 100 if not np.isnan(metrics["gap_recovered_frac"]) else -99)

    # ── Save ──────────────────────────────────────────────────────────────
    out_json = out_root / "ranking_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_json)

    df = pd.DataFrame(results).T
    for col in ["top1_rmsd", "best_rmsd", "random_mean_rmsd", "oracle_gap",
                "kendall_tau", "spearman_r", "p_select_best", "gap_recovered_frac"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(out_root / "ranking_summary.csv")

    # ── Print report ──────────────────────────────────────────────────────
    label = "OpenMM-minimize → Vina" if not args.no_minimize else "Raw Vina (no minimize)"
    print(f"\n{'='*75}")
    print(f"RANKING REPORT — {label}")
    print(f"{'='*75}")
    print(f"N={len(df)}  |  minimize_steps={args.min_steps if not args.no_minimize else 0}")

    def bucket_report(sub, lbl):
        n = len(sub)
        if n == 0:
            return
        rand = sub["random_mean_rmsd"].mean()
        vt1  = sub["top1_rmsd"].mean()
        best = sub["best_rmsd"].mean()
        pb   = sub["p_select_best"].mean() * 100
        tau  = sub["kendall_tau"].mean()
        rho  = sub["spearman_r"].mean()
        rec  = sub["gap_recovered_frac"].mean() * 100
        arrow = "↓" if rand > vt1 else "↑"
        valid = sub.dropna(subset=["top1_rmsd", "random_mean_rmsd"])
        if len(valid) > 4:
            _, p = scipy_stats.ttest_rel(valid["top1_rmsd"], valid["random_mean_rmsd"])
            sig = f"p={p:.3f}" + ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
        else:
            sig = "n/a"
        print(f"  {lbl:14s} N={n:3d}  rand={rand:.3f}  top1={vt1:.3f} {arrow}  "
              f"best={best:.3f}  P(best)={pb:.0f}%  τ={tau:.3f}  ρ={rho:.3f}  "
              f"gap_rec={rec:.0f}%  {sig}")

    print(f"\n{'Bucket':14s} {'N':>4}  {'rand':>6}  {'top1':>8}   {'best':>6}  "
          f"{'P(best)':>7}  {'τ':>6}  {'ρ':>6}  {'gap_rec':>8}  sig")
    print("-" * 85)
    bucket_report(df, "OVERALL")
    for lb in ["short", "medium", "long", "very_long"]:
        bucket_report(df[df["lb"] == lb], lb)
    print()
    for ss in ["HELIX", "SHEET", "UNUSUAL"]:
        bucket_report(df[df["ss"] == ss], ss)

    oracle_gap = df["oracle_gap"].mean()
    vina_rec   = (df["random_mean_rmsd"] - df["top1_rmsd"]).mean()
    print(f"\n  Oracle gap (random→best):              {oracle_gap:.3f}Å")
    print(f"  Recovery (random→top1_min_vina):       {vina_rec:.3f}Å")
    print(f"  Gap recovery fraction:                 {df['gap_recovered_frac'].mean()*100:.0f}%")

    # Compare to plain-Vina if available
    plain_vina_json = REPO / "logs" / "vina_ranking" / "ranking_results.json"
    if plain_vina_json.exists():
        with open(plain_vina_json) as f:
            vina_raw = json.load(f)
        common = set(df.index) & set(vina_raw.keys())
        if common:
            raw_top1s = [vina_raw[c]["top1_rmsd_vina"] for c in common
                         if "top1_rmsd_vina" in vina_raw[c]]
            min_top1s = [df.loc[c, "top1_rmsd"] for c in common
                         if not np.isnan(df.loc[c, "top1_rmsd"])]
            if raw_top1s and min_top1s:
                print(f"\n  vs plain-Vina on {len(common)} matched complexes:")
                print(f"    Plain Vina top1:    {np.mean(raw_top1s):.3f}Å")
                print(f"    OpenMM+Vina top1:   {np.mean(min_top1s):.3f}Å")
                delta = np.mean(raw_top1s) - np.mean(min_top1s)
                print(f"    Improvement:       {delta:+.3f}Å")

    print(f"\n  Results: {out_json}")


if __name__ == "__main__":
    main()
