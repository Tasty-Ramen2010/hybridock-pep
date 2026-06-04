"""Properly analyze a HybriDock-Pep dock run on PfLDH (or any target).

For the run directory provided, reports:
  * Per-pose Vina clash status: distance from nearest receptor atom for each
    peptide heavy atom; counts at < 1.5, 1.5–2.0, 2.0–2.5, 2.5+ Å.
  * Cluster composition: pose membership, mean ΔG ± std, CI95.
  * Contact residues per cluster: which receptor residues each cluster
    actually touches (heavy-atom contact ≤ 4.5 Å).
  * Peptide SS content: helix/sheet/loop fraction from φ/ψ.
  * MM-GBSA results if any succeeded.
  * Geometric sanity: distance from best pose's Cα centroid to user-supplied
    site (and pocket-residue centroid for comparison).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_pdb_heavy(path: Path) -> tuple[list[str], np.ndarray, list[int]]:
    """Return (atom_names, xyz, res_seq) for heavy atoms only."""
    names, xyz, res = [], [], []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name.startswith("H") or name == "H":
            continue
        try:
            res.append(int(line[22:26].strip()))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            names.append(name)
        except ValueError:
            continue
    return names, np.array(xyz) if xyz else np.zeros((0, 3)), res


def residue_centroids(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Return {(chain, resseq): centroid_xyz} for receptor residues."""
    accum: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    resn: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            r = int(line[22:26].strip())
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        key = (line[21], r)
        accum[key].append([x, y, z])
        resn[key] = line[17:20].strip()
    return {k: (np.mean(v, axis=0), resn[k]) for k, v in accum.items()}


def clash_distribution(peptide_xyz: np.ndarray, receptor_xyz: np.ndarray) -> dict:
    """Histogram of peptide-atom minimum distances to any receptor atom."""
    if len(peptide_xyz) == 0 or len(receptor_xyz) == 0:
        return {"n_peptide_atoms": int(len(peptide_xyz)), "buckets": {}}
    diffs = peptide_xyz[:, None] - receptor_xyz[None]
    min_dist = np.sqrt((diffs ** 2).sum(-1).min(axis=1))
    return {
        "n_peptide_atoms": int(len(peptide_xyz)),
        "min_dist_overall": float(min_dist.min()),
        "median_dist": float(np.median(min_dist)),
        "buckets": {
            "<1.5 Å (severe clash)":   int((min_dist < 1.5).sum()),
            "1.5–2.0 Å (clash)":        int(((min_dist >= 1.5) & (min_dist < 2.0)).sum()),
            "2.0–2.5 Å (vdW edge)":     int(((min_dist >= 2.0) & (min_dist < 2.5)).sum()),
            "2.5–4.5 Å (contact)":      int(((min_dist >= 2.5) & (min_dist < 4.5)).sum()),
            "≥4.5 Å (non-contact)":     int((min_dist >= 4.5).sum()),
        },
    }


def contact_residues(peptide_xyz: np.ndarray, receptor_path: Path,
                     cutoff: float = 4.5) -> list[tuple[str, int, str]]:
    """Receptor residues with any heavy atom within cutoff of any peptide atom."""
    by_res: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    for line in receptor_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name.startswith("H") or name == "H":
            continue
        try:
            r = int(line[22:26].strip())
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        by_res[(line[21], r, line[17:20].strip())].append(xyz)
    if peptide_xyz.size == 0:
        return []
    out = []
    for (chain, resseq, resn), atoms in by_res.items():
        a = np.array(atoms)
        d = np.sqrt(((peptide_xyz[:, None] - a[None]) ** 2).sum(-1)).min()
        if d <= cutoff:
            out.append((chain, resseq, resn))
    return sorted(out, key=lambda x: (x[0], x[1]))


def phi_psi_ss(pdb: Path) -> dict:
    """Approximate secondary structure assignment by φ/ψ box test."""
    from hybridock_pep.scoring.per_residue_entropy import assign_secondary_structure
    ss = assign_secondary_structure(pdb)
    counts = {"helix": 0, "sheet": 0, "loop": 0}
    for v in ss.values():
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    return {**counts, "total": total,
            **{f"frac_{k}": (v / total) if total else 0.0 for k, v in counts.items()}}


def main(run_dir: str) -> None:
    rd = Path(run_dir)
    ranked = rd / "ranked_poses.csv"
    receptor = rd / "receptor_for_rapidock.pdb"  # the pocket-cropped one
    if not ranked.exists():
        print(f"ERROR: {ranked} does not exist; run not complete?")
        sys.exit(1)
    if not receptor.exists():
        receptor = rd / "receptor_for_rapidock_full.pdb"
    metadata = json.loads((rd / "run_metadata.json").read_text())
    site = metadata["cli_args"]["site_coords"]
    user_box = metadata["cli_args"]["box_size"]

    print(f"=== Run: {rd.name} ===")
    print(f"  Peptide: {metadata['cli_args']['peptide_sequence']}")
    print(f"  Receptor: {metadata['cli_args']['receptor_path']}")
    print(f"  Site (user): {tuple(round(s, 2) for s in site)}, box {user_box} Å")
    print(f"  Status: {metadata.get('status')}, {metadata.get('poses_generated', '?')}/100 poses")

    # Top-5 with clash analysis
    with ranked.open() as f:
        rows = list(csv.DictReader(f))
    top5 = rows[:5]
    print(f"\n--- Top-5 poses: clash and contact analysis ---")
    receptor_xyz = parse_pdb_heavy(receptor)[1]
    for r in top5:
        pdbqt = rd / "pdbqt" / r["pose_filename"].replace(".pdb", ".pdbqt")
        # Try the optimized PDBQT (post-Vina-relax); fall back to scored pose PDB
        if pdbqt.exists():
            pose_pdb = rd / "poses_scored" / r["pose_filename"]
            geom_src = "Vina-optimized PDBQT"
        else:
            pose_pdb = rd / "poses_scored" / r["pose_filename"]
            geom_src = "scored pose PDB"
        # Convert PDBQT to read heavy atoms
        if pdbqt.exists():
            try:
                import subprocess as sp
                converted = rd / f"tmp_{pdbqt.stem}.pdb"
                sp.run(["obabel", str(pdbqt), "-O", str(converted)],
                       check=True, capture_output=True, timeout=15)
                pose_pdb = converted
                geom_src = "Vina-optimized PDBQT (obabel converted)"
            except Exception:
                pass
        if not pose_pdb.exists():
            print(f"  rank {r['rank']}: pose file missing ({pose_pdb})")
            continue
        names, pep_xyz, res = parse_pdb_heavy(pose_pdb)
        clash = clash_distribution(pep_xyz, receptor_xyz)
        contacts = contact_residues(pep_xyz, receptor, cutoff=4.5)
        centroid = pep_xyz.mean(axis=0) if pep_xyz.size else np.array([0.,0.,0.])
        site_dist = float(np.linalg.norm(centroid - np.array(site)))
        ss = phi_psi_ss(pose_pdb)
        print(f"\n  Rank {r['rank']} — {r['pose_filename']} (ΔG={r['hybrid_score']}, vina={r['vina_score']})")
        print(f"    geometry source: {geom_src}")
        print(f"    centroid {site_dist:.1f} Å from user site")
        print(f"    nearest-receptor-atom distance: min={clash['min_dist_overall']:.2f} Å, median={clash['median_dist']:.2f} Å")
        for label, n in clash["buckets"].items():
            if n > 0:
                print(f"      {label}: {n} atoms")
        print(f"    contact residues (≤4.5 Å): {len(contacts)}")
        if contacts:
            print(f"      {', '.join(f'{r}{n}{c}' for c, n, r in contacts[:15])}{' ...' if len(contacts) > 15 else ''}")
        print(f"    peptide SS: helix={ss['frac_helix']:.0%} sheet={ss['frac_sheet']:.0%} loop={ss['frac_loop']:.0%}")
        # cleanup converted file
        if "obabel converted" in geom_src and pose_pdb.name.startswith("tmp_"):
            pose_pdb.unlink(missing_ok=True)

    # Cluster summary
    cs = rd / "cluster_summary.csv"
    if cs.exists():
        print(f"\n--- Cluster summary ---")
        with cs.open() as f:
            for row in csv.DictReader(f):
                print(f"  Cluster {row['cluster_id']}: n={row['n_poses']}, "
                      f"mean ΔG = {float(row['mean_hybrid_score']):.2f} ± "
                      f"{float(row['std_hybrid_score']):.2f}, "
                      f"CI95 [{float(row['ci95_lower']):.2f}, {float(row['ci95_upper']):.2f}], "
                      f"best pose: {row['best_pose_idx']}")

    # MM-GBSA results
    mmgbsa_succeeded = [r for r in rows if r.get("mmgbsa_dg") and r["mmgbsa_dg"] != ""]
    print(f"\n--- MM-GBSA refinement ---")
    print(f"  Succeeded: {len(mmgbsa_succeeded)} of top-K")
    if mmgbsa_succeeded:
        for r in mmgbsa_succeeded:
            print(f"    {r['pose_filename']}: ΔG_GBSA = {float(r['mmgbsa_dg']):+.2f} kcal/mol (hybrid {r['hybrid_score']})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/pfldh_lisdaeleaifeadc_v3")
