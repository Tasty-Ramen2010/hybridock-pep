"""Stage 1.4 -- fold the complex from sequence, and fold the answer back into the docking.

This is the glue between `cofold.py` (which knows how to move a co-folded peptide into the
user's frame) and the driver. It exists as its own stage because a co-folding model is a
fundamentally different kind of predictor from a docker, and pretending otherwise produces
wrong answers:

  * It does not take a receptor.  It builds its own from the sequence, in its own frame, with
    its own sidechain placement.  Everything this tool reports -- interface energy, BSA,
    clash, MM-GBSA -- is defined against the user's structure, so a co-folded complex has to
    be transplanted before any of it means anything.
  * It does not take a site.  If the user asked about one pocket and the co-folding model
    put the peptide in another, that is a disagreement worth printing, not a pose to average
    into the ensemble.
  * It is one answer, not an ensemble.  A single confident structure cannot be clustered,
    has no convergence curve, and carries no spread.  It enters as one more candidate that
    has to win on the same scoring function as the other hundred.

WHAT THE STAGE ACTUALLY DOES
  pose  : transplant the co-folded peptide onto the user's receptor and add it as a pose.
  prior : take only the threading direction from it and drop the RAPiDock poses that run the
          other way.  This is the cheap half and the better-evidenced one -- on 9CCE the
          co-folded axis picked out the forward-threading poses with 98% precision (103 of
          105 kept poses genuinely forward, from a pool that was 79% backwards).
  both  : the default.

WHAT WE HAVE MEASURED, AND WHAT WE HAVE NOT
On 9CCE, a structure released after Boltz-2's training cutoff so its prediction is genuine:
the transplanted co-folded pose is 1.90 A from the crystal peptide where our best of 500
sampled poses is 5.66 A, and the direction filter cuts the pool's median RMSD from 17.0 A to
11.1 A.  It does NOT improve our best-of-N -- our best pose was already threading forward --
so the prior's value is in constraining what the scorer may pick, not in what the sampler can
reach.  That is one complex.  scripts/coventry_boltz_grid.py runs the same measurement over
324 cells; until it reports, treat everything above as a single anecdote.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _seq_of(receptor: Path) -> str:
    """One-letter sequence of the receptor's longest chain."""
    from hybridock_pep.sampling.cofold import read_chains

    chains = read_chains(receptor)
    if not chains:
        raise ValueError(f"{receptor} has no standard protein residues")
    return max(chains.values(), key=lambda c: len(c.seq)).seq


def _pose_ca(pdb: Path) -> np.ndarray:
    """Peptide CA coordinates of a pose file (chain B, or the shortest chain)."""
    from hybridock_pep.sampling.cofold import read_chains

    chains = read_chains(pdb)
    if not chains:
        return np.empty((0, 3))
    pep = chains.get("B") or min(chains.values(), key=lambda c: len(c.seq))
    return pep.ca


def run_cofold_stage(config, receptor_path: Path, records: list) -> tuple[list, dict]:
    """Run co-folding and apply its output to the Stage 1 pose set.

    Args:
        config: DockConfig. Reads cofold, cofold_pose, cofold_mode, cofold_max_fold_rmsd,
            cofold_msa_server, peptide_sequence, site_coords and output_dir.
        receptor_path: The receptor every pose is scored against.
        records: Stage 1 PoseRecord list. Not mutated; a new list is returned.

    Returns:
        (records after the stage, metadata dict for run_metadata.json). On any failure the
        records come back untouched and the metadata carries an "error" key -- co-folding is
        an enhancement and must never take a docking run down.
    """
    meta: dict = {"requested": config.cofold, "mode": config.cofold_mode, "applied": False}
    if config.cofold == "none":
        return records, meta

    from hybridock_pep.analysis.direction import filter_by_direction
    from hybridock_pep.sampling import cofold as cf

    work = Path(config.output_dir) / "cofold"
    work.mkdir(parents=True, exist_ok=True)

    # --- 1. obtain the co-folded complex --------------------------------------
    try:
        if config.cofold == "import":
            if not config.cofold_pose or not Path(config.cofold_pose).exists():
                raise FileNotFoundError(
                    "--cofold import needs --cofold-pose pointing at a co-folded complex"
                )
            src = Path(config.cofold_pose)
            if src.suffix.lower() in (".cif", ".mmcif"):
                from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb

                src = _cif_to_pdb(src, work / "cofolded.pdb")
            complex_pdb = work / "cofolded.pdb"
            if src != complex_pdb:
                complex_pdb.write_text(src.read_text())
            meta["source"] = str(config.cofold_pose)
        else:
            from hybridock_pep.sampling import cofold_boltz

            complex_pdb = work / "cofolded.pdb"
            info = cofold_boltz.cofold(
                _seq_of(receptor_path), config.peptide_sequence, complex_pdb,
                name=config.run_id or "cofold", use_msa_server=config.cofold_msa_server,
            )
            meta["confidence"] = {k: v for k, v in info.items() if k != "path"}
    except Exception as exc:  # noqa: BLE001 -- never let this take the run down
        logger.warning("Stage 1.4 co-folding failed (%s: %s); continuing without it",
                       type(exc).__name__, exc)
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return records, meta

    # --- 2. transplant into the user's frame and gate -------------------------
    try:
        tr = cf.transplant(complex_pdb, receptor_path, site=config.site_coords)
    except ValueError as exc:
        logger.warning("Stage 1.4 transplant failed (%s); continuing without it", exc)
        meta["error"] = str(exc)
        return records, meta

    meta.update({
        "fold_rmsd": round(tr.fold_rmsd, 3),
        "n_aligned_pocket_residues": tr.n_aligned,
        "site_offset": None if tr.site_offset is None else round(tr.site_offset, 2),
    })
    if tr.fold_rmsd > config.cofold_max_fold_rmsd:
        msg = (f"co-folded receptor disagrees with the supplied structure "
               f"({tr.fold_rmsd:.2f} A over {tr.n_aligned} pocket residues, limit "
               f"{config.cofold_max_fold_rmsd:.1f} A); the transplant would be meaningless")
        logger.warning("Stage 1.4: %s", msg)
        meta["error"] = msg
        return records, meta
    if tr.site_offset is not None and tr.site_offset > config.box_size:
        # Report, do not silently drop: a co-folding model disagreeing with the user about
        # WHICH pocket is a scientifically interesting statement, not a glitch.
        logger.warning(
            "Stage 1.4: co-folded peptide sits %.1f A from the requested site, outside the "
            "%.0f A box — the co-folding model does not agree this is the binding site",
            tr.site_offset, config.box_size,
        )
        meta["site_disagreement"] = True

    out = records
    # --- 3a. as a candidate pose ----------------------------------------------
    if config.cofold_mode in ("pose", "both"):
        pose_pdb = cf.write_transplanted_pose(tr, receptor_path, work / "pose_cofold.pdb")
        try:
            from hybridock_pep.sampling.pose_io import _parse_single_pose

            rec = _parse_single_pose(len(records), pose_pdb)
            out = [*records, rec]
            meta["pose_added"] = str(pose_pdb)
            logger.info("Stage 1.4: added the co-folded pose as candidate %d", len(records))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 1.4: co-folded pose could not be parsed (%s: %s)",
                           type(exc).__name__, exc)
            meta["pose_error"] = f"{type(exc).__name__}: {exc}"

    # --- 3b. as a direction prior ---------------------------------------------
    if config.cofold_mode in ("prior", "both") and records:
        poses = [(str(r.pdb_path), _pose_ca(Path(r.pdb_path))) for r in records]
        poses = [(k, c) for k, c in poses if len(c) >= 2]
        kept, diag = filter_by_direction(poses, tr.direction, tr.peptide_ca)
        meta["direction_filter"] = {k: v for k, v in diag.items() if k != "cosines"}
        if diag.get("applied"):
            keep = set(kept)
            survivors = [r for r in out if str(r.pdb_path) in keep
                         or str(r.pdb_path).endswith("pose_cofold.pdb")]
            logger.info(
                "Stage 1.4: direction prior kept %d of %d sampled poses",
                len(survivors) - (1 if meta.get("pose_added") else 0), len(records),
            )
            out = survivors

    meta["applied"] = True
    return out, meta
