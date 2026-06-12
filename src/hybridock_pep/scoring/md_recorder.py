"""MD trajectory recorder — capture the expensive MD signal so an ML surrogate can later learn it.

The interaction-entropy MD (``interaction_entropy.sample_interaction_energies``) runs a short Langevin
trajectory and records the per-frame receptor-peptide interaction energy, but only the summary
(⟨E_int⟩, std, −TΔS) survives. The per-frame series — and the structural fluctuation that drives the
entropy — is the training signal for a cheap surrogate (docs E70: −TΔS is only weakly predictable from
static features r≈0.27, so the surrogate genuinely needs recorded trajectories, not hand-features).

This module appends one JSON record per scored complex to a growing dataset (JSONL). Each record holds:
  * the full per-frame E_int series (the entropy estimator's raw input)
  * Cα RMSF over the trajectory (structural mobility — what the peptide actually DID)
  * the derived MD outputs (e_int_mean/std, minus_tds) for supervised targets
  * cheap static features + sequence, so a model can be trained input→MD-output without re-running MD
Accumulate across runs -> train a GNN/regressor to predict minus_tds, then drop the MD at inference.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/md_surrogate_dataset.jsonl")


def record_trajectory(
    complex_id: str,
    peptide_seq: str,
    interaction_energies: np.ndarray | list[float],
    minus_tds: float,
    *,
    ca_rmsf: list[float] | None = None,
    static_features: dict | None = None,
    dataset_path: Path = DEFAULT_DATASET,
) -> None:
    """Append one MD record to the surrogate-training dataset (JSONL, one record per line).

    Args:
        complex_id: Unique identifier (e.g. ``1YCR_pose03``).
        peptide_seq: One-letter peptide sequence (model input).
        interaction_energies: Per-frame E_int series (kcal/mol) — the raw entropy-estimator input.
        minus_tds: The derived −TΔS_IE (kcal/mol) — the primary supervised target.
        ca_rmsf: Optional per-residue Cα RMSF over the trajectory (Å) — structural fluctuation signal.
        static_features: Optional cheap descriptors (geometry/sequence) as additional model inputs.
        dataset_path: JSONL file to append to (created if absent).
    """
    e = np.asarray(interaction_energies, dtype=float)
    record = {
        "id": complex_id,
        "seq": peptide_seq,
        "n_frames": int(e.size),
        "e_int_series": [round(float(x), 4) for x in e],
        "e_int_mean": float(e.mean()) if e.size else None,
        "e_int_std": float(e.std()) if e.size else None,
        "minus_tds": float(minus_tds),
        "ca_rmsf": [round(float(x), 4) for x in ca_rmsf] if ca_rmsf is not None else None,
        "static_features": static_features or {},
    }
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("Recorded MD trajectory for %s (%d frames, −TΔS=%.2f) -> %s",
                complex_id, e.size, minus_tds, dataset_path)


def load_dataset(dataset_path: Path = DEFAULT_DATASET) -> list[dict]:
    """Load all recorded MD trajectory records (for surrogate training/inspection)."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return []
    return [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]


def dataset_summary(dataset_path: Path = DEFAULT_DATASET) -> dict:
    """Quick stats on the accumulated surrogate dataset (n records, frame counts, target range)."""
    recs = load_dataset(dataset_path)
    if not recs:
        return {"n_records": 0}
    tds = [r["minus_tds"] for r in recs]
    return {
        "n_records": len(recs),
        "unique_complexes": len({r["id"].split("_pose")[0] for r in recs}),
        "mean_frames": float(np.mean([r["n_frames"] for r in recs])),
        "minus_tds_range": [float(np.min(tds)), float(np.max(tds))],
        "has_rmsf": sum(1 for r in recs if r.get("ca_rmsf")),
    }
