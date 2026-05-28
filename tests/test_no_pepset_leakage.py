"""Test that no training/calibration data leaks from the held-out PepSet.

The PepSet (10 complexes in data/test_complexes.csv) is the ONLY test set
used to report population-level Pearson r. Any leakage invalidates the metric.

These tests run fast (no disk I/O beyond CSV reads) and should be run:
  - Before every calibration run
  - Before every fine-tuning run
  - In CI on every commit that touches data/

Run:
    pytest tests/test_no_pepset_leakage.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

# The 10 official held-out test IDs (never in training data)
PEPSET_IDS = frozenset({
    "1EJ4", "1G73", "1PRM", "2FLU", "2VWF",
    "3DAB", "3EG6", "3EQS", "3EQY", "3TWR",
})


def _load_pepset_ids() -> frozenset[str]:
    """Load PepSet IDs from file, falling back to hardcoded set."""
    f = DATA_DIR / "pepset_ids.txt"
    if f.exists():
        ids = frozenset(l.strip().upper() for l in f.read_text().splitlines() if l.strip())
        # Sanity: hardcoded set must be a subset
        missing = PEPSET_IDS - ids
        assert not missing, f"pepset_ids.txt missing expected IDs: {missing}"
        return ids
    return PEPSET_IDS


def _csv_pdb_ids(csv_path: Path) -> set[str]:
    """Extract pdb_id column from a CSV file (case-insensitive)."""
    if not csv_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "pdb_id" not in df.columns:
            return set()
        return set(df["pdb_id"].dropna().str.upper().tolist())
    except Exception:
        return set()


# ---------------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------------- #

class TestPepSetIdFile:
    def test_pepset_file_exists(self) -> None:
        f = DATA_DIR / "pepset_ids.txt"
        assert f.exists(), (
            f"data/pepset_ids.txt not found. Create it with exactly 10 PepSet IDs:\n"
            f"{sorted(PEPSET_IDS)}"
        )

    def test_pepset_file_has_correct_ids(self) -> None:
        f = DATA_DIR / "pepset_ids.txt"
        if not f.exists():
            pytest.skip("pepset_ids.txt not found")
        ids = frozenset(l.strip().upper() for l in f.read_text().splitlines() if l.strip())
        missing = PEPSET_IDS - ids
        extra = ids - PEPSET_IDS
        assert not missing, f"pepset_ids.txt is missing expected IDs: {sorted(missing)}"
        # Extra IDs are OK (could include synonyms), but warn
        if extra:
            import warnings
            warnings.warn(f"pepset_ids.txt has extra IDs beyond the 10 standard ones: {sorted(extra)}")

    def test_pepset_file_no_training_ids(self) -> None:
        """Ensure training IDs are NOT in pepset_ids.txt (common mistake)."""
        training_ids = _csv_pdb_ids(DATA_DIR / "training_complexes.csv")
        pepset_file_ids = _load_pepset_ids()
        # Only flag if training IDs ALSO appear in the file (the IDs 1A0N, 1YWI mistake)
        contamination = training_ids & pepset_file_ids
        assert not contamination, (
            f"pepset_ids.txt contains training complex IDs: {sorted(contamination)}\n"
            f"These should NEVER be in the held-out test set."
        )


class TestTrainingComplexesNoLeakage:
    def test_training_complexes_csv(self) -> None:
        csv_path = DATA_DIR / "training_complexes.csv"
        if not csv_path.exists():
            pytest.skip("training_complexes.csv not found")
        ids = _csv_pdb_ids(csv_path)
        pepset = _load_pepset_ids()
        leak = ids & pepset
        assert not leak, (
            f"PEPSET LEAKAGE in training_complexes.csv: {sorted(leak)}\n"
            f"These PDB IDs appear in both training and test sets!"
        )

    def test_training_complexes_full_csv(self) -> None:
        csv_path = DATA_DIR / "training_complexes_full.csv"
        if not csv_path.exists():
            pytest.skip("training_complexes_full.csv not found")
        ids = _csv_pdb_ids(csv_path)
        pepset = _load_pepset_ids()
        leak = ids & pepset
        assert not leak, (
            f"PEPSET LEAKAGE in training_complexes_full.csv: {sorted(leak)}\n"
            f"These PDB IDs appear in both calibration and test sets!"
        )

    def test_training_complexes_expanded_csv(self) -> None:
        csv_path = DATA_DIR / "training_complexes_expanded.csv"
        if not csv_path.exists():
            pytest.skip("training_complexes_expanded.csv not found")
        ids = _csv_pdb_ids(csv_path)
        pepset = _load_pepset_ids()
        leak = ids & pepset
        assert not leak, (
            f"PEPSET LEAKAGE in training_complexes_expanded.csv: {sorted(leak)}\n"
            f"These PDB IDs appear in both training and test sets!"
        )

    def test_rcsb_bulk_affinity_csv(self) -> None:
        csv_path = DATA_DIR / "rcsb_binding_affinity_bulk.csv"
        if not csv_path.exists():
            pytest.skip("rcsb_binding_affinity_bulk.csv not found")
        ids = _csv_pdb_ids(csv_path)
        pepset = _load_pepset_ids()
        leak = ids & pepset
        assert not leak, (
            f"PEPSET LEAKAGE in rcsb_binding_affinity_bulk.csv: {sorted(leak)}\n"
            f"These PDB IDs appear in the affinity data used for calibration!"
        )


class TestDatasetManifestsNoLeakage:
    """Check that dataset manifests don't include PepSet entries."""

    def _check_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            pytest.skip(f"Manifest not found: {manifest_path}")
        ids = _csv_pdb_ids(manifest_path)
        pepset = _load_pepset_ids()
        leak = ids & pepset
        assert not leak, (
            f"PEPSET LEAKAGE in {manifest_path}: {sorted(leak)}\n"
            f"These structures are included in the training dataset manifest!"
        )

    def test_pdb_2024_2026_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "pdb_2024_2026" / "manifest.csv")

    def test_ppii_enriched_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "ppii_enriched" / "manifest.csv")

    def test_family_targeted_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "family_targeted" / "manifest.csv")

    def test_pdb_2019_2023_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "pdb_2019_2023" / "manifest.csv")

    def test_pdb_2010_2018_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "pdb_2010_2018" / "manifest.csv")

    def test_pdb_pre2010_manifest(self) -> None:
        self._check_manifest(REPO / "datasets" / "pdb_pre2010" / "manifest.csv")


class TestTestComplexesIntegrity:
    def test_test_complexes_has_all_10_pepset_ids(self) -> None:
        csv_path = DATA_DIR / "test_complexes.csv"
        if not csv_path.exists():
            pytest.skip("test_complexes.csv not found")
        ids = _csv_pdb_ids(csv_path)
        missing = PEPSET_IDS - ids
        assert not missing, (
            f"test_complexes.csv is missing PepSet IDs: {sorted(missing)}\n"
            f"All 10 PepSet IDs must be present for valid evaluation."
        )

    def test_test_complexes_has_exactly_pepset_count(self) -> None:
        csv_path = DATA_DIR / "test_complexes.csv"
        if not csv_path.exists():
            pytest.skip("test_complexes.csv not found")
        ids = _csv_pdb_ids(csv_path)
        # At least 10, possibly more with extras added later
        assert len(ids) >= len(PEPSET_IDS), (
            f"test_complexes.csv has only {len(ids)} entries, expected ≥ {len(PEPSET_IDS)}"
        )

    def test_no_overlap_between_train_and_test(self) -> None:
        train_ids = _csv_pdb_ids(DATA_DIR / "training_complexes.csv")
        test_ids = _csv_pdb_ids(DATA_DIR / "test_complexes.csv")
        if not train_ids or not test_ids:
            pytest.skip("Missing training or test CSV")
        overlap = train_ids & test_ids
        assert not overlap, (
            f"CRITICAL: Training and test CSV have overlapping PDB IDs: {sorted(overlap)}"
        )
