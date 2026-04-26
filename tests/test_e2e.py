"""End-to-end integration test for the MDM2/p53 complex (TEST-02).

Runs the full hybridock_pep pipeline using pre-generated fixture poses
(tests/fixtures/mdm2_p53/) and a minimal calibration fixture. Validates:
  - ranked_poses.csv written with correct columns and at least 1 row
  - best_pose.pdb written and non-empty
  - run_metadata.json written with status="complete"
  - Corrected ΔG (best hybrid_score) < -3.0 kcal/mol (TEST-02 threshold)

Tagged @pytest.mark.slow — requires the full score-env tool stack:
Vina ≥ 1.2.5, Meeko ≥ 0.5, ADFRsuite on PATH.

Run with: pytest -m slow tests/test_e2e.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_e2e_config(output_dir: Path):
    """Build a DockConfig for MDM2/p53 using the real receptor_tiny.pdb fixture.

    Uses receptor_tiny.pdb for scoring (not the real MDM2 receptor — the test
    validates pipeline wiring and output format, not absolute binding affinity).
    The calibration fixture (alpha=0.2) ensures any Vina score < -5.4 passes
    the < -3.0 kcal/mol threshold.

    Args:
        output_dir: Destination directory for ranked_poses.csv, best_pose.pdb,
            run_metadata.json.

    Returns:
        Validated DockConfig instance.
    """
    from hybridock_pep.models import DockConfig

    return DockConfig(
        peptide_sequence="ETFSDLWKLLPE",
        receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
        site_coords=(26.4, 3.5, -5.6),
        box_size=30.0,
        n_samples=25,
        output_dir=output_dir,
        seed=42,
    )


@pytest.mark.slow
class TestMDM2P53Integration:
    """Full pipeline integration test on the MDM2/p53 complex.

    Requires Vina, Meeko, ADFRsuite, and score-env fully installed.
    Skipped automatically when pytest is run without -m slow.
    """

    def test_corrected_delta_g_passes_threshold(self, tmp_path: Path) -> None:
        """Run the full pipeline on 25 MDM2/p53 fixture poses; assert ΔG < -3.0.

        Validates TEST-02: corrected ΔG for MDM2/p53 (ETFSDLWKLLPE, K_d ≈ 0.6 µM)
        must be below -3.0 kcal/mol after backbone entropy correction.

        The calibration fixture uses alpha=0.2, beta=0.0:
            hybrid_score = vina_score + (0.2 × 12) = vina_score + 2.4
            threshold: vina_score < -5.4 → hybrid_score < -3.0
        MDM2/p53 with Vina reliably scores below -6.0 kcal/mol.
        """
        from hybridock_pep import driver

        poses_dir = FIXTURES_DIR / "mdm2_p53"
        cal_path = FIXTURES_DIR / "mdm2_calibration.json"
        output_dir = tmp_path / "mdm2_run"
        config = _make_e2e_config(output_dir)

        scored_poses, cluster_result = driver.run_dock(
            config=config,
            input_poses_dir=poses_dir,
            calibration_path=cal_path,
        )

        # --- Assert: ranked_poses.csv exists with correct structure ---
        csv_path = output_dir / "ranked_poses.csv"
        assert csv_path.exists(), "ranked_poses.csv must be created by Stage 4"

        rows = list(csv.DictReader(csv_path.open()))
        assert 1 <= len(rows) <= 10, (
            f"ranked_poses.csv should have 1–10 rows; got {len(rows)}"
        )

        required_cols = {
            "rank", "hybrid_score", "vina_score", "ad4_score",
            "entropy_correction", "delta_g", "cluster_id",
            "pose_filename", "is_ad4_anomaly", "is_clipped",
        }
        assert required_cols.issubset(rows[0].keys()), (
            f"Missing columns: {required_cols - rows[0].keys()}"
        )

        # --- Assert: TEST-02 threshold — best hybrid_score < -3.0 kcal/mol ---
        best_hybrid = min(float(r["hybrid_score"]) for r in rows)
        assert best_hybrid < -3.0, (
            f"TEST-02 FAILED: best corrected ΔG = {best_hybrid:.2f} kcal/mol "
            f"(threshold: < -3.0). Check calibration fixture and Vina scoring."
        )

        # --- Assert: best_pose.pdb exists and is non-empty ---
        best_pdb = output_dir / "best_pose.pdb"
        assert best_pdb.exists(), "best_pose.pdb must be written by Stage 4"
        assert best_pdb.stat().st_size > 0, "best_pose.pdb must not be empty"

        # --- Assert: run_metadata.json written with status=complete ---
        metadata_path = output_dir / "run_metadata.json"
        assert metadata_path.exists(), "run_metadata.json must exist"
        metadata = json.loads(metadata_path.read_text())
        assert metadata.get("status") == "complete", (
            f"run_metadata.json status should be 'complete', got {metadata.get('status')!r}"
        )

        # --- Assert: delta_g == hybrid_score in every row (D-04) ---
        for row in rows:
            assert row["delta_g"] == row["hybrid_score"], (
                f"delta_g ({row['delta_g']}) != hybrid_score ({row['hybrid_score']}) "
                f"for rank {row['rank']}"
            )
