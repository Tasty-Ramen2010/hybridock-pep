"""Tests for hybridock_pep.output.metadata (SAMP-02)."""
from __future__ import annotations

import json
import os
import unittest.mock as mock
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestMetadata:
    """Tests for write_metadata_skeleton(), finalize_metadata(), get_rapidock_commit_sha()."""

    @pytest.fixture()
    def config(self, tmp_path: Path):
        from hybridock_pep.models import DockConfig

        return DockConfig(
            peptide_sequence="ALA",
            receptor_path=FIXTURES_DIR / "receptor_tiny.pdb",
            site_coords=(0.0, 0.0, 0.0),
            box_size=20.0,
            output_dir=tmp_path / "out",
        )

    def _mock_version_funcs(self) -> list:
        """Return a list of patch objects for version-reporting helper functions."""
        return [
            mock.patch("hybridock_pep.output.metadata._get_git_sha", return_value="test-sha"),
            mock.patch(
                "hybridock_pep.output.metadata.get_rapidock_commit_sha",
                return_value="test-rapidock-sha",
            ),
            mock.patch("hybridock_pep.output.metadata._get_vina_version", return_value="1.2.5"),
            mock.patch("hybridock_pep.output.metadata._get_openmm_version", return_value="8.1.0"),
        ]

    # ------------------------------------------------------------------
    # task 4-03-01
    # ------------------------------------------------------------------

    def test_skeleton_status_is_running(self, config, tmp_path: Path) -> None:
        """write_metadata_skeleton → creates file with status == 'running'."""
        from hybridock_pep.output.metadata import write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        assert metadata_path.exists(), "run_metadata.json must be created"
        data = json.loads(metadata_path.read_text())
        assert data["status"] == "running"

    # ------------------------------------------------------------------
    # task 4-03-02
    # ------------------------------------------------------------------

    def test_skeleton_has_required_fields(self, config, tmp_path: Path) -> None:
        """Skeleton JSON must contain all 12 required D-16 fields (not timestamp_end or poses_generated)."""
        from hybridock_pep.output.metadata import write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        data = json.loads(metadata_path.read_text())

        required = {
            "status",
            "timestamp_start",
            "poses_requested",
            "seed",
            "cli_args",
            "git_sha",
            "rapidock_commit_sha",
            "receptor_sha256",
            "peptide_sequence_hash",
            "vina_version",
            "openmm_version",
            "cuda_version",
        }
        assert required.issubset(data.keys()), (
            f"Missing fields: {required - data.keys()}"
        )

        # Finalize-only fields must NOT be present in skeleton
        assert "timestamp_end" not in data, "timestamp_end must not be in skeleton"
        assert "poses_generated" not in data, "poses_generated must not be in skeleton"

    # ------------------------------------------------------------------
    # task 4-03-03  (status)
    # ------------------------------------------------------------------

    def test_finalize_status_is_complete(self, config, tmp_path: Path) -> None:
        """After finalize_metadata(), status must be 'complete'."""
        from hybridock_pep.output.metadata import finalize_metadata, write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        finalize_metadata(metadata_path, poses_generated=95)

        data = json.loads(metadata_path.read_text())
        assert data["status"] == "complete"

    # ------------------------------------------------------------------
    # task 4-03-03 / pitfall 6
    # ------------------------------------------------------------------

    def test_finalize_preserves_clipped_poses(self, tmp_path: Path) -> None:
        """finalize_metadata must not wipe existing clipped_poses list."""
        from hybridock_pep.output.metadata import finalize_metadata

        metadata_path = tmp_path / "run_metadata.json"
        metadata_path.write_text(
            json.dumps({"status": "running", "clipped_poses": [{"pose_idx": 3}]})
        )

        finalize_metadata(metadata_path, poses_generated=99)

        data = json.loads(metadata_path.read_text())
        assert data["clipped_poses"] == [{"pose_idx": 3}], (
            "clipped_poses must be preserved by finalize_metadata"
        )

    # ------------------------------------------------------------------
    # task 4-03-02  (poses_generated)
    # ------------------------------------------------------------------

    def test_finalize_records_poses_generated(self, config, tmp_path: Path) -> None:
        """finalize_metadata must write poses_generated field with the supplied value."""
        from hybridock_pep.output.metadata import finalize_metadata, write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        finalize_metadata(metadata_path, poses_generated=100)

        data = json.loads(metadata_path.read_text())
        assert data["poses_generated"] == 100

    # ------------------------------------------------------------------
    # timestamp_end
    # ------------------------------------------------------------------

    def test_finalize_adds_timestamp_end(self, config, tmp_path: Path) -> None:
        """finalize_metadata must add timestamp_end to the JSON."""
        from hybridock_pep.output.metadata import finalize_metadata, write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        finalize_metadata(metadata_path, poses_generated=100)

        data = json.loads(metadata_path.read_text())
        assert "timestamp_end" in data, "finalize_metadata must add timestamp_end field"

    # ------------------------------------------------------------------
    # task 4-03-04
    # ------------------------------------------------------------------

    def test_commit_sha_from_direct_url(self) -> None:
        """get_rapidock_commit_sha() reads commit_id from direct_url.json distribution metadata."""
        from hybridock_pep.output.metadata import get_rapidock_commit_sha

        direct_url_content = json.dumps({"vcs_info": {"commit_id": "abc123"}})

        mock_file = mock.MagicMock()
        mock_file.name = "direct_url.json"
        mock_file.read_text.return_value = direct_url_content

        mock_dist = mock.MagicMock()
        mock_dist.files = [mock_file]

        with mock.patch(
            "importlib.metadata.distribution", return_value=mock_dist
        ):
            result = get_rapidock_commit_sha()

        assert result == "abc123", f"Expected 'abc123', got {result!r}"

    # ------------------------------------------------------------------
    # atomic write
    # ------------------------------------------------------------------

    def test_atomic_write_uses_tmp_file(self, config, tmp_path: Path) -> None:
        """write_metadata_skeleton must use os.replace with a .tmp staging file."""
        from hybridock_pep.output.metadata import write_metadata_skeleton

        metadata_path = tmp_path / "run_metadata.json"
        replace_calls: list[tuple[str, str]] = []

        real_replace = os.replace

        def spy_replace(src, dst):
            replace_calls.append((str(src), str(dst)))
            return real_replace(src, dst)

        patches = self._mock_version_funcs()
        for p in patches:
            p.start()
        try:
            with mock.patch("os.replace", side_effect=spy_replace):
                write_metadata_skeleton(config, metadata_path)
        finally:
            for p in patches:
                p.stop()

        assert len(replace_calls) == 1, (
            f"os.replace must be called exactly once, got {len(replace_calls)} calls"
        )
        src_arg = replace_calls[0][0]
        assert src_arg.endswith(".tmp"), (
            f"First arg to os.replace must end with .tmp, got {src_arg!r}"
        )
