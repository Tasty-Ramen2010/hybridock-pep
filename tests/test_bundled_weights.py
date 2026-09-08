"""Guards for the RAPiDock checkpoints shipped inside this repository.

The two checkpoints used to be downloaded from Zenodo during install. That was
the only step of a first install that had to reach a host outside GitHub and
PyPI, and `zenodo.org` is blocked outright on some school and institutional
networks — which turned the Colab notebook into a hard failure at the install
cell for exactly the audience it was written for. Both files are 54 MB, well
under GitHub's 100 MB per-file limit, so they now live in `weights/`.

What can silently rot, and is pinned here:

* `weights/` losing a file, or one of them being replaced by something that is
  not the upstream checkpoint (an LFS pointer, a truncated copy, a Git filter
  mangling a binary as text);
* `weights/SHA256SUMS` drifting from the checksums `scripts/install_weights.sh`
  enforces, so one vouches for bytes the other rejects;
* an installer growing its own private download path again;
* the runtime fallback in `rapidock_runner` no longer finding `weights/`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
INSTALL_WEIGHTS_SH = REPO_ROOT / "scripts" / "install_weights.sh"

#: The upstream Zenodo files (record 14193621, CC-BY-4.0). Written out here
#: rather than read from either the script or SHA256SUMS, so this test is an
#: independent third opinion instead of an echo of one of them.
UPSTREAM = {
    "rapidock_local.pt": "d0f1ebe268354624c345f8730e765e1b21c016f946fffb637461236204919693",
    "rapidock_global.pt": "a5dfa8f0b20642e26b276d8fd3e7ac87377b5c5150b15b7afcabf9cd8558e0b5",
}

#: Both files are exactly this size upstream. A cheap first check that fails
#: with a useful number before the (slower) hash does.
UPSTREAM_BYTES = 56_648_773


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestWeightsDirectory:
    @pytest.mark.parametrize("name", sorted(UPSTREAM))
    def test_checkpoint_is_present_and_is_the_upstream_file(self, name: str) -> None:
        path = WEIGHTS_DIR / name
        assert path.is_file(), (
            f"{path} is missing. It is committed to this repository on purpose — "
            "see weights/README.md. A checkout without it falls back to a Zenodo "
            "download, which is what this change exists to avoid."
        )
        size = path.stat().st_size
        assert size == UPSTREAM_BYTES, (
            f"{name} is {size} bytes, expected {UPSTREAM_BYTES}. A few hundred "
            "bytes means a Git LFS pointer or an HTML error page; anything else "
            "means a truncated or modified file."
        )
        assert _sha256(path) == UPSTREAM[name], (
            f"{name} does not match the upstream Zenodo checkpoint. These are "
            "redistributed unmodified under CC-BY-4.0; a fine-tuned checkpoint "
            "must not be committed here under an upstream name."
        )

    def test_sha256sums_file_agrees(self) -> None:
        """weights/SHA256SUMS is what a user runs `sha256sum -c` against."""
        listed = {}
        for line in (WEIGHTS_DIR / "SHA256SUMS").read_text().split("\n"):
            if line.strip():
                digest, name = line.split()
                listed[name.lstrip("*")] = digest
        assert listed == UPSTREAM, f"SHA256SUMS drifted: {listed}"

    def test_installer_enforces_the_same_checksums(self) -> None:
        """scripts/install_weights.sh hard-codes them so a corrupted weights/
        directory cannot vouch for itself. They must not drift from the files."""
        text = INSTALL_WEIGHTS_SH.read_text(encoding="utf-8")
        found = dict(re.findall(r"(rapidock_\w+\.pt)\)\s+echo ([0-9a-f]{64})", text))
        assert found == UPSTREAM, f"install_weights.sh checksums drifted: {found}"


class TestInstallWeightsScript:
    def test_syntax(self) -> None:
        assert subprocess.run(
            ["bash", "-n", str(INSTALL_WEIGHTS_SH)], capture_output=True
        ).returncode == 0

    def test_installs_from_weights_without_touching_the_network(
        self, tmp_path: Path
    ) -> None:
        """The whole point: a clone can install its checkpoints offline.

        `curl` is shadowed with a stub that fails loudly, so any fall-through to
        the Zenodo path shows up as a failed install rather than as a slow but
        passing test on a machine that happens to have network.
        """
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        (fake_bin / "curl").write_text("#!/bin/sh\necho 'network used!' >&2\nexit 7\n")
        (fake_bin / "curl").chmod(0o755)

        model_dir = tmp_path / "model"
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin"}
        result = subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir)],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "network used!" not in result.stderr, "fell through to the Zenodo download"
        for name, digest in UPSTREAM.items():
            assert _sha256(model_dir / name) == digest, f"{name} installed wrong"

    def test_lite_installs_only_the_site_directed_checkpoint(self, tmp_path: Path) -> None:
        """rapidock_global.pt is read by exactly one code path — the exploratory
        pass of `dock --blind`. --lite may skip that one and nothing else."""
        model_dir = tmp_path / "model"
        result = subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir), "--lite"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (model_dir / "rapidock_local.pt").is_file()
        assert not (model_dir / "rapidock_global.pt").exists()

    def test_a_corrupt_destination_is_replaced_not_trusted(self, tmp_path: Path) -> None:
        """An existing-but-wrong checkpoint is worse than a missing one: it
        loads and produces silent garbage instead of failing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "rapidock_local.pt").write_bytes(b"not a checkpoint")
        subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir), "--lite"],
            capture_output=True, text=True, check=True,
        )
        assert _sha256(model_dir / "rapidock_local.pt") == UPSTREAM["rapidock_local.pt"]

    def test_rerun_is_a_no_op(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        first = subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir)],
            capture_output=True, text=True, check=True,
        )
        second = subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir)],
            capture_output=True, text=True, check=True,
        )
        assert "installed from weights/" in first.stdout
        assert "already in place and verified" in second.stdout
        assert "installed from weights/" not in second.stdout

    def test_installed_copy_is_independent_of_the_repository_copy(
        self, tmp_path: Path
    ) -> None:
        """Not a hard link and not a symlink. Sharing the bytes would mean
        anything writing through the model directory silently corrupts the
        committed 54 MB file, which `git status` then reports with no obvious
        cause."""
        model_dir = tmp_path / "model"
        subprocess.run(
            ["bash", str(INSTALL_WEIGHTS_SH), "--model-dir", str(model_dir), "--lite"],
            capture_output=True, check=True,
        )
        installed = model_dir / "rapidock_local.pt"
        source = WEIGHTS_DIR / "rapidock_local.pt"
        assert not installed.is_symlink()
        assert installed.stat().st_ino != source.stat().st_ino


class TestRuntimeFallback:
    """A checkout where neither installer was run must still be able to dock."""

    def test_runner_installs_from_weights_when_the_model_dir_is_empty(
        self, tmp_path: Path
    ) -> None:
        from hybridock_pep.sampling.rapidock_runner import _install_bundled_checkpoint

        model_dir = tmp_path / "empty_model_dir"
        _install_bundled_checkpoint(model_dir, "rapidock_local.pt")
        assert _sha256(model_dir / "rapidock_local.pt") == UPSTREAM["rapidock_local.pt"]

    def test_it_is_silent_when_there_is_nothing_to_copy(self, tmp_path: Path) -> None:
        """A wheel install has no weights/ and no repo root to find one in. The
        caller raises a far better error than this helper could, so it must not
        raise one of its own."""
        from hybridock_pep.sampling.rapidock_runner import _install_bundled_checkpoint

        _install_bundled_checkpoint(tmp_path / "m", "definitely_not_a_checkpoint.pt")
        assert not (tmp_path / "m" / "definitely_not_a_checkpoint.pt").exists()
