"""scripts/verify_artifacts.py: the shipped manifest is current, and the verifier notices damage."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_artifacts.py"


@pytest.fixture(scope="module")
def va():
    spec = importlib.util.spec_from_file_location("verify_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(va, tmp_path, monkeypatch):
    """A throwaway repo root holding one text and one binary artifact plus a manifest listing both."""
    monkeypatch.setattr(va, "ROOT", tmp_path)
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "model.joblib").write_bytes(bytes(range(256)))
    manifest = tmp_path / "ARTIFACTS.sha256"
    assert va.main(["--manifest", str(manifest), "--add", "table.csv", "model.joblib"]) == 0
    return tmp_path, manifest


def test_shipped_manifest_matches_the_repository(va):
    """If this fails, an artifact changed after its checksum was recorded (or the checkout is damaged)."""
    assert va.main([]) == 0


def test_shipped_manifest_has_no_malformed_or_duplicate_entries(va):
    entries, problems = va.read_manifest(va.MANIFEST)
    assert not problems
    assert len(entries) == len({name for _, name in entries}), "an artifact is listed twice"


def test_changed_file_is_reported(va, sandbox, capsys):
    root, manifest = sandbox
    (root / "model.joblib").write_bytes(b"tampered")
    assert va.main(["--manifest", str(manifest)]) == 1
    assert "CHANGED    model.joblib" in capsys.readouterr().err


def test_missing_file_is_reported(va, sandbox, capsys):
    root, manifest = sandbox
    (root / "table.csv").unlink()
    assert va.main(["--manifest", str(manifest)]) == 1
    assert "MISSING    table.csv" in capsys.readouterr().err


def test_malformed_manifest_line_is_an_error_not_a_crash(va, sandbox, capsys):
    _, manifest = sandbox
    manifest.write_text(manifest.read_text() + "not a checksum line\n")
    assert va.main(["--manifest", str(manifest)]) == 2
    assert "MALFORMED" in capsys.readouterr().err


def test_empty_manifest_is_an_error(va, tmp_path, monkeypatch):
    monkeypatch.setattr(va, "ROOT", tmp_path)
    manifest = tmp_path / "ARTIFACTS.sha256"
    manifest.write_text("# only a comment\n")
    assert va.main(["--manifest", str(manifest)]) == 2


def test_text_files_hash_the_same_with_windows_line_endings(va, tmp_path):
    lf, crlf = tmp_path / "lf.csv", tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert va.file_digest(lf) == va.file_digest(crlf)


def test_crlf_split_across_a_read_chunk_is_still_folded(va, tmp_path):
    """The CR is the last byte of one 1 MiB read and the LF is the first of the next."""
    path = tmp_path / "boundary.csv"
    body = b"x" * ((1 << 20) - 1) + b"\r\n" + b"tail\r\n"
    path.write_bytes(body)
    assert va.file_digest(path) == hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest()


def test_binary_files_are_hashed_byte_for_byte(va, tmp_path):
    path = tmp_path / "weights.pt"
    data = b"\x00\r\n\xff\r\n"
    path.write_bytes(data)
    assert va.file_digest(path) == hashlib.sha256(data).hexdigest()


def test_write_refuses_to_hash_a_file_that_is_gone(va, sandbox):
    root, manifest = sandbox
    (root / "table.csv").unlink()
    assert va.main(["--manifest", str(manifest), "--write"]) == 2
