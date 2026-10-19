#!/usr/bin/env python3
"""Verify (or regenerate) the checksums of the artifacts this repository ships.

``ARTIFACTS.sha256`` lists the files whose exact content the reported results depend on:
the Stage 1 checkpoints, the trained scoring models, the calibration files, and the small
benchmark tables that RESULTS.md reproduces from. A mismatch means a file changed after the
numbers were produced -- or a checkout corrupted it.

    python scripts/verify_artifacts.py             # verify; exit 0 only if every file matches
    python scripts/verify_artifacts.py --write     # re-hash every file already listed
    python scripts/verify_artifacts.py --add PATH  # list a new file (and hash it)

Exit codes: 0 all match, 1 mismatch or missing file, 2 unreadable or malformed manifest.

Text files (.csv .json .jsonl .txt .md) are hashed with CRLF folded to LF, so a Windows
checkout with ``core.autocrlf=true`` verifies the same as a Linux one. Binary files
(.pt .joblib .xls ...) are hashed byte for byte. Standard library only, so it runs
before any dependency is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ARTIFACTS.sha256"
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".md"}
_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
_HEADER = (
    "# SHA-256 of the artifacts HybriDock-Pep ships. Format: <hash><two spaces><path from repo root>.\n"
    "# Verify:     python scripts/verify_artifacts.py\n"
    "# Regenerate: python scripts/verify_artifacts.py --write   (only after a deliberate change)\n"
    "# Text files are hashed with CRLF folded to LF; see scripts/verify_artifacts.py.\n"
)


def file_digest(path: Path) -> str:
    """SHA-256 of ``path``, folding CRLF to LF for text files. Streams, so a 60 MB checkpoint is fine."""
    digest = hashlib.sha256()
    fold = path.suffix.lower() in TEXT_SUFFIXES
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            if fold:
                chunk = carry + chunk
                # A CR at the very end of a chunk may belong to a CRLF split across chunks.
                carry = b"\r" if chunk.endswith(b"\r") else b""
                chunk = (chunk[:-1] if carry else chunk).replace(b"\r\n", b"\n")
            digest.update(chunk)
    digest.update(carry)
    return digest.hexdigest()


def read_manifest(manifest: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Return ``(entries, problems)``. Blank lines and ``#`` comments are skipped; bad lines are reported."""
    entries: list[tuple[str, str]] = []
    problems: list[str] = []
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(raw.rstrip("\n"))
        if match is None:
            problems.append(f"{manifest.name}:{number}: not '<64 hex>  <path>': {raw[:60]!r}")
            continue
        entries.append((match.group(1), match.group(2)))
    return entries, problems


def write_manifest(manifest: Path, paths: list[str]) -> None:
    lines = [f"{file_digest(ROOT / name)}  {name}\n" for name in paths]
    manifest.write_text(_HEADER + "".join(lines), encoding="utf-8")


def verify(manifest: Path) -> int:
    try:
        entries, problems = read_manifest(manifest)
    except OSError as error:
        print(f"Cannot read {manifest}: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"MALFORMED  {problem}", file=sys.stderr)
    if problems or not entries:
        if not entries:
            print(f"No entries in {manifest}", file=sys.stderr)
        return 2
    missing = [name for _, name in entries if not (ROOT / name).is_file()]
    changed = [name for expected, name in entries
               if name not in missing and file_digest(ROOT / name) != expected]
    for name in missing:
        print(f"MISSING    {name}", file=sys.stderr)
    for name in changed:
        print(f"CHANGED    {name}", file=sys.stderr)
    if missing or changed:
        print(f"{len(missing)} missing, {len(changed)} changed of {len(entries)} artifacts.", file=sys.stderr)
        return 1
    print(f"Verified {len(entries)} artifact checksums.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest to use (default: ARTIFACTS.sha256)")
    parser.add_argument("--write", action="store_true", help="re-hash every file already listed")
    parser.add_argument("--add", nargs="+", metavar="PATH", help="list new files (paths from the repo root) and hash them")
    args = parser.parse_args(argv)
    if not (args.write or args.add):
        return verify(args.manifest)
    entries, problems = read_manifest(args.manifest) if args.manifest.exists() else ([], [])
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    paths = [name for _, name in entries]
    for name in args.add or []:
        if not (ROOT / name).is_file():
            print(f"No such file: {name}", file=sys.stderr)
            return 2
        if name not in paths:
            paths.append(name)
    absent = [name for name in paths if not (ROOT / name).is_file()]
    if absent:
        print("Listed but missing, refusing to write: " + ", ".join(absent), file=sys.stderr)
        return 2
    write_manifest(args.manifest, paths)
    print(f"Wrote {len(paths)} checksums to {args.manifest.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
