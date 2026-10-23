"""Public mode: what the web server does differently when strangers can reach it.

``hybridock-pep serve`` is a local tool. It trusts whoever talks to it: a request may name any file on the computer as the
receptor, any folder as the output, and may cancel anyone's run. That is fine on 127.0.0.1 and not fine on the internet.

``serve --public`` puts a policy in front of the same server. Everything a visitor sends is rebuilt from a short allow-list;
paths never cross the wire (the page sends ``example:<name>`` or ``upload:<name>`` and the server maps it to a file it
placed itself); runs are capped so one visitor cannot hold the computer for an hour; and nothing about this computer
(home folder, user name, absolute paths) is sent back. See ``docs/SHARING.md``.

Nothing here runs unless ``--public`` is given.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: The only form fields a visitor may set. Everything else (output folder, input-poses folder, calibration file, MM-GBSA
#: and ultra tiers, ...) is ignored and left at its default.
ALLOWED_KEYS = frozenset({
    "mode", "peptide", "receptor", "site", "box", "n_samples", "seed", "blind",
    "peptide_pdb", "offtarget_receptor", "offtarget_site", "offtarget_box",
})
_PATH_KEYS = ("receptor", "peptide_pdb", "offtarget_receptor")

_PEPTIDE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.pdb$")
_NUMBER = re.compile(r"^-?\d{1,4}(\.\d{1,3})?$")


@dataclass
class PublicPolicy:
    """The rules for a public server. The defaults suit a laptop that several people may use in turn."""

    allowed_origins: tuple[str, ...] = ()
    max_samples: int = 25            # the most a visitor may ask for; visitors pick one of allowed_samples
    allowed_samples: tuple[int, ...] = (5, 10, 25)   # the three sizes the page offers: every run finishes in minutes
    max_compare_samples: int = 25    # a comparison is two dockings
    max_peptide: int = 25
    max_upload_bytes: int = 6 * 1024 * 1024
    keep_hours: float = 24.0         # finished runs and uploads older than this are deleted
    runs_per_window: int = 8         # per visitor address, per window
    uploads_per_window: int = 40
    window_seconds: float = 600.0
    runs_dir: Path = field(default_factory=lambda: Path("runs/public"))
    _hits: dict[tuple[str, str], deque[float]] = field(default_factory=lambda: defaultdict(deque), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ who may call
    def origin_ok(self, origin: str | None, host: str | None) -> bool:
        """A browser call is welcome from the hosted site, or from this server's own page."""
        if not origin:
            return True  # not a browser page (curl, a health check): nothing to protect from
        if origin in self.allowed_origins:
            return True
        try:
            return bool(host) and urlparse(origin).netloc == host
        except ValueError:
            return False

    def allow_origin_header(self, origin: str | None, host: str | None) -> str | None:
        return origin if origin and origin != "null" and self.origin_ok(origin, host) else None

    def too_many(self, who: str, what: str, limit: int) -> bool:
        """True when ``who`` has already done ``what`` ``limit`` times in the last window (and counts this call otherwise)."""
        now = time.time()
        with self._lock:
            q = self._hits[(who, what)]
            while q and now - q[0] > self.window_seconds:
                q.popleft()
            if len(q) >= limit:
                return True
            q.append(now)
            return False

    # ------------------------------------------------------------------ files
    @property
    def uploads_dir(self) -> Path:
        return self.runs_dir / "uploads"

    def save_upload(self, name: str, content: str) -> str:
        """Store a visitor's PDB under a name this server chose, and return the token the page sends back."""
        base = Path(str(name or "")).name
        if not _SAFE_NAME.match(base):
            raise ValueError("Use a .pdb file whose name is letters, numbers, dots, dashes or underscores.")
        raw = content.encode("utf-8", "replace")
        if len(raw) > self.max_upload_bytes:
            raise ValueError(f"That file is too big for the shared computer (limit {self.max_upload_bytes // (1024 * 1024)} MB).")
        if "ATOM" not in content:
            raise ValueError("That PDB has no ATOM records. Is it the right file?")
        digest = hashlib.sha256(raw).hexdigest()[:12]
        stored = f"{digest}_{base}"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self.uploads_dir / stored
        if not target.exists():
            tmp = target.with_name(f".{stored}.tmp")
            tmp.write_bytes(raw)
            tmp.replace(target)
        return f"upload:{stored}"

    def resolve_path(self, token: str, examples: dict[str, Path]) -> str:
        """``example:pdbs/x.pdb`` or ``upload:<name>`` to a real file. Anything else is refused."""
        kind, _, rest = str(token or "").partition(":")
        if kind == "example" and rest in examples:
            return str(examples[rest])
        if kind == "upload" and re.match(r"^[0-9a-f]{12}_", rest) and _SAFE_NAME.match(rest[13:] or ""):
            path = self.uploads_dir / rest
            if path.is_file():
                return str(path.resolve())
            raise ValueError("That uploaded file is no longer on the shared computer. Add it again.")
        raise ValueError("Choose a built-in protein or upload a .pdb file. Files on the computer cannot be named here.")

    # ------------------------------------------------------------------ requests
    def clean_values(self, raw: dict[str, Any], mode: str, examples: dict[str, Path], run_dir: Path) -> dict[str, str]:
        """Rebuild the form values from the allow-list, checked and capped. Raises ValueError with a plain message."""
        values = {k: ("" if v is None else str(v)) for k, v in raw.items() if k in ALLOWED_KEYS}
        peptide = values.get("peptide", "").strip().upper()
        if mode != "crystal" or peptide:
            if not (3 <= len(peptide) <= self.max_peptide) or not _PEPTIDE.match(peptide):
                raise ValueError(f"Type a peptide of 3 to {self.max_peptide} amino acids (letters only).")
        values["peptide"] = peptide
        if values.get("blind", "n").strip().lower().startswith("y"):
            raise ValueError("Searching the whole protein is not offered on the shared computer. Mark the binding site instead.")
        values["blind"] = "n"
        cap = self.max_compare_samples if mode == "selectivity" else self.max_samples
        samples = values.get("n_samples", "").strip()
        if mode != "crystal":
            allowed = [n for n in self.allowed_samples if n <= cap]
            if not samples.isdigit() or int(samples) not in allowed:
                raise ValueError(f"Choose {', '.join(str(n) for n in allowed[:-1])} or {allowed[-1]} poses.")
        for key in ("site", "offtarget_site"):
            if values.get(key, "").strip():
                parts = values[key].split()
                if len(parts) != 3 or not all(_NUMBER.match(p) for p in parts):
                    raise ValueError("The binding site must be three numbers.")
        for key in ("box", "offtarget_box"):
            if values.get(key, "").strip() and not _NUMBER.match(values[key].strip()):
                raise ValueError("The box size must be a number.")
        if values.get("seed", "").strip() and not values["seed"].strip().isdigit():
            raise ValueError("The seed must be a whole number.")
        for key in _PATH_KEYS:
            if values.get(key, "").strip():
                values[key] = self.resolve_path(values[key], examples)
        values["output_dir"] = str(run_dir)
        return values

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    # ------------------------------------------------------------------ housekeeping
    def prune(self, busy_dir: Path | None = None) -> int:
        """Delete finished runs and uploads older than ``keep_hours``. Returns how many things were removed."""
        removed = 0
        cutoff = time.time() - self.keep_hours * 3600
        if not self.runs_dir.is_dir():
            return 0
        for child in self.runs_dir.iterdir():
            try:
                if busy_dir is not None and child.resolve() == busy_dir.resolve():
                    continue
                if child.name == "uploads":
                    for f in child.iterdir():
                        if f.stat().st_mtime < cutoff:
                            f.unlink(missing_ok=True)
                            removed += 1
                elif child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------ what leaves
    def describe(self) -> dict[str, Any]:
        """Sent to the page inside /api/env so it can explain the limits instead of letting a run fail."""
        return {
            "max_samples": self.max_samples, "max_compare_samples": self.max_compare_samples, "allowed_samples": list(self.allowed_samples),
            "max_peptide": self.max_peptide, "blind": False, "custom_paths": False,
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
        }


def scrub(text: str, home: str, cwd: str) -> str:
    """Remove this computer's folder names from text that is sent to a visitor."""
    for needle, repl in ((cwd, "."), (home, "~")):
        if needle and needle not in ("/", "."):
            text = text.replace(needle, repl)
    return text
