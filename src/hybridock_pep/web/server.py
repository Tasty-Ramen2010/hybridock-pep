"""Stdlib HTTP server behind ``hybridock-pep serve``.

Deliberately dependency-free: ``http.server`` + threads, no FastAPI/Flask, so the
web UI installs with the package and runs anywhere the CLI already runs. It binds
to loopback only — this is a local tool, not a public service.

Every run is a ``subprocess`` of the CLI, built by the *same* functions the
terminal UI uses (:mod:`hybridock_pep.ui.tui`). That is the point of the design:
the two UIs cannot drift, and a run started in the browser is reproducible by
copy-pasting the command the browser shows you.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import subprocess
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from hybridock_pep._paths import data_file
from hybridock_pep.ui import tui

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

#: Where uploaded receptors and browser-started runs land. Kept out of the repo
#: root so a `serve` session never writes into someone's working tree by surprise.
SESSION_ROOT = Path("runs/studio")

MAX_BODY_BYTES = 64 * 1024 * 1024  # a big receptor PDB is a few MB; this is slack
LOG_RING = 4000  # lines kept per job for the live console

_AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}


# --------------------------------------------------------------------------- #
#  Examples — the "I just want to press go" path
# --------------------------------------------------------------------------- #

#: Curated demo systems. Sites/boxes are the validated ones from CLAUDE.md and the
#: test fixtures, so a first-time user gets a run that actually works rather than a
#: box in the wrong place. `note` is shown in the UI, caveats included.
EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "mdm2",
        "name": "MDM2 + p53 peptide",
        "peptide": "ETFSDLWKLLPE",
        "receptor": "pdbs/1YCR_mdm2.pdb",
        "site": [25.20, -25.61, -7.97],
        "box": 30,
        "blurb": "The integration-test baseline. Known binder, Kd about 0.6 µM.",
        "note": "If this returns weaker than −3 kcal/mol, something in the pipeline is broken.",
        "expect": "around −9 kcal/mol",
    },
    {
        "id": "pfldh",
        "name": "PfLDH + malaria diagnostic peptide",
        "peptide": "LISDAELEAIFEADC",
        "receptor": "pdbs/1T2D_receptor.pdb",
        "site": [24.84, 22.73, 41.69],
        "box": 30,
        "blurb": "Our own iGEM target: the peptide we want binding malaria LDH.",
        "note": "Charged 15-mer, so the ΔG carries the charged-floor caveat.",
        "expect": "around −11 kcal/mol",
    },
    {
        "id": "hldh",
        "name": "human LDH (the off-target)",
        "peptide": "LISDAELEAIFEADC",
        "receptor": "pdbs/1I0Z.pdb",
        "site": [24.84, 22.73, 41.69],
        "box": 30,
        "blurb": "The counter-target. Pair it with PfLDH in Compare to get ΔΔG.",
        "note": "Selectivity is a direction here, not a measurement — the gap is inside the noise.",
        "expect": "around −10 kcal/mol",
    },
]


def _example_receptor(rel: str) -> Path:
    """Resolve a bundled example receptor to an absolute path."""
    return data_file(rel)


def available_examples() -> list[dict[str, Any]]:
    """Return the examples whose receptor PDB is actually present on this install."""
    out = []
    for ex in EXAMPLES:
        path = _example_receptor(ex["receptor"])
        if path.exists():
            item = dict(ex)
            item["receptor_path"] = str(path)
            out.append(item)
    return out


# --------------------------------------------------------------------------- #
#  Small helpers the UI leans on
# --------------------------------------------------------------------------- #

def peptide_stats(seq: str) -> dict[str, Any]:
    """Describe a peptide in the terms the results page will need.

    Args:
        seq: One-letter sequence, any case.

    Returns:
        Length, net charge at neutral pH, and the flags that decide which caveats
        the UI shows (charged peptides carry the charged-floor warning; a
        C-terminal cysteine is the ref2015 relax issue from CLAUDE.md §2.5).
    """
    up = (seq or "").strip().upper()
    net = sum(_AA_CHARGE.get(c, 0) for c in up)
    return {
        "length": len(up),
        "net_charge": round(net, 1),
        "charged": abs(net) >= 2,
        "has_cys": "C" in up,
        "ends_in_cys": up.endswith("C"),
        "band": _length_band(len(up)),
    }


def _length_band(n: int) -> str:
    if n <= 8:
        return "short"
    if n <= 12:
        return "medium"
    if n <= 16:
        return "long"
    return "very long"


def estimate_seconds(values: dict[str, str], mode: str) -> int:
    """Rough wall-clock estimate shown on the run button.

    Measured anchors: RAPiDock is about 0.6 s/pose on the 5070, Stage 2 scoring is
    2.8 s/pose end-to-end. MM-GBSA and ultra are per-pose on top of that. It is an
    estimate and the UI says so — nobody should be surprised by a 3× miss on CPU.
    """
    if mode == "crystal":
        return 10
    n = _int_or(values.get("n_samples"), 100)
    sampling: float
    if (values.get("input_poses") or "").strip():
        sampling = 0.0
    elif _truthy(values.get("blind")) and not (values.get("site") or "").strip():
        n_search = _int_or(values.get("n_pocket_search"), 300)
        n_pock = _int_or(values.get("n_pockets"), 3)
        n_per = _int_or(values.get("n_per_pocket"), 150)
        sampling = 0.6 * (n_search + n_pock * n_per)
        n = n_pock * n_per
    else:
        sampling = 0.6 * n
    total = sampling + 2.8 * n
    topk = _int_or(values.get("refine_topk"), 0)
    if topk:
        total += 45 * topk  # MM-GBSA, GPU; CPU-only is roughly 30-60 s/pose on top
    ultra = _int_or(values.get("ultra"), 0)
    if ultra:
        total += 60 * max(1, ultra // 8)
    if mode == "selectivity":
        total *= 2
    return int(total)


def _int_or(raw: Any, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in ("y", "yes", "true", "1", "on")


def serialize_fields() -> list[dict[str, Any]]:
    """Expose the TUI's form model to the browser.

    The web form is *generated* from :data:`hybridock_pep.ui.tui.FIELDS` rather
    than hand-written, so a new CLI flag shows up in both UIs the moment it is
    added to that list.
    """
    out = []
    for f in tui.FIELDS:
        out.append(
            {
                "key": f.key,
                "label": f.label,
                "default": f.default,
                "help": f.help,
                "is_path": bool(getattr(f, "is_path", False)),
                "is_dir": bool(getattr(f, "is_dir", False)),
                "optional": bool(getattr(f, "optional", False)),
                "kind": _field_kind(f),
            }
        )
    return out


def _field_kind(field: Any) -> str:
    """Classify a form field so the browser can pick a sensible control."""
    key = field.key
    if key in ("blind", "ultra_charged", "no_minimize", "ensemble", "free_entropy",
               "mmgbsa_ie", "mmgbsa_3traj", "mmgbsa_cpu_only"):
        return "toggle"
    if key == "mode":
        return "mode"
    if key in ("n_samples", "refine_topk", "ultra", "box", "seed", "n_pocket_search",
               "n_pockets", "n_per_pocket", "long_checkpoint_threshold"):
        return "number"
    if key == "mmgbsa_dielectric":
        return "decimal"
    if getattr(field, "is_dir", False):
        return "dir"
    if getattr(field, "is_path", False):
        return "file"
    return "text"


def check_environment() -> dict[str, Any]:
    """Report what this machine can actually do, for the status lamp.

    Kept cheap (path lookups plus one short ``nvidia-smi``) because the UI polls
    it on load and after a failed run.
    """
    import shutil as _shutil

    def _which(name: str) -> str | None:
        return tui._resolve_exe(name) or _shutil.which(name)

    checks: dict[str, Any] = {}
    checks["cli"] = {"ok": bool(_which("hybridock-pep")),
                     "detail": "hybridock-pep on PATH",
                     "fix": "conda activate score-env"}
    checks["vina"] = {"ok": bool(_which("vina")), "detail": "AutoDock Vina",
                      "fix": "conda install -c conda-forge vina"}
    checks["receptor_prep"] = {"ok": bool(_which("prepare_receptor")),
                               "detail": "ADFRsuite prepare_receptor",
                               "fix": "see INSTALL.md (licensed download)"}

    gpu = False
    gpu_name = ""
    smi = _which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if Path(smi).exists():
        try:
            res = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=6, check=False)
            gpu = res.returncode == 0 and bool(res.stdout.strip())
            gpu_name = res.stdout.strip().splitlines()[0] if gpu else ""
        except (OSError, subprocess.SubprocessError):
            gpu = False
    checks["gpu"] = {"ok": gpu, "detail": gpu_name or "no CUDA GPU detected",
                     "fix": "CPU works, just slower — or use Score for an existing pose"}

    try:
        import openmm  # noqa: F401
        checks["openmm"] = {"ok": True, "detail": "OpenMM available", "fix": ""}
    except ImportError:
        checks["openmm"] = {"ok": False, "detail": "OpenMM missing",
                            "fix": "only needed for MM-GBSA refinement"}

    weights = data_file("affinity_ai_nofix.joblib")
    checks["scorer"] = {"ok": weights.exists(), "detail": "affinity model weights",
                        "fix": "bash scripts/install_weights.sh"}

    essential = ("cli", "scorer")
    checks_ok = all(checks[k]["ok"] for k in essential)
    return {"ready": checks_ok, "checks": checks, "examples": len(available_examples())}


# --------------------------------------------------------------------------- #
#  Jobs
# --------------------------------------------------------------------------- #

class Job:
    """One pipeline run started from the browser."""

    def __init__(self, job_id: str, cmd: list[str], mode: str, values: dict[str, str],
                 output_dir: Path) -> None:
        self.id = job_id
        self.cmd = cmd
        self.mode = mode
        self.values = values
        self.output_dir = output_dir
        self.state = "queued"  # queued | running | done | failed | cancelled
        self.lines: list[str] = []
        self.progress = tui.PipelineProgress()
        self.proc: subprocess.Popen[str] | None = None
        self.started = time.time()
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line.rstrip("\n"))
            if len(self.lines) > LOG_RING:
                del self.lines[: len(self.lines) - LOG_RING]
        self.progress.feed(line)

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            total_lines = len(self.lines)
            new = self.lines[since:] if since < total_lines else []
        return {
            "id": self.id,
            "mode": self.mode,
            "state": self.state,
            "returncode": self.returncode,
            "error": self.error,
            "elapsed": int((self.finished_at or time.time()) - self.started),
            "stage": self.progress.stage,
            "stage_label": self.progress.label(),
            "fraction": round(self.progress.fraction(), 4),
            "counter": self.progress.counter(),
            "lines": new,
            "line_count": total_lines,
            "output_dir": str(self.output_dir),
            "command": " ".join(self.cmd),
        }


class JobManager:
    """Runs one job at a time; the GPU cannot sensibly share a dock run anyway."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._current: str | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        with self._lock:
            cur = self.jobs.get(self._current or "")
            return bool(cur and cur.state in ("queued", "running"))

    def start(self, cmd: list[str], mode: str, values: dict[str, str],
              output_dir: Path) -> Job:
        """Spawn the CLI and stream its output into a new :class:`Job`.

        Raises:
            RuntimeError: If a run is already in flight, or the CLI is not on PATH.
        """
        if self.busy:
            raise RuntimeError("A run is already going. Wait for it, or stop it first.")
        exe = tui._resolve_exe(cmd[0])
        if exe is None:
            raise RuntimeError(
                f"'{cmd[0]}' is not on PATH. Start the server from the score-env "
                "conda environment (conda activate score-env)."
            )
        job = Job(uuid.uuid4().hex[:12], cmd, mode, values, output_dir)
        with self._lock:
            self.jobs[job.id] = job
            self._current = job.id
        thread = threading.Thread(target=self._run, args=(job, exe), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, exe: str) -> None:
        job.state = "running"
        job.append(f"$ {' '.join(job.cmd)}")
        try:
            job.proc = subprocess.Popen(
                [exe, *job.cmd[1:]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Its own process group, which terminate_process_tree() requires:
                # it stops the run by signalling the GROUP (the dock spawns the
                # rapidock env's python as a child that would otherwise survive and
                # keep the GPU). Without this the group is the *server's* own, and
                # pressing Stop SIGTERMs the studio — and the shell it runs in.
                start_new_session=True,
            )
        except OSError as exc:
            job.state = "failed"
            job.error = f"Could not start the pipeline: {exc}"
            job.finished_at = time.time()
            return

        assert job.proc.stdout is not None
        for line in job.proc.stdout:
            job.append(line)
        job.returncode = job.proc.wait()
        job.finished_at = time.time()
        if job.state == "cancelled":
            return
        if job.returncode == 0:
            job.state = "done"
            job.progress.finished = True
        else:
            job.state = "failed"
            job.error = _explain_failure(job)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.state not in ("queued", "running"):
            return False
        job.state = "cancelled"
        job.error = "Stopped."
        if job.proc is not None:
            tui.terminate_process_tree(job.proc)
        return True


def _explain_failure(job: Job) -> str:
    """Turn a non-zero exit into something a non-programmer can act on.

    Scans the tail of the log for the failures we actually hit in practice; falls
    back to the last real line rather than a bare exit code.
    """
    tail = "\n".join(job.lines[-60:])
    low = tail.lower()
    if "cuda" in low and ("out of memory" in low or "oom" in low):
        return ("The GPU ran out of memory. Try fewer poses (Quick), or close other "
                "GPU programs and run it again.")
    if "no such file" in low or "does not exist" in low:
        return "A file in the form was not found. Check the receptor path and try again."
    if "not on path" in low or "command not found" in low:
        return "Part of the toolchain is missing — check the status lamp at the top."
    if "conda" in low and "not found" in low:
        return "The rapidock conda environment was not found. See INSTALL.md."
    for line in reversed(job.lines):
        text = line.strip()
        if text and not text.startswith("$") and len(text) > 12:
            return text[:400]
    return f"The run stopped with exit code {job.returncode}."


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #

def read_results(job: Job) -> dict[str, Any]:
    """Collect whatever the run produced into one JSON payload for the UI."""
    out_dir = job.output_dir
    payload: dict[str, Any] = {"output_dir": str(out_dir), "mode": job.mode, "poses": [],
                               "files": [], "headline": None, "selectivity": None}

    if job.mode == "crystal":
        payload["headline"] = _crystal_headline(job)
        return payload

    csv_path = out_dir / "ranked_poses.csv"
    if csv_path.exists():
        rows = _read_csv(csv_path)
        payload["poses"] = rows[:100]
        if rows:
            best = rows[0]
            dg = _first_float(best, ["pooled_affinity_dg", "delta_g", "mmgbsa_dg", "hybrid_score"])
            payload["headline"] = {
                "delta_g": dg,
                "kd": _kd_from_dg(dg),
                "pose_file": best.get("pose_filename"),
                "charged_confidence": best.get("charged_confidence"),
                "n_poses": len(rows),
                "n_clusters": len({r.get("cluster_id") for r in rows if r.get("cluster_id")}),
            }
        payload["cloud"] = _pose_cloud(out_dir, rows)

    if job.mode == "selectivity":
        payload["selectivity"] = _selectivity_summary(job)

    for name in ("ranked_poses.csv", "cluster_summary.csv", "best_pose.pdb",
                 "convergence_plot.png", "silhouette_plot.png", "dendrogram.png",
                 "run_metadata.json"):
        if (out_dir / name).exists():
            payload["files"].append(name)
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        raw = (row.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _kd_from_dg(dg: float | None) -> str | None:
    """ΔG (kcal/mol) → a human-scaled Kd string. RT at 298 K."""
    if dg is None:
        return None
    import math

    kd_molar = math.exp(dg / 0.5924)
    for label, scale in (("fM", 1e-15), ("pM", 1e-12), ("nM", 1e-9),
                         ("µM", 1e-6), ("mM", 1e-3), ("M", 1.0)):
        if kd_molar < scale * 1000:
            return f"{kd_molar / scale:.3g} {label}"
    return f"{kd_molar:.3g} M"


def _pose_cloud(out_dir: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Project each pose's Cα centroid to 2D so the stage can draw the real cloud.

    Not decoration: these are the actual centroids of the poses that were scored,
    projected on their own first two principal axes, coloured by cluster. An empty
    list just means the pose PDBs were cleaned up.
    """
    import numpy as np

    pts: list[list[float]] = []
    keep: list[dict[str, str]] = []
    for row in rows[:100]:
        name = row.get("pose_filename") or ""
        pose = _find_pose(out_dir, name)
        if pose is None:
            continue
        coords = _ca_centroid(pose)
        if coords is None:
            continue
        pts.append(coords)
        keep.append(row)
    if len(pts) < 3:
        return []
    arr = np.asarray(pts, dtype=float)
    arr -= arr.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(arr, full_matrices=False)
        proj = arr @ vt[:2].T
    except np.linalg.LinAlgError:
        proj = arr[:, :2]
    span = float(np.abs(proj).max()) or 1.0
    cloud = []
    for (x, y), row in zip(proj / span, keep, strict=False):
        cloud.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "cluster": _int_or(row.get("cluster_id"), 0),
            "rank": _int_or(row.get("rank"), 0),
            "dg": _first_float(row, ["pooled_affinity_dg", "delta_g"]),
            "pose": row.get("pose_filename"),
        })
    return cloud


def _find_pose(out_dir: Path, name: str) -> Path | None:
    if not name:
        return None
    for sub in ("poses_scored", "poses_minimized", "poses", "poses_raw"):
        candidate = out_dir / sub / name
        if candidate.exists():
            return candidate
    direct = out_dir / name
    return direct if direct.exists() else None


def _ca_centroid(pdb: Path) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    try:
        with pdb.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
    except (OSError, ValueError):
        return None
    if not xs:
        return None
    return [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]


_DDG_RE = re.compile(r"ΔΔG[^-\d]*(-?\d+\.?\d*)")
_DG_RE = re.compile(r"ΔG[^-\d]*(-?\d+\.?\d*)")


def _selectivity_summary(job: Job) -> dict[str, Any] | None:
    """Pull the ΔΔG line out of the selectivity run's own output."""
    text = "\n".join(job.lines[-200:])
    match = _DDG_RE.search(text)
    if not match:
        return None
    ddg = float(match.group(1))
    return {
        "ddg": ddg,
        "verdict": "on-target selective" if ddg < 0 else "off-target favoured",
        "caveat": ("Below about 1 kcal/mol this is a direction, not a measurement — "
                   "the charged floor is the same size."),
    }


def _crystal_headline(job: Job) -> dict[str, Any] | None:
    text = "\n".join(job.lines[-80:])
    match = _DG_RE.search(text)
    if not match:
        return None
    dg = float(match.group(1))
    return {"delta_g": dg, "kd": _kd_from_dg(dg), "n_poses": 1, "n_clusters": 1,
            "pose_file": None, "charged_confidence": None}


# --------------------------------------------------------------------------- #
#  HTTP
# --------------------------------------------------------------------------- #

class StudioHandler(BaseHTTPRequestHandler):
    """Routes. Anything not under /api is a static file from ``web/``."""

    server_version = "HybriDockStudio"
    manager: JobManager  # injected on the server instance

    # quieter logs: one line per request at DEBUG, not stderr spam
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # ---- plumbing ----

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400, **extra: Any) -> None:
        self._send_json({"error": {"message": message, **extra}}, status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        try:
            parsed: dict[str, Any] = json.loads(raw.decode("utf-8"))
            return parsed
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed request: {exc}") from exc

    def _send_file(self, path: Path, download: bool = False) -> None:
        if not path.is_file():
            self._error("Not found", 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    # ---- GET ----

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                self._send_file(WEB_DIR / "index.html")
            elif route.startswith("/static/"):
                self._serve_static(route)
            elif route == "/api/env":
                self._send_json(check_environment())
            elif route == "/api/fields":
                self._send_json({
                    "fields": serialize_fields(),
                    "dock_keys": tui.DOCK_KEYS,
                    "crystal_keys": tui.CRYSTAL_KEYS,
                    "selectivity_keys": tui.SEL_KEYS,
                    "stages": [{"key": k, "label": lbl, "weight": w} for k, lbl, w in tui.STAGES],
                })
            elif route == "/api/examples":
                self._send_json({"examples": available_examples()})
            elif route == "/api/jobs":
                self._send_json({"jobs": [j.snapshot() for j in self.manager.jobs.values()]})
            elif route.startswith("/api/jobs/"):
                self._job_get(route, query)
            else:
                self._error("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-poll; nothing to do
        except Exception as exc:
            logger.exception("GET %s failed", route)
            self._error(str(exc), 500)

    def _serve_static(self, route: str) -> None:
        rel = route[len("/static/"):]
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self._error("Forbidden", 403)  # path traversal
            return
        self._send_file(target)

    def _job_get(self, route: str, query: dict[str, list[str]]) -> None:
        parts = route.split("/")  # ['', 'api', 'jobs', '<id>', maybe 'results'|'file']
        job = self.manager.jobs.get(parts[3] if len(parts) > 3 else "")
        if job is None:
            self._error("No such run", 404)
            return
        tail = parts[4] if len(parts) > 4 else ""
        if not tail:
            since = _int_or(query.get("since", ["0"])[0], 0)
            self._send_json(job.snapshot(since))
        elif tail == "results":
            self._send_json(read_results(job))
        elif tail == "file":
            name = query.get("name", [""])[0]
            target = (job.output_dir / name).resolve()
            if not str(target).startswith(str(job.output_dir.resolve())):
                self._error("Forbidden", 403)
                return
            self._send_file(target, download=query.get("download", ["0"])[0] == "1")
        elif tail == "pose":
            name = query.get("name", [""])[0]
            pose = _find_pose(job.output_dir, name)
            if pose is None:
                self._error("Pose file not found", 404)
                return
            self._send_file(pose)
        else:
            self._error("Not found", 404)

    # ---- POST ----

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as exc:
            self._error(str(exc), 400)
            return
        try:
            if route == "/api/validate":
                self._send_json(validate_request(body))
            elif route == "/api/preview":
                self._send_json({"command": " ".join(_build_command(body))})
            elif route == "/api/run":
                self._start_run(body)
            elif route == "/api/upload":
                self._send_json(self._upload(body))
            elif route.startswith("/api/jobs/") and route.endswith("/cancel"):
                job_id = route.split("/")[3]
                self._send_json({"cancelled": self.manager.cancel(job_id)})
            else:
                self._error("Not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as exc:
            self._error(str(exc), 400)
        except RuntimeError as exc:
            self._error(str(exc), 409)
        except Exception as exc:
            logger.exception("POST %s failed", route)
            self._error(str(exc), 500)

    def _start_run(self, body: dict[str, Any]) -> None:
        check = validate_request(body)
        if not check["ok"]:
            self._error("Some settings still need fixing.", 400, fields=check["errors"])
            return
        values, mode = _values_and_mode(body)
        cmd = _build_command(body)
        out_dir = Path(values.get("output_dir") or SESSION_ROOT).expanduser()
        job = self.manager.start(cmd, mode, values, out_dir)
        self._send_json({"job": job.snapshot()})

    def _upload(self, body: dict[str, Any]) -> dict[str, Any]:
        """Accept a PDB as text and write it where the CLI can read it.

        JSON rather than multipart on purpose: PDB files are text, the server is
        loopback-only, and this keeps the stdlib server free of a form parser.
        """
        name = Path(str(body.get("name") or "uploaded.pdb")).name
        if not name.lower().endswith(".pdb"):
            raise ValueError("That file is not a .pdb — HybriDock-Pep needs a PDB structure.")
        content = str(body.get("content") or "")
        if "ATOM" not in content and "HETATM" not in content:
            raise ValueError("That PDB has no ATOM records. Is it the right file?")
        target_dir = SESSION_ROOT / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        target.write_text(content, encoding="utf-8")
        chains = sorted({line[21] for line in content.splitlines()
                         if line.startswith("ATOM") and len(line) > 21})
        n_atoms = sum(1 for line in content.splitlines() if line.startswith("ATOM"))
        return {"path": str(target.resolve()), "name": name,
                "atoms": n_atoms, "chains": chains}


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Check one set of form values the way the terminal UI would.

    Args:
        body: ``{"mode": ..., "values": {...}}`` straight from the browser.

    Returns:
        ``ok`` plus per-field ``errors``, the ``skipped`` settings that do not
        apply to these choices, peptide stats for the chips, a runtime estimate,
        and the exact command that would run (``None`` if anything is wrong).
    """
    values, mode = _values_and_mode(body)
    keys = _keys_for_mode(mode)
    errors: dict[str, str] = {}
    # The browser greys out inactive settings using this list rather than
    # reimplementing the skip rules in JS, where they would quietly drift.
    skipped = [f.key for f in tui.FIELDS
               if f.key not in keys or tui._skip_key(f.key, values)]
    for key in keys:
        if tui._skip_key(key, values):
            continue
        field = tui.FIELD.get(key)
        if field is None:
            continue
        problem = field.validate(values.get(key, ""))  # type: ignore[misc]
        if problem:
            errors[key] = problem
    if mode == "selectivity":
        for key in ("offtarget_receptor", "offtarget_site", "offtarget_box"):
            if not (values.get(key) or "").strip():
                errors[key] = "needed to compare two targets"
    if mode == "crystal" and not (values.get("peptide_pdb") or "").strip():
        # peptide_pdb is an optional FormField (the dock path ignores it), but it
        # is the whole input in crystal mode — without it the CLI would be handed
        # an empty --peptide-pdb and fail minutes later instead of here.
        errors["peptide_pdb"] = "pick the bound peptide pose — that is what gets scored"
    command = None
    if not errors:
        try:
            command = " ".join(_build_command(body))
        except (ValueError, KeyError, IndexError) as exc:
            errors["_command"] = f"Could not build the command: {exc}"
    return {
        "ok": not errors,
        "errors": errors,
        "skipped": skipped,
        "peptide": peptide_stats(values.get("peptide", "")),
        "estimate_seconds": estimate_seconds(values, mode),
        "command": command,
    }


def _values_and_mode(body: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Normalise a request body into the TUI's ``values`` dict plus a mode."""
    raw = body.get("values") or {}
    values = {str(k): ("" if v is None else str(v)) for k, v in raw.items()}
    mode = str(body.get("mode") or values.get("mode") or "ai").strip().lower()
    if mode == "ai":
        mode = "dock"
    values.setdefault("mode", "crystal" if mode == "crystal" else "ai")
    for field in tui.FIELDS:  # fill gaps with the documented defaults
        values.setdefault(field.key, field.default)
    if mode == "selectivity":
        # build_selectivity_command needs both sites; there is no blind path for a
        # two-receptor comparison, so don't let a leftover blind=y break the build.
        values["blind"] = "n"
    return values, mode


def _keys_for_mode(mode: str) -> list[str]:
    if mode == "crystal":
        return list(tui.CRYSTAL_KEYS)
    if mode == "selectivity":
        return list(tui.SEL_KEYS)
    return list(tui.DOCK_KEYS)


def _build_command(body: dict[str, Any]) -> list[str]:
    """Hand the request to the TUI's own command builders."""
    values, mode = _values_and_mode(body)
    scoring = str(body.get("scoring") or "vina")
    # tui is an untyped module (it predates the strict-mypy modules); list() both
    # normalises the return type and makes the boundary explicit.
    if mode == "crystal":
        return list(tui.build_crystal_command(values))
    if mode == "selectivity":
        return list(tui.build_selectivity_command(values, scoring=scoring))
    return list(tui.build_dock_command(values, scoring=scoring))


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> int:
    """Run the studio until interrupted.

    Args:
        host: Interface to bind. Loopback by default — this is a local tool.
        port: TCP port; if taken, the next 20 are tried.
        open_browser: Open the default browser at the URL once bound.

    Returns:
        Process exit code (0 on a clean Ctrl-C).
    """
    manager = JobManager()
    handler = type("BoundHandler", (StudioHandler,), {"manager": manager})

    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, candidate), handler)
            port = candidate
            break
        except OSError:
            continue
    if httpd is None:
        print(f"Could not bind a port in {port}\u2013{port + 19}. Is another studio running?",
              flush=True)
        return 1

    url = f"http://{host}:{port}/"
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    # flush: the URL is the only thing this command exists to tell you, and a
    # piped/redirected stdout would otherwise buffer it until the server exits.
    print(f"\n  HybriDock-Pep studio is running at  {url}", flush=True)
    print("  Open that in your browser. Ctrl-C here when you're done.\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.", flush=True)
        for job in manager.jobs.values():
            if job.state in ("queued", "running"):
                manager.cancel(job.id)
    finally:
        httpd.server_close()
    return 0
