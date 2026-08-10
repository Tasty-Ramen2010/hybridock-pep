"""HybriDock-Pep terminal UI — a btop/nvtop-style front-end for the docking pipeline.

Cross-platform (Windows / macOS / Linux) via prompt_toolkit — no curses, no browser. Ways to run:

    hybridock-tui            # full-screen interactive UI (default)
    hybridock-tui --demo     # full-screen UI, auto-run a synthetic pipeline (no GPU — watch the bars)
    hybridock-tui --cli      # plain step-by-step wizard (no full-screen; SSH / dumb terminals)
    hybridock-tui --print    # build & print the `hybridock-pep dock` command, run nothing

Run modes (buttons + keys): Full (n=100, MM-GBSA), Half (n=50), Quick (n=20) — all three run the
AI-pose docking pipeline (Scoring mode "ai"); the physics backends (Vina for clash relief, AD4 for
optional telemetry) are chosen automatically per preset and are never a user-facing setting.
Selectivity ΔΔG (target vs off-target) and Demo run the same "ai" pipeline. Crystal Score runs the
other mode: score an EXISTING bound pose (a crystal structure, no sampling) with the crystal-tuned
model — see the "Scoring mode" field. A built-in file/folder browser (Browse button / Ctrl-B) fills
any path field; you can also just drag a file onto a path field. Controls are mouse-clickable
buttons + universal Ctrl-key accelerators (no Mac/Windows-only function keys). The layout resizes
with the terminal. Dependency-light (prompt_toolkit only).
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from hybridock_pep.output import art
from hybridock_pep.ui import help_content

AA = set("ACDEFGHIKLMNPQRSTVWY")

# --------------------------------------------------------------------------- #
#  First-run state
# --------------------------------------------------------------------------- #

# A dotfile in $HOME rather than anything inside the repo: the repo is often a
# fresh clone (or read-only, or shared), and "have I seen the tour" is a
# property of the person, not of the checkout.
STATE_DIR = Path.home() / ".hybridock-pep"
WELCOME_MARKER = STATE_DIR / "ui_welcomed"


def is_first_run(marker: Path | None = None) -> bool:
    """True when the guided tour has never been dismissed on this machine.

    Fails to True: if $HOME is unreadable we would rather show a returning
    user the tour again than leave a first-time user staring at a bare form.
    """
    path = marker if marker is not None else WELCOME_MARKER
    try:
        return not path.exists()
    except OSError:
        return True


def mark_welcomed(marker: Path | None = None) -> None:
    """Record that the tour has been dismissed. Never raises — a read-only
    home directory must not stop the UI from opening."""
    path = marker if marker is not None else WELCOME_MARKER
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n")
    except OSError:
        pass

# --------------------------------------------------------------------------- #
#  Emergency stop
# --------------------------------------------------------------------------- #


def terminate_process_tree(proc, grace: float = 4.0) -> bool:
    """Stop a running dock and everything it spawned. Returns True if it died.

    Killing only `proc` is not enough. `hybridock-pep dock` launches the
    rapidock environment's python as a *child* process to do GPU sampling; kill
    the parent alone and that child keeps running, holding the GPU and writing
    into the output directory. For an emergency stop that is the worst possible
    outcome — the user thinks they stopped it and it is still going.

    So the run is started in its own process group (``start_new_session=True``)
    and the whole group is signalled: SIGTERM first so OpenMM/torch can unwind,
    then SIGKILL for anything still alive after `grace` seconds.

    Falls back to plain terminate()/kill() where process groups do not exist
    (Windows). Never raises: a stop button that can throw is not a stop button.
    """
    if proc is None or proc.poll() is not None:
        return True

    def _signal_group(sig) -> None:
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass  # group already gone, or no permission — fall through
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError, ValueError):
            pass

    _signal_group(signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)

    _signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return proc.poll() is not None


# --------------------------------------------------------------------------- #
#  Form model + validation
# --------------------------------------------------------------------------- #


@dataclass
class FormField:
    key: str
    label: str
    default: str
    help: str
    validate: "callable"
    is_path: bool = False
    is_dir: bool = False
    optional: bool = False


def _valid_peptide(v):
    v = v.strip().upper()
    if not v:
        return "peptide is required"
    bad = sorted(set(v) - AA)
    if bad:
        return f"non-amino-acid letter(s): {' '.join(bad)}"
    if not 3 <= len(v) <= 30:
        return "length must be 3–30 residues"
    return None


def _valid_pdb_file(v):
    v = v.strip()
    if not v:
        return "path is required"
    return None if Path(v).expanduser().is_file() else "file not found"


def _valid_receptor(v):
    return _valid_pdb_file(v)


def _valid_site(v):
    parts = v.split()
    if len(parts) != 3:
        return "need three numbers: x y z"
    try:
        [float(p) for p in parts]
    except ValueError:
        return "x y z must be numbers"
    return None


def _posint(minv, maxv):
    def f(v):
        try:
            n = int(v)
        except ValueError:
            return "must be an integer"
        return None if minv <= n <= maxv else f"out of range [{minv}, {maxv}]"
    return f


#: The two things a user can actually ask this tool to do — NOT the vina/ad4
#: physics backends, which run automatically underneath "ai" and are never a
#: user-facing choice (see docs/guide topic 7, "AI scoring vs physics"; Vina's
#: score is raw clash-relief telemetry, not calibrated ΔG). "ai" runs the full
#: RAPiDock-sampling + affinity-model pipeline (`dock`); "crystal" scores an
#: EXISTING bound pose with the crystal-tuned model, no sampling (`crystal-score`).
_MODES = {"ai", "crystal"}


def _valid_mode(v):
    v = v.strip().lower()
    return None if v in _MODES else "use ai or crystal"


def _valid_dir(v):
    return None if v.strip() else "output dir is required"


def _optional(fn):
    def f(v):
        return None if not v.strip() else fn(v)
    return f


FIELDS = [
    FormField("peptide", "Peptide sequence", "LISAAALAAIFAAALAC",
              "One-letter amino-acid sequence to dock (3–30 residues).", _valid_peptide),
    FormField("receptor", "Target receptor PDB", "data/pdbs/1T2D_receptor.pdb",
              "On-target receptor .pdb — drag the file here or use Browse (Ctrl-B).",
              _valid_receptor, is_path=True),
    FormField("site", "Target site  x y z", "28.25 15.44 66.27",
              "On-target box center (Å) — three numbers.", _valid_site),
    FormField("box", "Target box (Å)", "30",
              "On-target cubic box edge (Å). 30 for 12-mers+.", _posint(10, 60)),
    FormField("n_samples", "N samples", "100",
              "RAPiDock diffusion poses to generate (per receptor).", _posint(1, 500)),
    FormField("mode", "Scoring mode", "ai",
              "ai = AI-pose docking (default; runs RAPiDock sampling + the affinity model — "
              "Full/Half/Quick/Selectivity/Demo). crystal = score an EXISTING bound pose "
              "(a crystal structure) with the crystal-tuned model, no sampling — "
              "Crystal Score, needs Peptide pose PDB below.", _valid_mode),
    FormField("peptide_pdb", "Peptide pose PDB (crystal mode)", "",
              "crystal mode only: the bound peptide PDB (a crystal/native pose) to score.",
              _optional(_valid_pdb_file), is_path=True, optional=True),
    FormField("refine_topk", "Refine top-K (MM-GBSA)", "0",
              "MM-GBSA on top-K clusters (0 = off; ai mode only).", _posint(0, 50)),
    FormField("output_dir", "Output dir", "runs/tui_run",
              "Where results are written — Browse (Ctrl-B) to pick a folder.", _valid_dir,
              is_path=True, is_dir=True),
    FormField("offtarget_receptor", "Off-target PDB (selectivity)", "",
              "Selectivity only: the off-target receptor .pdb.", _optional(_valid_receptor),
              is_path=True, optional=True),
    FormField("offtarget_site", "Off-target site  x y z", "",
              "Selectivity only: off-target box center (Å).", _optional(_valid_site), optional=True),
    FormField("offtarget_box", "Off-target box (Å)", "",
              "Selectivity only: off-target box edge (Å).", _optional(_posint(10, 60)), optional=True),
]
FIELD = {f.key: f for f in FIELDS}
DOCK_KEYS = ["peptide", "receptor", "site", "box", "n_samples", "refine_topk", "output_dir"]
SEL_KEYS = DOCK_KEYS + ["offtarget_receptor", "offtarget_site", "offtarget_box"]
#: Fields needed by "crystal" mode (score an existing bound pose — no site/box/
#: n_samples/output_dir; see build_crystal_command()).
CRYSTAL_KEYS = ["peptide", "receptor", "peptide_pdb"]


def _resolve_exe(name: str) -> str | None:
    """Find `name` to actually execute, tolerating a PATH that doesn't
    include the current environment's bin/ directory.

    launch_ui.sh locates the right Python interpreter for score-env by
    searching common conda install locations and execs it by full path —
    but that alone does not add that environment's bin/ to PATH for the
    process it starts. Every real run in this UI shells out to the bare
    name "hybridock-pep" (kept bare deliberately in build_dock_command /
    build_selectivity_command so the *echoed* command a user sees or
    copies stays clean), so a plain shutil.which() lookup against an
    unmodified PATH can fail even though the TUI itself is running
    correctly from inside score-env — the exact "not on PATH" error a
    freshly-launched UI hit right after install. sys.executable is always
    correct (it's this very interpreter), so its sibling in the same
    bin/ directory is the real `name`, regardless of what PATH says.
    """
    found = shutil.which(name)
    if found:
        return found
    sibling = Path(sys.executable).parent / name
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def clean_dropped_path(s: str) -> str:
    """Normalise a path a terminal produced from a file-drop (quotes, file://, escaped spaces)."""
    s = s.strip()
    for _ in range(2):  # may be quote-wrapped AND file://-prefixed, either order
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            s = s[1:-1].strip()
        if s.startswith("file://"):
            s = s[7:]
    return s.replace("\\ ", " ").replace("\\~", "~").replace("\\(", "(").replace("\\)", ")").strip()


def build_dock_command(v, scoring="vina", exe="hybridock-pep"):
    """Build an ``ai``-mode (`dock`) command line.

    ``scoring`` is the vina/ad4 physics-backend selection — NOT a user-facing
    field (see the "Scoring mode" FormField, which is ai/crystal). Callers pick
    it programmatically per run preset (see run_dock()'s Full/Half/Quick
    presets); it is never read from form input.
    """
    x, y, z = v["site"].split()
    cmd = [exe, "dock", "--peptide", v["peptide"].strip().upper(),
           "--receptor", str(Path(v["receptor"]).expanduser()),
           "--site", x, y, z, "--box", v["box"].strip(),
           "--n-samples", v["n_samples"].strip(), "--scoring", scoring,
           "--output-dir", v["output_dir"].strip()]
    # .strip() before the truthiness test: "   " is truthy but int("   ")
    # raises, which crashed command building when a field was typed into and
    # then cleared.
    topk = (v.get("refine_topk") or "").strip()
    if topk and int(topk) > 0:
        cmd += ["--refine-topk", topk]
    return cmd


def build_crystal_command(v, exe="hybridock-pep"):
    """Build a ``crystal``-mode (`crystal-score`) command line: score an
    existing bound pose with the crystal-tuned model. No sampling, no
    site/box/n-samples/output-dir — see CRYSTAL_KEYS."""
    return [exe, "crystal-score",
            "--receptor", str(Path(v["receptor"]).expanduser()),
            "--peptide-pdb", str(Path(v["peptide_pdb"]).expanduser()),
            "--peptide", v["peptide"].strip().upper()]


def build_selectivity_command(v, scoring="vina,ad4", exe="hybridock-pep"):
    """Build a selectivity command line. Selectivity is ai-mode only (it docks
    against two receptors); ``scoring`` is the automatic physics-backend
    selection, same caveat as build_dock_command()."""
    tx, ty, tz = v["site"].split()
    ox, oy, oz = v["offtarget_site"].split()
    return [exe, "selectivity", "--peptide", v["peptide"].strip().upper(),
            "--target-receptor", str(Path(v["receptor"]).expanduser()),
            "--target-site", tx, ty, tz, "--target-box", v["box"].strip(),
            "--offtarget-receptor", str(Path(v["offtarget_receptor"]).expanduser()),
            "--offtarget-site", ox, oy, oz, "--offtarget-box", v["offtarget_box"].strip(),
            "--n-samples", v["n_samples"].strip(), "--scoring", scoring,
            "--output-dir", v["output_dir"].strip()]


def validate(values, keys):
    out = []
    for k in keys:
        f = FIELD[k]
        if f.optional and not values.get(k, "").strip():
            out.append(f"{f.label}: required for this mode")
            continue
        if (m := f.validate(values.get(k, ""))):
            out.append(f"{f.label}: {m}")
    return out


# --------------------------------------------------------------------------- #
#  Pipeline progress model
# --------------------------------------------------------------------------- #

STAGES = [("sample", "AI sampling (RAPiDock)", 0.50), ("min", "minimize poses", 0.05),
          ("score", "physics rescoring", 0.25), ("cluster", "RMSD clustering", 0.05),
          ("affinity", "affinity / MM-GBSA", 0.15)]
_ORDER = [k for k, _, _ in STAGES]
_WEIGHT = {k: w for k, _, w in STAGES}
_LABEL = {k: l for k, l, _ in STAGES}
_COUNT = re.compile(r"(\d+)\s*(?:/|of)\s*(\d+)")
_PCT = re.compile(r"(\d+)\s*%")

#: A real run's unambiguous stage announcements — the exact text
#: output/progress.py's PipelineProgress.step() writes ("▶ [n/total] <label>…"),
#: keyed by LABELS' phrasing (lowercased substring match) → this module's
#: 5-bucket stage model. Matched ONLY against lines that start with the "▶"
#: marker itself (see feed() below) — never against warnings, per-item
#: progress-tick lines, or other incidental output, which is what let
#: "%d poses failed Vina scoring" (a WARNING driver.py logs whenever any pose
#: fails scoring — not rare) match a loose "pose" substring check and yank
#: p.stage back to "sample" mid-Stage-2, corrupting fraction() and making the
#: Stage-1 gallery animation spuriously restart partway through a real run.
_STEP_LABEL_TO_STAGE = {
    "generating poses": "sample",
    "loading input poses": "sample",
    "scoring poses": "score",
    "clustering poses": "cluster",
    "refining top poses": "affinity",       # "Refining top poses (MM-GBSA + entropy)"
    "charged-residue correction": "affinity",
    "final ranking": "affinity",            # "Final ranking & ΔG"
}


class PipelineProgress:
    def __init__(self):
        self.stage = None
        self.cur = self.total = 0
        self.done: set[str] = set()
        self.finished = False
        self.t0 = time.time()

    def _enter(self, key):
        if key == self.stage:
            return
        if key in _ORDER:
            for k in _ORDER[:_ORDER.index(key)]:
                self.done.add(k)
        self.stage, self.cur, self.total = key, 0, 0

    def feed(self, line):
        low = line.lower()
        if any(w in low for w in ("run_metadata", "best pose", "results written", "ranked_poses",
                                  "ΔΔg", "δδg", "selectivity")):
            self.finished = True
            self.done.update(_ORDER)
            return

        if line.strip().startswith("▶"):
            # A real run's own unambiguous stage announcement — trust this
            # over any loose keyword match (see _STEP_LABEL_TO_STAGE).
            for phrase, stage_key in _STEP_LABEL_TO_STAGE.items():
                if phrase in low:
                    self._enter(stage_key)
                    break
        # Below: demo_lines()'s literal "Stage N[.M]:" markers only — these are
        # synthetic text this module fully controls, not real subprocess
        # output, so there is no risk of an incidental substring elsewhere
        # (a warning, a per-item progress tick) matching by accident.
        elif "stage 3.6" in low or "stage 3.5" in low:
            self._enter("affinity")
        elif "stage 3" in low:
            self._enter("cluster")
        elif "stage 2" in low:
            self._enter("score")
        elif "stage 1.5" in low:
            self._enter("min")
        elif "stage 1" in low:
            self._enter("sample")

        if "complete" in low and self.stage:
            self.done.add(self.stage)
            self.cur = self.total or self.cur
        if (m := _COUNT.search(line)):
            self.cur, self.total = int(m.group(1)), int(m.group(2))
        elif (p := _PCT.search(line)):
            self.total, self.cur = 100, int(p.group(1))

    def fraction(self):
        if self.finished:
            return 1.0
        frac = sum(_WEIGHT[k] for k in self.done)
        if self.stage and self.stage not in self.done:
            frac += _WEIGHT.get(self.stage, 0) * min((self.cur / self.total) if self.total else 0, 1)
        return max(0.0, min(frac, 1.0))

    def counter(self):
        return f"{self.cur}/{self.total}" if self.total else ""

    def label(self):
        return "done" if self.finished else _LABEL.get(self.stage, "starting…")

    def elapsed(self):
        return int(time.time() - self.t0)




def demo_lines(n=100):
    yield ("Stage 1: RAPiDock-Reloaded sampling — device: mps", 0.15)
    for i in range(1, n + 1):
        yield (f"  pose {i}/{n} sampled", 0.010)
    # A deliberate pause here, not inside the fast per-pose loop above: a real
    # Stage 1 wait is the one part of a run that's genuinely silent for a
    # while, and 100 lines landing at 10ms apart would scroll straight
    # through a gallery piece before a human — or a terminal repaint — ever
    # caught it. "__ART__:0" is a sentinel worker() recognizes and hands to
    # show_art() instead of appending literally — that's what animates it.
    yield ("__ART__:0", 0.3)
    yield (f"Stage 1 complete: {n} poses parsed", 0.1)
    yield ("Stage 1.5: OpenMM minimization", 0.1)
    yield (f"Stage 1.5 complete: {n} poses minimized", 0.1)
    yield ("Stage 2: physics rescoring (vina + ad4)", 0.1)
    for i in range(1, n + 1):
        yield (f"  scored {i}/{n} poses", 0.007)
    yield (f"Stage 2 complete: {n} poses scored", 0.1)
    yield ("Stage 3: RMSD clustering", 0.15)
    yield ("Stage 3 complete: 7 clusters", 0.1)
    yield (f"Stage 3.6: AI-pose affinity ΔG on {n}/{n} poses", 0.2)
    yield ("Best pose ΔG = -9.31 kcal/mol  (cluster 1, 34 members)", 0.1)
    yield ("Results written → runs/tui_run/ranked_poses.csv", 0.05)


# --------------------------------------------------------------------------- #
#  Plain CLI wizard
# --------------------------------------------------------------------------- #

def run_cli_wizard(print_only=False):
    from prompt_toolkit import prompt
    print("\n  HybriDock-Pep · guided dock  (Ctrl-C to abort)\n  " + "-" * 48)
    values = {}
    for k in DOCK_KEYS:
        f = FIELD[k]
        print(f"\n  {f.label}\n    {f.help}")
        while True:
            ans = prompt(f"  {f.key} > ", default=f.default)
            if f.is_path:
                ans = clean_dropped_path(ans)
            if (m := f.validate(ans)) is None:
                values[k] = ans
                break
            print(f"    ✗ {m}")
    cmd = build_dock_command(values)
    print("\n  Command:\n    " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if print_only:
        return 0
    if prompt("  Run it now? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("  Not run.")
        return 0
    exe_path = _resolve_exe(cmd[0])
    if exe_path is None:
        print(f"  ✗ '{cmd[0]}' not on PATH — run `conda activate score-env` first.")
        return 127
    proc = subprocess.Popen([exe_path, *cmd[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write("  " + line)
    return proc.wait()


# --------------------------------------------------------------------------- #
#  Full-screen UI
# --------------------------------------------------------------------------- #

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _bar(frac, width):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_fullscreen(auto_demo=False):
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import DynamicContainer, HSplit, VSplit, Window, WindowAlign
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, Frame, TextArea

    progress = {"p": None}
    state = {"running": False, "rc": None, "lines": [
        "Ready. Fill the form, then press a run button (Full ▶ / Half / Quick / Selectivity).",
        "No GPU? Press Demo ▷ (Ctrl-T) to watch a full simulated n=100 run.",
        "Path fields: drag a .pdb on, or press Browse (Ctrl-B). Ctrl-G for help.",
    ]}
    inputs = {f.key: TextArea(text=f.default, multiline=False, height=1, style="class:input", prompt=" ")
              for f in FIELDS}

    def vals():
        return {k: ta.text for k, ta in inputs.items()}

    # wrap_lines=True: with wrapping off, a long line (the echoed `$ hybridock-pep
    # dock --peptide ... --output-dir ...` command, well over 100 columns) drags the
    # buffer's horizontal scroll far enough right to keep the cursor in view that
    # every *shorter* line above it scrolls out of view along with it — the whole
    # pane's horizontal offset is shared, not per-line. The visible symptom was
    # short status lines reduced to a trailing fragment ("p.", "/ selectivity).")
    # with everything before that scroll offset invisible. Wrapping avoids the
    # shared horizontal scroll entirely; the ASCII art gallery's widest frame
    # (~56 cols) still fits unwrapped in any reasonably-sized terminal.
    output = TextArea(text="\n".join(state["lines"]), read_only=True, scrollbar=True,
                      style="class:output", focus_on_click=True, wrap_lines=True)

    def append(line):
        state["lines"].append(line.rstrip("\n"))
        output.text = "\n".join(state["lines"][-2000:])
        output.buffer.cursor_position = len(output.text)

    def replace_tail(n, lines):
        if n:
            state["lines"][-n:] = lines
        else:
            state["lines"].extend(lines)
        output.text = "\n".join(state["lines"][-2000:])
        output.buffer.cursor_position = len(output.text)

    def show_art(index, delay=0.16):
        """Animate one gallery piece into the output log, frame by frame.

        Blocks the calling thread between frames (same trick as
        art.animate_gallery_piece for the plain CLI, minus the raw ANSI —
        here we just rewrite the tail of state["lines"]). Used for one-off
        showings only (the demo's single scripted appearance, the Ctrl-A
        peek) — a *real* run's Stage 1 animation is continuous and lives in
        the progress panel instead (see refresh_progress), driven by
        elapsed time on every redraw tick rather than by dropping repeated
        copies into this scrolling log. The Ctrl-A keybinding handler runs
        on the main event loop and must wrap this in its own thread, or it
        would freeze the UI while it sleeps between frames.

        The first frame lingers an extra art.HOLD_FIRST_FRAME_S before the
        animation starts moving, and the completed last frame lingers an
        extra art.HOLD_LAST_FRAME_S before this returns — same idea as
        art.animate_gallery_piece for the plain CLI.
        """
        try:
            title, frames = art.gallery_piece(index)
            first = frames[0].splitlines()
            append(f"\n   · {title}")
            replace_tail(0, [f"   {line}" for line in first])
            n = len(first)
            for i, frame in enumerate(frames[1:], start=1):
                time.sleep(delay + (art.HOLD_FIRST_FRAME_S if i == 1 else 0.0))
                lines = frame.splitlines()
                replace_tail(n, [f"   {line}" for line in lines])
                n = len(lines)
            time.sleep(art.HOLD_LAST_FRAME_S)
            append("")
        except Exception:
            pass

    hint = FormattedTextControl(text="")

    def current_mode() -> str:
        return inputs["mode"].text.strip().lower()

    def refresh_hint():
        # What the ✓/✗ line checks tracks the "Scoring mode" field: ai needs
        # the dock fields (site/box/n-samples/output-dir), crystal needs
        # peptide_pdb instead and ignores those. Full/Half/Quick/Selectivity/
        # Demo force mode back to "ai" before they validate (see run_dock()),
        # so this only actually matters while the user has typed "crystal".
        if current_mode() == "crystal":
            errs = validate(vals(), CRYSTAL_KEYS)
            ready = "  ✓ ready — Crystal Score ⚗ (Ctrl-Y)"
        else:
            errs = validate(vals(), DOCK_KEYS)
            ready = "  ✓ ready — Full ▶ (Ctrl-R) or Demo ▷ (Ctrl-T)"
        hint.text = ([("class:err", "  ✗ " + errs[0])] if errs
                     else [("class:ok", ready)])

    _busy = {"on": False}

    def make_cb(f):
        def _cb(_=None):
            if _busy["on"]:
                return
            if f.is_path:
                cur = inputs[f.key].text
                cl = clean_dropped_path(cur)
                if cl != cur:
                    _busy["on"] = True
                    inputs[f.key].text = cl
                    inputs[f.key].buffer.cursor_position = len(cl)
                    _busy["on"] = False
            refresh_hint()
        return _cb

    for f in FIELDS:
        inputs[f.key].buffer.on_text_changed += make_cb(f)

    # ----- progress panel -----
    progress_ctrl = FormattedTextControl(text="")

    def refresh_progress():
        p = progress["p"]
        if p is None:
            progress_ctrl.text = [("class:dim", "  idle — no run in progress")]
            return
        spin = "" if p.finished else _SPINNER[int(time.time() * 10) % len(_SPINNER)]
        verb = "" if p.finished else art.spinner_word(time.time())
        frac = p.fraction()
        cls = "class:okbar" if p.finished else "class:bar"
        cnt = p.counter()
        lines = [("class:dim", "  "), (cls, _bar(frac, 34)),
                 ("class:pct", f" {int(frac*100):3d}% "), ("class:dim", spin), ("", "\n"),
                 ("class:dim", "  stage: "), ("class:stage", p.label()),
                 ("class:dim", f"   {cnt}" if cnt else ""),
                 ("class:dim", f"   ·  {p.elapsed()}s"),
                 ("class:verb", f"   {verb}" if verb else "")]

        # One piece, animating continuously in place for the duration of
        # Stage 1 — not a periodic log entry. Driven purely by elapsed time
        # on every redraw tick (same mechanism as the spinner/verb above):
        # no thread, no appending to the output log, so it cannot interleave
        # badly with concurrently-arriving progress lines or accumulate
        # without bound no matter how long Stage 1 runs. A version of this
        # that dropped a fresh copy into the scrolling log every 30s was
        # exactly the "the log keeps growing / scrolling never catches up"
        # bug on a real, long run.
        if p.stage == "sample" and not p.finished:
            if state.get("art_stage") != "sample":
                state["art_stage"] = "sample"
                state["art_stage_t0"] = time.time()
            idx = state.get("art_piece_idx", 0)
            title, frames = art.gallery_piece(idx)
            frame_delay = 0.32
            elapsed_in_stage = time.time() - state["art_stage_t0"]
            # frame_index_for_elapsed (not a plain modulo) gives the first and
            # last frame of the loop an extra beat before/after the motion —
            # a uniform per-frame delay here made the loop snap straight from
            # last frame back to first with no beat at either end.
            fi = art.frame_index_for_elapsed(len(frames), elapsed_in_stage, frame_delay)
            lines.append(("", "\n"))
            lines.append(("class:dim", f"  · {title}\n"))
            for art_line in frames[fi].splitlines():
                lines.append(("class:artline", f"  {art_line}\n"))
        else:
            state["art_stage"] = p.stage

        progress_ctrl.text = lines

    # ----- run machinery -----
    _run_lock = threading.Lock()  # Prevent race condition on Ctrl+R spam

    def start_stream(cmd, demo=False, title=""):
        with _run_lock:  # Atomic check + set to prevent dual-spawn on Ctrl+R spam
            if state["running"]:
                append("… a run is already in progress")
                return
            state["running"], state["rc"] = True, None
            state["cancel"], state["proc"] = False, None
        progress["p"] = PipelineProgress()
        append("")
        append(f"▶ {title}" if title else "▶ run")
        if not demo:
            append("$ " + " ".join(shlex.quote(c) for c in cmd))
        egg = art.easter_egg_for_peptide(vals().get("peptide", ""))
        if egg:
            append(egg)

        # One gallery piece, picked once for this run, animated continuously
        # in the progress panel for the whole of Stage 1 (see refresh_progress)
        # — not dropped into the scrolling output log on a timer. state["art_stage"]
        # is reset here so a re-entry into "sample" (there's only ever one, but
        # this keeps the animation's own clock starting fresh per run) restarts
        # its clock from frame 0 instead of carrying over a stale timestamp.
        import random as _random  # noqa: PLC0415
        state["art_piece_idx"] = _random.randrange(len(art.GALLERY))
        state["art_stage"] = None

        def worker():
            try:
                if demo:
                    for text, delay in demo_lines(int(vals()["n_samples"] or 100)):
                        if state["cancel"]:
                            break
                        if text.startswith("__ART__:"):
                            show_art(int(text.split(":", 1)[1]))
                        else:
                            progress["p"].feed(text)
                            append(text)
                        time.sleep(delay)
                    state["rc"] = 0
                else:
                    exe_path = _resolve_exe(cmd[0])
                    if exe_path is None:
                        append(f"✗ '{cmd[0]}' not on PATH — run `conda activate score-env` first.")
                        state["rc"] = 127
                        return
                    # Own process group so Stop can signal the dock AND the
                    # rapidock sampling child it spawns. See
                    # terminate_process_tree().
                    proc = subprocess.Popen(
                        [exe_path, *cmd[1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        start_new_session=(os.name != "nt"),
                    )
                    state["proc"] = proc
                    for line in proc.stdout:
                        progress["p"].feed(line)
                        append(line)
                    state["rc"] = proc.wait()
                progress["p"].finished = True
                if state["cancel"]:
                    append("── STOPPED by user ──")
                else:
                    append(f"── finished (exit {state['rc']}) ──")
            except Exception as exc:  # noqa: BLE001
                append(f"✗ error: {exc}")
                state["rc"] = 1
            finally:
                state["running"] = False

        threading.Thread(target=worker, daemon=True).start()

    def stop_run():
        """Emergency stop — abort whatever is running, right now."""
        if not state["running"]:
            append("… nothing is running to stop")
            return
        state["cancel"] = True
        proc = state.get("proc")
        if proc is None:                       # demo, or process not spawned yet
            append("■ stopping…")
            return
        append("■ STOP requested — terminating the run and its sampling child…")
        # Off the UI thread: terminate_process_tree waits out a grace period,
        # and blocking the event loop would freeze the screen mid-stop and make
        # the user think the button did nothing.
        threading.Thread(
            target=lambda: append(
                "■ stopped." if terminate_process_tree(proc) else
                "■ could not confirm the process died — check with: ps aux | grep hybridock"
            ),
            daemon=True,
        ).start()

    def run_dock(n, scoring, base_title):
        # Full/Half/Quick are ai-mode presets — force the mode field back to
        # "ai" so pressing one of these always runs the dock pipeline even if
        # the user had typed "crystal" into Scoring mode. `scoring` (vina/ad4)
        # is chosen here per preset and never surfaced as a field.
        #
        # refine_topk is deliberately NOT overridden here — it's the user's
        # "Refine top-K (MM-GBSA)" field (default "0" = off, per its own
        # hint). It used to be force-set to 10 for Full/Ctrl-R regardless of
        # what the user had typed or left at the default, so MM-GBSA silently
        # ran even when the field said off. The field's current value is
        # exactly what build_dock_command() reads, so it now actually governs
        # every run button, not just do_print()'s preview.
        inputs["mode"].text = "ai"
        inputs["n_samples"].text = str(n)
        errs = validate(vals(), DOCK_KEYS)
        if errs:
            append("")
            append("✗ fix before running:")
            [append("    - " + e) for e in errs]
            return
        v = vals()
        topk = (v.get("refine_topk") or "").strip()
        title = base_title + (f", MM-GBSA top-{topk}" if topk and topk.isdigit() and int(topk) > 0 else "")
        start_stream(build_dock_command(v, scoring=scoring), title=title)

    def run_selectivity():
        inputs["mode"].text = "ai"
        errs = validate(vals(), SEL_KEYS)
        if errs:
            append("")
            append("✗ Selectivity needs the 3 Off-target fields filled:")
            [append("    - " + e) for e in errs]
            return
        start_stream(build_selectivity_command(vals(), scoring="vina,ad4"),
                     title="Selectivity ΔΔG (target vs off-target)")

    def run_crystal():
        # The other mode: score an EXISTING bound pose (a crystal structure)
        # with the crystal-tuned model. No sampling, no site/box/n-samples —
        # see CRYSTAL_KEYS and build_crystal_command().
        inputs["mode"].text = "crystal"
        errs = validate(vals(), CRYSTAL_KEYS)
        if errs:
            append("")
            append("✗ Crystal Score needs:")
            [append("    - " + e) for e in errs]
            return
        start_stream(build_crystal_command(vals()), title="Crystal score (existing bound pose)")

    def do_print():
        if current_mode() == "crystal":
            errs = validate(vals(), CRYSTAL_KEYS)
            if errs:
                append("✗ " + errs[0])
                return
            append("")
            append("$ " + " ".join(shlex.quote(c) for c in build_crystal_command(vals())))
            append("(printed only — press Crystal Score ⚗ / Ctrl-Y to run)")
            return
        errs = validate(vals(), DOCK_KEYS)
        if errs:
            append("✗ " + errs[0])
            return
        append("")
        append("$ " + " ".join(shlex.quote(c) for c in build_dock_command(vals())))
        append("(printed only — press Full ▶ / Ctrl-R to run)")

    def do_clear():
        state["lines"] = []
        output.text = ""

    # ----- overlays: help + file picker -----
    # Show the guided tour the first time this user ever opens the UI. A form
    # full of unexplained fields is the single biggest barrier for a
    # non-computational user, and they have no way to know help exists.
    view = {
        "mode": "welcome" if is_first_run() else "main",  # main|help|picker|welcome
        "topic": "1",
    }

    def _focus(win):
        """Move focus, tolerating a not-yet-rendered layout.

        Every view switch must carry focus with it: prompt_toolkit raises if
        the focused control is not part of the layout currently on screen.
        """
        try:
            get_app().layout.focus(win)
        except Exception:
            pass

    def go_main():
        view["mode"] = "main"
        _focus(inputs["peptide"])

    def toggle_help():
        if view["mode"] == "help":
            go_main()
        else:
            view["mode"] = "help"
            _focus(help_window)

    def show_welcome():
        view["mode"] = "welcome"
        _focus(welcome_window)

    picker = {"dir": Path.cwd(), "items": [], "idx": 0, "target": "receptor", "want_dir": False}

    def picker_refresh():
        d = picker["dir"]
        items = [("../", d.parent, True)]
        try:
            entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (PermissionError, OSError):
            entries = []
        for e in entries:
            if e.is_dir():
                items.append((e.name + "/", e, True))
            elif not picker["want_dir"] and e.suffix.lower() in (".pdb", ".pdbqt", ".ent"):
                items.append((e.name, e, False))
        picker["items"], picker["idx"] = items, 0

    def open_picker():
        # target = focused path field, else receptor
        target = "receptor"
        cur = get_app().layout.current_window
        for k, ta in inputs.items():
            if FIELD[k].is_path and ta.window is cur:
                target = k
                break
        picker["target"] = target
        picker["want_dir"] = FIELD[target].is_dir
        start = inputs[target].text.strip()
        p = Path(start).expanduser() if start else Path.cwd()
        picker["dir"] = p if p.is_dir() else (p.parent if p.parent.exists() else Path.cwd())
        picker_refresh()
        view["mode"] = "picker"
        get_app().layout.focus(picker_window)

    def picker_choose():
        name, path, is_dir = picker["items"][picker["idx"]]
        if is_dir:
            picker["dir"] = path.resolve()
            picker_refresh()
        else:
            inputs[picker["target"]].text = str(path)
            view["mode"] = "main"
            get_app().layout.focus(inputs[picker["target"]])

    def picker_use_dir():
        if picker["want_dir"]:
            inputs[picker["target"]].text = str(picker["dir"])
            view["mode"] = "main"
            get_app().layout.focus(inputs[picker["target"]])

    picker_ctrl = FormattedTextControl(text="", focusable=True)

    def refresh_picker():
        mode = "folder" if picker["want_dir"] else "file"
        frags = [("class:stage", f"  📁 {picker['dir']}  "),
                 ("class:dim", f"(pick a {mode} for “{FIELD[picker['target']].label}”)"), ("", "\n\n")]
        for i, (name, _p, is_dir) in enumerate(picker["items"]):
            sel = i == picker["idx"]
            style = "class:pickersel" if sel else ("class:pickerdir" if is_dir else "class:pickerfile")
            frags.append((style, ("  ▶ " if sel else "    ") + name + "\n"))
        picker_ctrl.text = frags

    picker_window = Window(picker_ctrl, style="class:picker")

    # ----- buttons -----
    def B(text, handler, w):
        return Button(text, handler=handler, width=w)

    run_row = VSplit([
        B("Full ▶", lambda: run_dock(100, "vina", "Full run (n=100)"), 10),
        B("Half", lambda: run_dock(50, "vina,ad4", "Half run (n=50)"), 8),
        B("Quick", lambda: run_dock(20, "vina", "Quick run (n=20)"), 9),
        B("Selectivity ⚖", run_selectivity, 16),
        B("Crystal ⚗", run_crystal, 11),
        B("Demo ▷", lambda: start_stream(None, demo=True, title="DEMO (simulated, no GPU)"), 10),
    ], padding=1, height=1)
    tool_row = VSplit([
        B("STOP ■", stop_run, 9),
        B("Browse 📁", open_picker, 12),
        B("Print", do_print, 9),
        B("Help ?", toggle_help, 10),
        B("Clear", do_clear, 9),
        B("Quit ✕", lambda: get_app().exit(result=0), 10),
    ], padding=1, height=1)

    def focused_key():
        """Which form field currently has the caret, or None."""
        try:
            control = get_app().layout.current_control
        except Exception:
            return None
        for key, ta in inputs.items():
            if ta.control is control:
                return key
        return None

    def label_fragments(f):
        """Label text, marked when its field is focused.

        Without this every label looked identical whether or not it was
        selected, so Tab appeared to do nothing and the form read as
        "only the first box is editable". The caret alone is not enough of a
        cue on a form this tall.
        """
        if focused_key() == f.key:
            return [("class:labelfocus", f"{'▶ ' + f.label:>28} ")]
        return [("class:label", f"{f.label:>28} ")]

    form_rows = [VSplit([
        Window(FormattedTextControl(lambda f=f: label_fragments(f)),
               width=29, height=1, dont_extend_width=True),
        inputs[f.key],
    ], height=1) for f in FIELDS]

    def focus_field(delta):
        """Move focus by `delta` fields, wrapping at the ends.

        Deliberately not layout.focus_next(): that walks every focusable window
        in the layout, so Tab wandered off into the output pane and the buttons
        instead of stepping through the form. This stays inside FIELDS.
        """
        keys = [f.key for f in FIELDS]
        cur = focused_key()
        idx = keys.index(cur) if cur in keys else 0
        _focus(inputs[keys[(idx + delta) % len(keys)]])
    form = HSplit(form_rows + [Window(hint, height=1)])

    title = Window(FormattedTextControl(
        [("class:title", "  HybriDock-Pep "), ("class:titledim", "· peptide → poses → ΔG → selectivity ")]),
        height=1, align=WindowAlign.LEFT, style="class:titlebar")
    footer = Window(FormattedTextControl(
        [("class:key", " Ctrl-R "), ("class:footer", "Full "), ("class:key", " Ctrl-T "), ("class:footer", "Demo "),
         ("class:key", " Ctrl-Y "), ("class:footer", "Crystal "),
         ("class:key", " Ctrl-C "), ("class:footer", "STOP "), ("class:key", " Ctrl-B "), ("class:footer", "Browse "),
         ("class:key", " Ctrl-G "), ("class:footer", "Help "), ("class:key", " Ctrl-Q "), ("class:footer", "Quit "),
         ("class:footer", " · ↑↓ or Tab move between fields · drag a .pdb onto a path field ")]),
        height=1, style="class:footerbar")

    main_view = HSplit([title, Frame(form, title="inputs"),
                        Frame(Window(progress_ctrl, height=D(min=2, max=24)), title="progress"),
                        Frame(output, title="output", height=D(min=3, weight=1)),
                        run_row, tool_row, footer])
    # Both of these must own a FOCUSABLE window. prompt_toolkit requires the
    # focused control to exist in the currently-rendered layout; focusing a
    # form input while showing the welcome screen raises "Window does not
    # appear in the layout" and drops the whole app to the --cli fallback.
    help_window = Window(FormattedTextControl(
                             text=lambda: help_content.render(view["topic"]),
                             focusable=True),
                         wrap_lines=True, style="class:help")
    help_view = HSplit([title, Frame(help_window,
                                     title="help — press 0-9 for a topic · Esc closes",
                                     height=D(weight=1)), footer])

    welcome_window = Window(FormattedTextControl(
                                text=lambda: help_content.LOGO + help_content.WELCOME,
                                focusable=True),
                            wrap_lines=True, style="class:help")
    welcome_view = HSplit([title,
                           Frame(welcome_window,
                                 title="welcome — Enter to start · Ctrl-G for help",
                                 height=D(weight=1)),
                           footer])
    picker_view = HSplit([title, Frame(picker_window, title="browse — ↑↓ move · Enter open/pick · U use folder · Esc cancel",
                                       height=D(weight=1)),
                          Window(FormattedTextControl([("class:footer",
                                 " ↑/↓ move · Enter open folder or pick file · U use this folder · Esc cancel ")]),
                                 height=1, style="class:footerbar")])
    def current_view():
        return {"help": help_view, "picker": picker_view,
                "welcome": welcome_view}.get(view["mode"], main_view)

    layout = Layout(DynamicContainer(current_view))

    # ----- key bindings -----
    kb = KeyBindings()
    in_picker = Condition(lambda: view["mode"] == "picker")
    not_picker = Condition(lambda: view["mode"] != "picker")
    not_running = Condition(lambda: not state["running"])
    not_picker_or_running = Condition(lambda: view["mode"] != "picker" and not state["running"])

    @kb.add("c-q")
    def _(e):
        e.app.exit(result=0)

    @kb.add("c-r", filter=not_picker_or_running)
    def _(e):
        run_dock(100, "vina", "Full run (n=100)")

    @kb.add("c-t", filter=not_picker_or_running)
    def _(e):
        start_stream(None, demo=True, title="DEMO (simulated, no GPU)")

    @kb.add("c-y", filter=not_picker_or_running)
    def _(e):
        run_crystal()

    @kb.add("c-a", filter=not_picker)
    def _(e):
        # Undocumented on purpose — a small gallery/easter-egg peek, any time,
        # run or no run. `random` (not a seeded rng) is deliberate: this has
        # no effect on anything scored or reproducible. show_art() sleeps
        # between frames, so it runs on its own thread — this handler fires
        # on the main event loop and must not block it.
        import random as _random  # noqa: PLC0415
        egg = art.maybe_rare()
        if egg:
            append(egg)
        else:
            threading.Thread(
                target=lambda: show_art(_random.randrange(len(art.GALLERY))), daemon=True
            ).start()

    @kb.add("c-b", filter=not_picker)
    def _(e):
        open_picker()

    @kb.add("c-p", filter=not_picker)
    def _(e):
        do_print()

    @kb.add("c-l", filter=not_picker)
    def _(e):
        do_clear()

    @kb.add("c-g", filter=not_picker)
    def _(e):
        toggle_help()

    @kb.add("escape", eager=True)
    def _(e):
        go_main()

    in_welcome = Condition(lambda: view["mode"] == "welcome")
    in_help = Condition(lambda: view["mode"] == "help")

    @kb.add("c-w", filter=not_picker)
    def _(e):
        show_welcome()

    @kb.add("enter", filter=in_welcome)
    def _(e):
        # Leaving the tour is what marks it as seen — so a user who quits from
        # the welcome screen still gets it next time.
        mark_welcomed()
        go_main()

    for _topic_key, _title in help_content.topic_titles():
        @kb.add(_topic_key, filter=in_help)
        def _(e, _k=_topic_key):
            view["topic"] = _k

    in_form = Condition(lambda: view["mode"] == "main")

    @kb.add("tab", filter=not_picker)
    def _(e):
        focus_field(+1)

    @kb.add("s-tab", filter=not_picker)
    def _(e):
        focus_field(-1)

    # Arrows too: on a labelled vertical form this is what people reach for
    # first, and the inputs are single-line so up/down do nothing inside them.
    @kb.add("down", filter=in_form)
    def _(e):
        focus_field(+1)

    @kb.add("up", filter=in_form)
    def _(e):
        focus_field(-1)

    # Emergency stop. Ctrl-C is the instinct in a panic, so bind that as well as
    # a mnemonic; quitting stays on Ctrl-Q.
    @kb.add("c-c", filter=not_picker)
    def _(e):
        stop_run()

    @kb.add("c-k", filter=not_picker)
    def _(e):
        stop_run()

    @kb.add("up", filter=in_picker)
    def _(e):
        picker["idx"] = max(0, picker["idx"] - 1)

    @kb.add("down", filter=in_picker)
    def _(e):
        picker["idx"] = min(len(picker["items"]) - 1, picker["idx"] + 1)

    @kb.add("enter", filter=in_picker)
    def _(e):
        picker_choose()

    @kb.add("u", filter=in_picker)
    @kb.add("U", filter=in_picker)
    def _(e):
        picker_use_dir()

    style = Style.from_dict({
        "titlebar": "bg:#0d3b66 #ffffff bold", "title": "bg:#0d3b66 #ffffff bold",
        "titledim": "bg:#0d3b66 #a9c7e8", "label": "bold #cfe3ff", "input": "bg:#11151c #ffffff",
        "labelfocus": "bold #ffd166",
        "output": "bg:#0a0e14 #b6f0c4", "help": "bg:#0d1830 #d3e2ff", "picker": "bg:#0a0e14 #d3e2ff",
        "pickersel": "bg:#0d3b66 #ffffff bold", "pickerdir": "#7fd0ff", "pickerfile": "#b6f0c4",
        "bar": "#39c0ff", "okbar": "#4ade80", "pct": "bold #ffffff", "stage": "bold #ffd166",
        "verb": "italic #7fd0ff", "artline": "#6fa88e",
        "dim": "#7f8c9b", "err": "#ff6b6b bold", "ok": "#4ade80 bold", "footerbar": "bg:#11151c",
        "footer": "#9fb3c8", "key": "bg:#0d3b66 #ffffff bold", "frame.border": "#2b6cb0",
        "button": "#cfe3ff", "button.focused": "bg:#0d3b66 #ffffff bold",
    })

    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True,
                      mouse_support=True)
    # Focus must match the view we are actually opening on. Focusing a form
    # input while the welcome tour is on screen makes prompt_toolkit reject the
    # layout outright and the app silently degrades to the --cli wizard.
    app.layout.focus(welcome_window if view["mode"] == "welcome" else inputs["peptide"])
    refresh_hint()
    refresh_progress()

    def _tick(_app):
        refresh_progress()
        if view["mode"] == "picker":
            refresh_picker()
    app.before_render += _tick

    def _animating() -> bool:
        """True while the spinner/progress bar/gallery actually need to move."""
        p = progress["p"]
        return state["running"] or (p is not None and not p.finished)

    def _ticker():
        # progress_ctrl/hint are plain FormattedTextControls, not Buffers, so
        # (unlike the output log's TextArea) changing their .text does not
        # auto-invalidate the app — something has to periodically force a
        # redraw for the time-driven spinner/progress-bar/gallery animation
        # to move at all. A flat Application(refresh_interval=0.15) used to
        # do that unconditionally, 24/7, which meant a full-screen redraw
        # ~6.7x/sec forever even with the TUI just sitting open and idle —
        # the process kept measurably burning CPU long after any run (or
        # Ctrl-C, which only stops the run, not the app — Ctrl-Q quits).
        # Tick fast only while something is actually animating; otherwise
        # back off to an occasional idle nudge (picker navigation and typing
        # already invalidate on their own via key-press handling).
        while True:
            time.sleep(0.15 if _animating() else 0.5)
            try:
                app.invalidate()
            except Exception:  # noqa: BLE001 — a missed tick must never kill the app
                pass

    threading.Thread(target=_ticker, daemon=True).start()

    if auto_demo:
        threading.Thread(target=lambda: (time.sleep(0.4),
                         start_stream(None, demo=True, title="DEMO (simulated, no GPU)")), daemon=True).start()
    return app.run() or 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    if "--print" in argv:
        return run_cli_wizard(print_only=True)
    if "--cli" in argv:
        return run_cli_wizard(print_only=False)
    try:
        return run_fullscreen(auto_demo="--demo" in argv)
    except Exception as exc:  # noqa: BLE001
        print(f"(full-screen UI unavailable: {exc}; falling back to --cli)")
        return run_cli_wizard(print_only=False)


if __name__ == "__main__":
    raise SystemExit(main())
