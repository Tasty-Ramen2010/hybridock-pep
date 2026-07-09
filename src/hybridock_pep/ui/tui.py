"""HybriDock-Pep terminal UI — a btop/nvtop-style front-end for the docking pipeline.

Cross-platform (Windows / macOS / Linux) via prompt_toolkit — no curses, no browser. Ways to run:

    hybridock-tui            # full-screen interactive UI (default)
    hybridock-tui --demo     # full-screen UI, auto-run a synthetic pipeline (no GPU — watch the bars)
    hybridock-tui --cli      # plain step-by-step wizard (no full-screen; SSH / dumb terminals)
    hybridock-tui --print    # build & print the `hybridock-pep dock` command, run nothing

Run modes (buttons + keys): Full (n=100, vina+ad4, MM-GBSA), Half (n=50), Quick (n=20, vina),
Selectivity ΔΔG (target vs off-target), and Demo. A built-in file/folder browser (Browse button /
Ctrl-B) fills any path field; you can also just drag a file onto a path field. Controls are
mouse-clickable buttons + universal Ctrl-key accelerators (no Mac/Windows-only function keys).
The layout resizes with the terminal. Dependency-light (prompt_toolkit only).
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")

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


def _valid_receptor(v):
    v = v.strip()
    if not v:
        return "receptor PDB path is required"
    return None if Path(v).expanduser().is_file() else "file not found"


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


def _valid_scoring(v):
    ok = {"vina", "ad4"}
    toks = [t.strip() for t in v.split(",") if t.strip()]
    return None if toks and all(t in ok for t in toks) else "use vina, ad4, or vina,ad4"


def _valid_dir(v):
    return None if v.strip() else "output dir is required"


def _optional(fn):
    def f(v):
        return None if not v.strip() else fn(v)
    return f


FIELDS = [
    FormField("peptide", "Peptide sequence", "LISDAELEAIFEADC",
              "One-letter amino-acid sequence to dock (3–30 residues).", _valid_peptide),
    FormField("receptor", "Target receptor PDB", "data/pdbs/1T2D_receptor.pdb",
              "On-target receptor .pdb — drag the file here or use Browse (Ctrl-B).",
              _valid_receptor, is_path=True),
    FormField("site", "Target site  x y z", "31.9 17.5 9.5",
              "On-target box center (Å) — three numbers.", _valid_site),
    FormField("box", "Target box (Å)", "20",
              "On-target cubic box edge (Å). 30 for 12-mers+.", _posint(10, 60)),
    FormField("n_samples", "N samples", "100",
              "RAPiDock diffusion poses to generate (per receptor).", _posint(1, 500)),
    FormField("scoring", "Scoring", "vina,ad4",
              "Physics rescoring: vina, ad4, or vina,ad4.", _valid_scoring),
    FormField("refine_topk", "Refine top-K (MM-GBSA)", "0",
              "MM-GBSA on top-K clusters (0 = off; dock mode only).", _posint(0, 50)),
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
DOCK_KEYS = ["peptide", "receptor", "site", "box", "n_samples", "scoring", "refine_topk", "output_dir"]
SEL_KEYS = DOCK_KEYS + ["offtarget_receptor", "offtarget_site", "offtarget_box"]


def clean_dropped_path(s: str) -> str:
    """Normalise a path a terminal produced from a file-drop (quotes, file://, escaped spaces)."""
    s = s.strip()
    for _ in range(2):  # may be quote-wrapped AND file://-prefixed, either order
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            s = s[1:-1].strip()
        if s.startswith("file://"):
            s = s[7:]
    return s.replace("\\ ", " ").replace("\\~", "~").replace("\\(", "(").replace("\\)", ")").strip()


def build_dock_command(v, exe="hybridock-pep"):
    x, y, z = v["site"].split()
    cmd = [exe, "dock", "--peptide", v["peptide"].strip().upper(),
           "--receptor", str(Path(v["receptor"]).expanduser()),
           "--site", x, y, z, "--box", v["box"].strip(),
           "--n-samples", v["n_samples"].strip(), "--scoring", v["scoring"].strip(),
           "--output-dir", v["output_dir"].strip()]
    if int(v["refine_topk"] or 0) > 0:
        cmd += ["--refine-topk", v["refine_topk"].strip()]
    return cmd


def build_selectivity_command(v, exe="hybridock-pep"):
    tx, ty, tz = v["site"].split()
    ox, oy, oz = v["offtarget_site"].split()
    return [exe, "selectivity", "--peptide", v["peptide"].strip().upper(),
            "--target-receptor", str(Path(v["receptor"]).expanduser()),
            "--target-site", tx, ty, tz, "--target-box", v["box"].strip(),
            "--offtarget-receptor", str(Path(v["offtarget_receptor"]).expanduser()),
            "--offtarget-site", ox, oy, oz, "--offtarget-box", v["offtarget_box"].strip(),
            "--n-samples", v["n_samples"].strip(), "--scoring", v["scoring"].strip(),
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
        if "stage 3.6" in low or ("affinity" in low and "δg" in low) or "gbsa" in low or "stage 3.5" in low:
            self._enter("affinity")
        elif "stage 3" in low or "cluster" in low:
            self._enter("cluster")
        elif "stage 2" in low or ("scor" in low and "fail" not in low):
            self._enter("score")
        elif "stage 1.5" in low or "minimi" in low:
            self._enter("min")
        elif "stage 1" in low or "rapidock" in low or "sampl" in low or "pose" in low:
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


HELP_TEXT = """\
 HybriDock-Pep — terminal UI

 WHAT IT DOES
   peptide + receptor → AI diffusion poses (RAPiDock) → physics rescoring
   → ranked ΔG (kcal/mol), best pose, clusters, plots. Selectivity mode also
   docks an off-target and reports ΔΔG (does the peptide prefer the target?).

 RUN MODES  (buttons, top row)
   Full ▶       n=100, vina+ad4, MM-GBSA top-10   (the real thing)
   Half         n=50,  vina+ad4                    (faster)
   Quick        n=20,  vina                         (fastest sanity check)
   Selectivity  target vs off-target ΔΔG (fill the 3 Off-target fields)
   Demo ▷       simulated full run — no GPU, just watch the progress bar

 FILLING THE FORM
   • Tab / Shift-Tab move; every field validates live (✓ / ✗).
   • Path fields: DRAG a file on, or press Browse (Ctrl-B) for a file/folder picker.
   • Off-target fields are only needed for Selectivity.

 CONTROLS  (click buttons, or these keys — work on every OS, no function keys)
   Ctrl-R Full   Ctrl-T Demo   Ctrl-B Browse   Ctrl-P Print
   Ctrl-L Clear  Ctrl-G Help   Ctrl-Q Quit     Tab move

 FILE BROWSER
   ↑/↓ move · Enter open folder / pick file · U use current folder · Esc cancel

 Press Ctrl-G or Esc to close this help.
"""


def demo_lines(n=100):
    yield ("Stage 1: RAPiDock-Reloaded sampling — device: mps", 0.15)
    for i in range(1, n + 1):
        yield (f"  pose {i}/{n} sampled", 0.010)
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
    if shutil.which(cmd[0]) is None:
        print(f"  ✗ '{cmd[0]}' not on PATH — run `conda activate score-env` first.")
        return 127
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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

    output = TextArea(text="\n".join(state["lines"]), read_only=True, scrollbar=True,
                      style="class:output", focus_on_click=True, wrap_lines=False)

    def append(line):
        state["lines"].append(line.rstrip("\n"))
        output.text = "\n".join(state["lines"][-2000:])
        output.buffer.cursor_position = len(output.text)

    hint = FormattedTextControl(text="")

    def refresh_hint():
        errs = validate(vals(), DOCK_KEYS)
        hint.text = ([("class:err", "  ✗ " + errs[0])] if errs
                     else [("class:ok", "  ✓ ready — Full ▶ (Ctrl-R) or Demo ▷ (Ctrl-T)")])

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
        frac = p.fraction()
        cls = "class:okbar" if p.finished else "class:bar"
        cnt = p.counter()
        progress_ctrl.text = ([("class:dim", "  "), (cls, _bar(frac, 34)),
                               ("class:pct", f" {int(frac*100):3d}% "), ("class:dim", spin), ("", "\n"),
                               ("class:dim", "  stage: "), ("class:stage", p.label()),
                               ("class:dim", f"   {cnt}" if cnt else ""),
                               ("class:dim", f"   ·  {p.elapsed()}s")])

    # ----- run machinery -----
    def start_stream(cmd, demo=False, title=""):
        if state["running"]:
            append("… a run is already in progress")
            return
        state["running"], state["rc"] = True, None
        progress["p"] = PipelineProgress()
        append("")
        append(f"▶ {title}" if title else "▶ run")
        if not demo:
            append("$ " + " ".join(shlex.quote(c) for c in cmd))

        def worker():
            try:
                if demo:
                    for text, delay in demo_lines(int(vals()["n_samples"] or 100)):
                        progress["p"].feed(text)
                        append(text)
                        time.sleep(delay)
                    state["rc"] = 0
                else:
                    if shutil.which(cmd[0]) is None:
                        append(f"✗ '{cmd[0]}' not on PATH — run `conda activate score-env` first.")
                        state["rc"] = 127
                        return
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1)
                    for line in proc.stdout:
                        progress["p"].feed(line)
                        append(line)
                    state["rc"] = proc.wait()
                progress["p"].finished = True
                append(f"── finished (exit {state['rc']}) ──")
            except Exception as exc:  # noqa: BLE001
                append(f"✗ error: {exc}")
                state["rc"] = 1
            finally:
                state["running"] = False

        threading.Thread(target=worker, daemon=True).start()

    def run_dock(n, scoring, refine, title):
        inputs["n_samples"].text, inputs["scoring"].text, inputs["refine_topk"].text = str(n), scoring, str(refine)
        errs = validate(vals(), DOCK_KEYS)
        if errs:
            append("")
            append("✗ fix before running:")
            [append("    - " + e) for e in errs]
            return
        start_stream(build_dock_command(vals()), title=title)

    def run_selectivity():
        errs = validate(vals(), SEL_KEYS)
        if errs:
            append("")
            append("✗ Selectivity needs the 3 Off-target fields filled:")
            [append("    - " + e) for e in errs]
            return
        start_stream(build_selectivity_command(vals()), title="Selectivity ΔΔG (target vs off-target)")

    def do_print():
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
    view = {"mode": "main"}  # main | help | picker

    def toggle_help():
        view["mode"] = "main" if view["mode"] == "help" else "help"

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
        B("Full ▶", lambda: run_dock(100, "vina,ad4", 10, "Full run (n=100, vina+ad4, MM-GBSA)"), 10),
        B("Half", lambda: run_dock(50, "vina,ad4", 0, "Half run (n=50, vina+ad4)"), 8),
        B("Quick", lambda: run_dock(20, "vina", 0, "Quick run (n=20, vina)"), 9),
        B("Selectivity ⚖", run_selectivity, 16),
        B("Demo ▷", lambda: start_stream(None, demo=True, title="DEMO (simulated, no GPU)"), 10),
    ], padding=1, height=1)
    tool_row = VSplit([
        B("Browse 📁", open_picker, 12),
        B("Print", do_print, 9),
        B("Help ?", toggle_help, 10),
        B("Clear", do_clear, 9),
        B("Quit ✕", lambda: get_app().exit(result=0), 10),
    ], padding=1, height=1)

    form_rows = [VSplit([
        Window(FormattedTextControl(lambda f=f: [("class:label", f"{f.label:>28} ")]),
               width=29, height=1, dont_extend_width=True),
        inputs[f.key],
    ], height=1) for f in FIELDS]
    form = HSplit(form_rows + [Window(hint, height=1)])

    title = Window(FormattedTextControl(
        [("class:title", "  HybriDock-Pep "), ("class:titledim", "· peptide → poses → ΔG → selectivity ")]),
        height=1, align=WindowAlign.LEFT, style="class:titlebar")
    footer = Window(FormattedTextControl(
        [("class:key", " Ctrl-R "), ("class:footer", "Full "), ("class:key", " Ctrl-T "), ("class:footer", "Demo "),
         ("class:key", " Ctrl-B "), ("class:footer", "Browse "), ("class:key", " Ctrl-G "), ("class:footer", "Help "),
         ("class:key", " Ctrl-Q "), ("class:footer", "Quit "),
         ("class:footer", " · Tab moves · drag a .pdb onto a path field · click any button ")]),
        height=1, style="class:footerbar")

    main_view = HSplit([title, Frame(form, title="inputs"),
                        Frame(Window(progress_ctrl, height=2), title="progress"),
                        Frame(output, title="output", height=D(min=3, weight=1)),
                        run_row, tool_row, footer])
    help_view = HSplit([title, Frame(Window(FormattedTextControl(text=HELP_TEXT), wrap_lines=True,
                                            style="class:help"), title="help — Ctrl-G / Esc to close",
                                     height=D(weight=1)), footer])
    picker_view = HSplit([title, Frame(picker_window, title="browse — ↑↓ move · Enter open/pick · U use folder · Esc cancel",
                                       height=D(weight=1)),
                          Window(FormattedTextControl([("class:footer",
                                 " ↑/↓ move · Enter open folder or pick file · U use this folder · Esc cancel ")]),
                                 height=1, style="class:footerbar")])

    def current_view():
        return {"help": help_view, "picker": picker_view}.get(view["mode"], main_view)

    layout = Layout(DynamicContainer(current_view))

    # ----- key bindings -----
    kb = KeyBindings()
    in_picker = Condition(lambda: view["mode"] == "picker")
    not_picker = Condition(lambda: view["mode"] != "picker")

    @kb.add("c-q")
    def _(e):
        e.app.exit(result=0)

    @kb.add("c-r", filter=not_picker)
    def _(e):
        run_dock(100, "vina,ad4", 10, "Full run (n=100, vina+ad4, MM-GBSA)")

    @kb.add("c-t", filter=not_picker)
    def _(e):
        start_stream(None, demo=True, title="DEMO (simulated, no GPU)")

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
        view["mode"] = "main"

    @kb.add("tab", filter=not_picker)
    def _(e):
        e.app.layout.focus_next()

    @kb.add("s-tab", filter=not_picker)
    def _(e):
        e.app.layout.focus_previous()

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
        "output": "bg:#0a0e14 #b6f0c4", "help": "bg:#0d1830 #d3e2ff", "picker": "bg:#0a0e14 #d3e2ff",
        "pickersel": "bg:#0d3b66 #ffffff bold", "pickerdir": "#7fd0ff", "pickerfile": "#b6f0c4",
        "bar": "#39c0ff", "okbar": "#4ade80", "pct": "bold #ffffff", "stage": "bold #ffd166",
        "dim": "#7f8c9b", "err": "#ff6b6b bold", "ok": "#4ade80 bold", "footerbar": "bg:#11151c",
        "footer": "#9fb3c8", "key": "bg:#0d3b66 #ffffff bold", "frame.border": "#2b6cb0",
        "button": "#cfe3ff", "button.focused": "bg:#0d3b66 #ffffff bold",
    })

    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True,
                      mouse_support=True, refresh_interval=0.15)
    app.layout.focus(inputs["peptide"])
    refresh_hint()
    refresh_progress()

    def _tick(_app):
        refresh_progress()
        if view["mode"] == "picker":
            refresh_picker()
    app.before_render += _tick

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
