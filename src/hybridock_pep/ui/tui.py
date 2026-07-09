"""HybriDock-Pep terminal UI — a btop/nvtop-style front-end for the docking pipeline.

Cross-platform (Windows / macOS / Linux) via prompt_toolkit — no curses, no browser. Ways to run:

    hybridock-tui            # full-screen interactive UI (default)
    hybridock-tui --demo     # full-screen UI, auto-run a synthetic pipeline (no GPU needed — watch the bars)
    hybridock-tui --cli      # plain step-by-step wizard (no full-screen; great over SSH / dumb terminals)
    hybridock-tui --print    # just build & print the `hybridock-pep dock` command, run nothing

It never re-implements the science: it collects + validates inputs, then shells out to the real
`hybridock-pep dock` CLI and streams its output live, turning the pipeline's `Stage N …` / `x/N poses`
log lines into a live progress bar. Controls are mouse-clickable buttons + universal Ctrl-key accelerators
(no Mac/Windows-only function keys). Drag a file onto a path field to fill it. Resizes with the terminal.

Dependency-light (prompt_toolkit only).
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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


def _valid_peptide(v):
    v = v.strip().upper()
    if not v:
        return "peptide is required"
    bad = sorted(set(v) - AA)
    if bad:
        return f"non-amino-acid letter(s): {' '.join(bad)}"
    if len(v) < 3:
        return "too short (min 3 residues)"
    if len(v) > 30:
        return "too long (max 30 residues)"
    return None


def _valid_receptor(v):
    v = v.strip()
    if not v:
        return "receptor PDB path is required"
    if not Path(v).expanduser().is_file():
        return "file not found"
    return None


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


FIELDS = [
    FormField("peptide", "Peptide sequence", "LISDAELEAIFEADC",
              "One-letter amino-acid sequence to dock (3–30 residues).", _valid_peptide),
    FormField("receptor", "Receptor PDB", "data/pdbs/1T2D_receptor.pdb",
              "Path to the receptor .pdb — or just drag the file onto this field.", _valid_receptor, is_path=True),
    FormField("site", "Binding site  x y z", "31.9 17.5 9.5",
              "Docking box center in Å — three numbers separated by spaces.", _valid_site),
    FormField("box", "Box size (Å)", "20",
              "Cubic search-box edge (Å). Use 30 for 12-mers+ to contain the full peptide.", _posint(10, 60)),
    FormField("n_samples", "N samples", "100",
              "How many RAPiDock diffusion poses to generate. Fewer = faster, noisier.", _posint(1, 500)),
    FormField("scoring", "Scoring", "vina,ad4",
              "Physics rescoring: 'vina', 'ad4', or 'vina,ad4' (both, recommended).", _valid_scoring),
    FormField("refine_topk", "Refine top-K (MM-GBSA)", "0",
              "Run MM-GBSA on the top-K cluster centroids (0 = off; 5–10 = sharper, slower).", _posint(0, 50)),
    FormField("output_dir", "Output dir", "runs/tui_run",
              "Where results (ranked CSV, best pose, plots, metadata) are written.", _valid_dir, is_path=True),
]


def clean_dropped_path(s: str) -> str:
    """Normalise a path pasted by a terminal file-drop (quotes, file://, escaped spaces)."""
    s = s.strip()
    for _ in range(2):  # a dropped path may be quote-wrapped AND file://-prefixed, in either order
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
            s = s[1:-1].strip()
        if s.startswith("file://"):
            s = s[7:]
    s = s.replace("\\ ", " ").replace("\\~", "~").replace("\\(", "(").replace("\\)", ")")
    return s.strip()


def build_command(values, exe="hybridock-pep"):
    x, y, z = values["site"].split()
    cmd = [exe, "dock",
           "--peptide", values["peptide"].strip().upper(),
           "--receptor", str(Path(values["receptor"]).expanduser()),
           "--site", x, y, z,
           "--box", values["box"].strip(),
           "--n-samples", values["n_samples"].strip(),
           "--scoring", values["scoring"].strip(),
           "--output-dir", values["output_dir"].strip()]
    if int(values["refine_topk"] or 0) > 0:
        cmd += ["--refine-topk", values["refine_topk"].strip()]
    return cmd


def validate_all(values):
    return [f"{f.label}: {m}" for f in FIELDS if (m := f.validate(values.get(f.key, "")))]


# --------------------------------------------------------------------------- #
#  Pipeline progress model — parses the real CLI's log lines into a %.
# --------------------------------------------------------------------------- #

# key, label, weight (fractions of the whole run)
STAGES = [
    ("sample",   "AI sampling (RAPiDock)",   0.50),
    ("min",      "minimize poses",           0.05),
    ("score",    "physics rescoring",        0.25),
    ("cluster",  "RMSD clustering",          0.05),
    ("affinity", "affinity / MM-GBSA",       0.15),
]
_ORDER = [k for k, _, _ in STAGES]
_WEIGHT = {k: w for k, _, w in STAGES}
_LABEL = {k: l for k, l, _ in STAGES}
_COUNT = re.compile(r"(\d+)\s*(?:/|of)\s*(\d+)")
_PCT = re.compile(r"(\d+)\s*%")


class PipelineProgress:
    def __init__(self):
        self.stage = None
        self.cur = 0
        self.total = 0
        self.done: set[str] = set()
        self.finished = False
        self.t0 = time.time()

    def _enter(self, key):
        if key == self.stage:
            return
        # mark everything up to (not including) the new stage as done
        if key in _ORDER:
            for k in _ORDER[:_ORDER.index(key)]:
                self.done.add(k)
        self.stage = key
        self.cur, self.total = 0, 0

    def feed(self, line: str):
        low = line.lower()
        if any(w in low for w in ("run_metadata", "best pose", "results written", "── finished", "ranked_poses")):
            self.finished = True
            for k in _ORDER:
                self.done.add(k)
            return
        # stage detection (order matters: check most specific first)
        if "stage 3.6" in low or ("affinity" in low and "δg" in low) or "mm-gbsa" in low or "gbsa" in low or "stage 3.5" in low:
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
        m = _COUNT.search(line)
        if m:
            self.cur, self.total = int(m.group(1)), int(m.group(2))
        elif (p := _PCT.search(line)):
            self.total, self.cur = 100, int(p.group(1))

    def fraction(self) -> float:
        if self.finished:
            return 1.0
        frac = sum(_WEIGHT[k] for k in self.done)
        if self.stage and self.stage not in self.done:
            sub = (self.cur / self.total) if self.total else 0.0
            frac += _WEIGHT.get(self.stage, 0.0) * min(sub, 1.0)
        return max(0.0, min(frac, 1.0))

    def counter(self) -> str:
        return f"{self.cur}/{self.total}" if self.total else ""

    def label(self) -> str:
        if self.finished:
            return "done"
        return _LABEL.get(self.stage, "starting…")

    def elapsed(self) -> int:
        return int(time.time() - self.t0)


HELP_TEXT = """\
 HybriDock-Pep — terminal UI

 WHAT IT DOES
   peptide + receptor  →  AI diffusion poses (RAPiDock)  →  physics rescoring
   →  ranked ΔG (kcal/mol), best pose, clusters, plots.

 FILLING THE FORM
   • Tab / Shift-Tab move between fields; every field validates live (✓ / ✗).
   • Peptide   one-letter AA, 3–30 residues (e.g. ETFSDLWKLLPE).
   • Receptor  path to a .pdb — or DRAG THE FILE onto the field.
   • Site      three numbers = box center (Å). Box 30 for long peptides.
   • Refine    0 = fast; 5–10 = MM-GBSA on top clusters (slower, sharper).

 CONTROLS  (click the buttons, or use these keys — they work everywhere)
   Ctrl-R  Run        Ctrl-T  Demo run (no GPU)      Ctrl-G  Help
   Ctrl-P  Print cmd  Ctrl-L  Clear log              Ctrl-Q  Quit
   Mouse click works on every button; the layout resizes with your terminal.

 NO GPU / just looking?
   Press Ctrl-T (or the Demo button) to watch a full simulated run drive the
   progress bar — or use --print to see the exact CLI, --cli for an SSH wizard.

 Press Ctrl-G or Esc to close this help.
"""


# --------------------------------------------------------------------------- #
#  Demo stream — synthetic pipeline output so the UI is testable without a GPU
# --------------------------------------------------------------------------- #

def demo_lines(n=100):
    yield ("Stage 1: RAPiDock-Reloaded sampling — device: mps", 0.15)
    for i in range(1, n + 1):
        yield (f"  pose {i}/{n} sampled", 0.012)
    yield (f"Stage 1 complete: {n} poses parsed", 0.1)
    yield ("Stage 1.5: OpenMM minimization", 0.1)
    yield (f"Stage 1.5 complete: {n} poses minimized", 0.1)
    yield ("Stage 2: physics rescoring (vina + ad4)", 0.1)
    for i in range(1, n + 1):
        yield (f"  scored {i}/{n} poses", 0.008)
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
    for f in FIELDS:
        print(f"\n  {f.label}\n    {f.help}")
        while True:
            ans = prompt(f"  {f.key} > ", default=f.default)
            if f.is_path:
                ans = clean_dropped_path(ans)
            msg = f.validate(ans)
            if msg is None:
                values[f.key] = ans
                break
            print(f"    ✗ {msg}")
    cmd = build_command(values)
    print("\n  Command:\n    " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if print_only:
        return 0
    if prompt("  Run it now? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("  Not run.")
        return 0
    exe = shutil.which(cmd[0])
    if exe is None:
        print(f"  ✗ '{cmd[0]}' not on PATH — run `conda activate score-env` first.")
        return 127
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write("  " + line)
    return proc.wait()


# --------------------------------------------------------------------------- #
#  Full-screen interactive UI
# --------------------------------------------------------------------------- #

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _bar(frac, width):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def run_fullscreen(auto_demo=False):
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.layout.containers import DynamicContainer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, Frame, Label, TextArea

    progress = {"p": None}          # active PipelineProgress or None
    state = {"running": False, "rc": None, "lines": [
        "Ready. Fill the form, then press the Run button (or Ctrl-R).",
        "No GPU handy? Press Demo (Ctrl-T) to watch a full simulated run.",
        "Tip: drag a .pdb file straight onto the Receptor field.",
    ]}
    inputs = {}

    for f in FIELDS:
        inputs[f.key] = TextArea(text=f.default, multiline=False, height=1, style="class:input", prompt=" ")

    def vals():
        return {k: ta.text for k, ta in inputs.items()}

    # ---- output log ----
    output = TextArea(text="\n".join(state["lines"]), read_only=True, scrollbar=True,
                      style="class:output", focus_on_click=True, wrap_lines=False)

    def append(line):
        state["lines"].append(line.rstrip("\n"))
        output.text = "\n".join(state["lines"][-2000:])
        output.buffer.cursor_position = len(output.text)

    # ---- live hint (validation) ----
    hint = FormattedTextControl(text="")

    def refresh_hint():
        errs = validate_all(vals())
        hint.text = [("class:err", "  ✗ " + errs[0])] if errs else [("class:ok", "  ✓ inputs valid — press Run (Ctrl-R)")]

    # path-cleaning + hint on every field change
    _busy = {"on": False}

    def make_on_change(f):
        def _cb(_=None):
            if _busy["on"]:
                return
            if f.is_path:
                cur = inputs[f.key].text
                cleaned = clean_dropped_path(cur)
                if cleaned != cur:
                    _busy["on"] = True
                    inputs[f.key].text = cleaned
                    inputs[f.key].buffer.cursor_position = len(cleaned)
                    _busy["on"] = False
            refresh_hint()
        return _cb

    for f in FIELDS:
        inputs[f.key].buffer.on_text_changed += make_on_change(f)

    # ---- progress panel ----
    progress_ctrl = FormattedTextControl(text="")

    def refresh_progress():
        p = progress["p"]
        if p is None:
            progress_ctrl.text = [("class:dim", "  idle — no run in progress")]
            return
        spin = "" if p.finished else _SPINNER[int(time.time() * 10) % len(_SPINNER)]
        frac = p.fraction()
        cls = "class:okbar" if p.finished else "class:bar"
        line1 = [("class:dim", "  "), (cls, _bar(frac, 34)),
                 ("class:pct", f" {int(frac * 100):3d}% "), ("class:dim", spin)]
        cnt = p.counter()
        line2 = [("class:dim", "  stage: "), ("class:stage", p.label()),
                 ("class:dim", f"   {cnt}" if cnt else ""),
                 ("class:dim", f"   ·  {p.elapsed()}s elapsed")]
        progress_ctrl.text = line1 + [("", "\n")] + line2

    # ---- run machinery ----
    def start_stream(cmd_or_demo, demo=False):
        if state["running"]:
            return
        if not demo:
            errs = validate_all(vals())
            if errs:
                append("")
                append("✗ Fix before running:")
                for e in errs:
                    append("    - " + e)
                return
        state["running"] = True
        state["rc"] = None
        progress["p"] = PipelineProgress()
        append("")
        if demo:
            append("▶ DEMO RUN (simulated pipeline — no GPU used)")
        else:
            append("$ " + " ".join(shlex.quote(c) for c in cmd_or_demo))
            append("running…")

        def worker():
            try:
                if demo:
                    for text, delay in demo_lines(int(vals()["n_samples"] or 100)):
                        progress["p"].feed(text)
                        append(text)
                        time.sleep(delay)
                    state["rc"] = 0
                else:
                    exe = shutil.which(cmd_or_demo[0])
                    if exe is None:
                        append(f"✗ '{cmd_or_demo[0]}' not on PATH — run `conda activate score-env` first.")
                        state["rc"] = 127
                        return
                    proc = subprocess.Popen(cmd_or_demo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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

    def do_run():
        start_stream(build_command(vals()), demo=False)

    def do_demo():
        start_stream(None, demo=True)

    def do_print():
        errs = validate_all(vals())
        if errs:
            append("✗ " + errs[0])
            return
        append("")
        append("$ " + " ".join(shlex.quote(c) for c in build_command(vals())))
        append("(printed only — press Run / Ctrl-R to execute)")

    def do_clear():
        state["lines"] = []
        output.text = ""

    help_on = {"v": False}

    def toggle_help():
        help_on["v"] = not help_on["v"]

    # ---- buttons ----
    buttons = VSplit([
        Button("Run ▶", handler=do_run, width=10),
        Button("Demo ▷", handler=do_demo, width=11),
        Button("Print", handler=do_print, width=9),
        Button("Help ?", handler=toggle_help, width=10),
        Button("Clear", handler=do_clear, width=9),
        Button("Quit ✕", handler=lambda: get_app().exit(result=0), width=10),
    ], padding=1, height=1)

    # ---- form ----
    form_rows = []
    for f in FIELDS:
        form_rows.append(VSplit([
            Window(FormattedTextControl(lambda f=f: [("class:label", f"{f.label:>22} ")]),
                   width=23, height=1, dont_extend_width=True),
            inputs[f.key],
        ], height=1))
    form = HSplit(form_rows + [Window(hint, height=1)])

    title = Window(FormattedTextControl(
        [("class:title", "  HybriDock-Pep "), ("class:titledim", "· dock a peptide → calibrated ΔG  ")]),
        height=1, align=WindowAlign.LEFT, style="class:titlebar")

    footer = Window(FormattedTextControl(
        [("class:key", " Ctrl-R "), ("class:footer", "Run  "),
         ("class:key", " Ctrl-T "), ("class:footer", "Demo  "),
         ("class:key", " Ctrl-G "), ("class:footer", "Help  "),
         ("class:key", " Ctrl-L "), ("class:footer", "Clear  "),
         ("class:key", " Ctrl-Q "), ("class:footer", "Quit  "),
         ("class:footer", " · Tab moves · drag a .pdb onto Receptor · click any button ")]),
        height=1, style="class:footerbar")

    main_view = HSplit([
        title,
        Frame(form, title="inputs"),
        Frame(Window(progress_ctrl, height=2), title="progress"),
        Frame(output, title="output", height=D(min=4, weight=1)),
        buttons,
        footer,
    ])

    help_view = HSplit([
        title,
        Frame(Window(FormattedTextControl(text=HELP_TEXT), wrap_lines=True, style="class:help"),
              title="help — Ctrl-G or Esc to close", height=D(weight=1)),
        footer,
    ])

    from prompt_toolkit.application.current import get_app

    layout = Layout(DynamicContainer(lambda: help_view if help_on["v"] else main_view))

    kb = KeyBindings()

    @kb.add("c-q")
    def _(e):
        e.app.exit(result=0)

    @kb.add("c-r")
    def _(e):
        do_run()

    @kb.add("c-t")
    def _(e):
        do_demo()

    @kb.add("c-p")
    def _(e):
        do_print()

    @kb.add("c-l")
    def _(e):
        do_clear()

    @kb.add("c-g")
    @kb.add("escape", eager=True)
    def _(e):
        toggle_help()

    @kb.add("tab")
    def _(e):
        e.app.layout.focus_next()

    @kb.add("s-tab")
    def _(e):
        e.app.layout.focus_previous()

    style = Style.from_dict({
        "titlebar": "bg:#0d3b66 #ffffff bold",
        "title": "bg:#0d3b66 #ffffff bold",
        "titledim": "bg:#0d3b66 #a9c7e8",
        "label": "bold #cfe3ff",
        "input": "bg:#11151c #ffffff",
        "output": "bg:#0a0e14 #b6f0c4",
        "help": "bg:#0d1830 #d3e2ff",
        "bar": "#39c0ff",
        "okbar": "#4ade80",
        "pct": "bold #ffffff",
        "stage": "bold #ffd166",
        "dim": "#7f8c9b",
        "err": "#ff6b6b bold",
        "ok": "#4ade80 bold",
        "footerbar": "bg:#11151c",
        "footer": "#9fb3c8",
        "key": "bg:#0d3b66 #ffffff bold",
        "frame.border": "#2b6cb0",
        "button": "#cfe3ff",
        "button.focused": "bg:#0d3b66 #ffffff bold",
    })

    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True,
                      mouse_support=True, refresh_interval=0.15)
    app.layout.focus(inputs[FIELDS[0].key])
    refresh_hint()
    refresh_progress()

    # keep the progress panel live
    def _tick():
        refresh_progress()
    app.before_render += lambda _app: _tick()

    if auto_demo:
        def _kick():
            time.sleep(0.4)
            do_demo()
        threading.Thread(target=_kick, daemon=True).start()

    return app.run() or 0


# --------------------------------------------------------------------------- #

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
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        print(f"(full-screen UI unavailable: {exc}; falling back to --cli)")
        return run_cli_wizard(print_only=False)


if __name__ == "__main__":
    raise SystemExit(main())
