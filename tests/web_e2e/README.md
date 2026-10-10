# Browser end-to-end and stress suites for the web UI

These drive the real UI in a real browser (Playwright / Chromium) against a running server, the way a person would:
they click, type, upload files and read what is on the screen. They do **not** call the API directly. They are not part
of the Python test suite (`pytest`). The fast ones (01 02 04 06 07 08 10 11 14 20 21 22) run in CI (`.github/workflows/web.yml`) against a live
server and against the static demo site; the long ones that run real dockings are run by hand against any server.

```bash
cd tests/web_e2e
npm install                         # playwright + axe-core (then: npx playwright install chromium, once)
hybridock-pep serve --no-browser &  # or point BASE at any machine, e.g. through an SSH tunnel
BASE=http://127.0.0.1:8000 node run_all.mjs          # everything
BASE=http://127.0.0.1:8000 node run_all.mjs 01 05    # just suites 01 and 05
```

Screenshots and results JSON go to `$E2E_OUT` (default: a `hybridock-e2e` folder in your temp directory). Exit code is non-zero if
anything failed.

| Suite | What it checks |
| --- | --- |
| `01_smoke` | Live status chip and popover, every route, Guided/Expert + theme survive a reload, fonts, no console errors |
| `02_score` | **Real** Score runs: rejects non-PDB / empty files politely, gives a sensible ΔG, the same number twice |
| `03_dock_engine` | Run example: a result if the docking engine is installed, otherwise a plain-language "engine not installed" card |
| `04_setup_fuzz` | Protein search / RCSB download / bad uploads, 20+ peptide inputs (length 3–30, letters, unicode, HTML), box limits, Expert command preview |
| `05_concurrency` | 12 tabs at once, 300 route changes, 3 simultaneous runs (one runs, the others are told), 24 real runs from 4 contending tabs |
| `06_monkey` | 400 route changes (heap growth), 200 theme/mode flips, 700 random clicks / keys / drags / junk text, no uncaught errors or XSS |
| `07_accessibility` | axe-core on every screen in light and dark (no serious or critical violations), dialogs included |
| `08_uploads`, `08b_big_structure` | Up to a 33 MB / 420k-atom upload: loads or is refused politely, the page stays responsive, the 3D view still animates |
| `09_faults` | Server frozen, killed and restarted mid-run, browser offline, reload mid-run. Needs `FAULT_SSH` and `FAULT_START_CMD` (skipped otherwise) |
| `10_history` | 40 real runs: History keeps 30, the Home strip shows 4, the lifetime counter, reopen after reload, Clear all |
| `11_keyboard_mobile` | Keyboard-only walkthrough (skip link, focus, dialogs, arrow keys on the box); phone viewport: no horizontal scroll, 44 px tap targets |
| `12_real_dock` | A real Dock that re-scores saved poses (`--input-poses`), results, downloads. Needs `E2E_INPUT_POSES` (skipped otherwise) |

| `13_example_run` | The one-click "Run example": a **real** Dock end to end (slow on CPU), honest progress, complete result |
| `20_topbar_home_help` | Every top-bar control (Guided/Expert, status chip, 4 accents, theme, name/avatar), Home, Help and History dialogs |
| `21_setup_every_control` | All four setup steps: 8 proteins, search, upload, 20+ peptide inputs, slider, typed coordinates, the 3D box (drag, resize, click the protein, keyboard), blind mode, thoroughness, **every Expert setting's effect on the command**, validation messages, and each protein's suggested site |
| `22_compare_score_controls` | Every Compare control (both menus, upload, 3D site editor, same-protein and invalid-peptide guards) and Score control (uploads, drag-and-drop, mismatch warnings) |
| `23_guide` | The in-app guide: four-destination nav, contents, deep links, in-page links, Copy buttons, every screenshot loads and has a caption, no raw Markdown on the page, phone layout |
| `24_env_states` | What the app says for each kind of computer (Apple Metal, NVIDIA, AMD, CPU-only), the first-prediction notice, and the "setup isn't finished" notice, by replaying `/api/env`. Live server only |
| `30_dock_matrix` | A matrix of **real Dock runs** (Expert options, peptides, sites, every built-in protein, uploads, thoroughness) each followed by a full Results-screen and downloads check. `CASES=a,b BASE=... node 30_dock_matrix.mjs`; `--list` shows the cases |
| `31_compare_matrix` | **Real Compare runs** (two dockings each), then the whole comparison result: ΔΔG, interval, verdict, ΔG cards, 3D tabs, command |
| `32_run_control` | Real runs under real behaviour: wander the app mid-run, reload, close the tab and come back after it finished, Stop and run again, server restart (the last needs `FAULT_SSH` / `FAULT_START_CMD`) |

A real Dock on a CPU-only machine takes 4 to 10 minutes per Quick run after a one-time setup (RAPiDock builds its SO(3) tables and
downloads the 2.5 GB ESM-2 weights on the very first run). Run several servers on different ports, each with its own `CASES`, to use
more cores; give each run one Vina worker (`HYBRIDOCK_VINA_WORKERS=1`) when you do, or three parallel scorings can exhaust memory.

Machines without a GPU or the sampling environment can still run everything except a sampled Dock: Score is real, and
the Dock failure is checked to be a clear message.

## Regenerating the guide's screenshots

`guide_shots.mjs` drives a **live** server through the real UI and writes `docs/guide-img/*.png` (it starts one real Quick dock and,
optionally, one real Compare, so allow 15 to 30 minutes on a CPU-only machine). `ONLY=home,menu` takes just some of them.
Afterwards run `python3 scripts/build_web.py`, which copies them into the app (CI fails if you forget).
