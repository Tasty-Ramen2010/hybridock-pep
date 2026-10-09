# Browser end-to-end and stress suites for the web UI

These drive the real UI in a real browser (Playwright / Chromium) against a running server, the way a person would:
they click, type, upload files and read what is on the screen. They do **not** call the API directly. They are not part
of the Python test suite (`pytest`) or the CI; run them by hand against any server.

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

Machines without a GPU or the sampling environment can still run everything except a sampled Dock: Score is real, and
the Dock failure is checked to be a clear message.
