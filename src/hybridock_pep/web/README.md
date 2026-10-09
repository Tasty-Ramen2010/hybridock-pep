# The browser UI (`hybridock-pep serve`)

`index.html` + `static/` is a dependency-free front end (native ES modules, Canvas 2D) for the stdlib server in
`server.py`. Its sources are bundled into `static/dist/app.{mjs,css}` by `python3 scripts/build_web.py` (Node needed only
for that step; the bundle is committed). The server is unchanged by it: it still serves `index.html` at `/` and everything
under `/static/`, and every run is still the CLI command the terminal UI would build.

| Path | What it is |
| --- | --- |
| `index.html` | The app shell (the previous single page is kept at `static/studio.html`) |
| `static/js/app/adapter.mjs` | **The only file that talks to the server.** Live adapter (`/api/*`) and a Demo adapter |
| `static/js/app/stage/` | The full-page floating protein (Canvas 2D) |
| `static/js/app/ui/` | One module per screen: home, setup, site picker, running, results, compare, score, history, help |
| `static/js/app/config.mjs` | Pose counts for Quick / Half / Full, the typical error, accent colours, Expert defaults |
| `static/css/app/tokens.css` | Every colour, font and size. `--accent` is the single accent token |
| `static/js/app/errors.mjs` | Turns a failed run into a plain-language title and next step (raw message kept under *Technical details*) |
| `static/data/` | `proteins.json` and the trimmed PDB files shown in 3D |
| `static/fonts/` | Bundled Inter and JetBrains Mono (OFL), the fallback for the system font, so it looks the same offline |

## Look and feel

The styling follows Apple's Human Interface Guidelines: content first, flat system-gray surfaces with hairline
separators, the system font (SF Pro on Apple devices), a segmented control with a raised thumb for Guided / Expert and
for the choices, accent colour only on primary actions, thin-material panels over the protein, 44 pt tap targets on touch
screens, and motion that stops under *Reduce Motion*. The first visit follows the system light/dark setting; the toggle in
the top bar overrides it. Tokens live in `static/css/app/tokens.css`.

## Live vs Demo

* Served by `hybridock-pep serve` (it answers `/api/env`) → **live**.
* Opened any other way, or with `?demo` → **Demo**: simulated, repeatable results, labelled *Demo* everywhere,
  with downloads named `*_DEMO`. `?live` forces live.
* **A free demo site.** `python3 scripts/build_pages.py` assembles a static, demo-only copy (relative URLs, so it works under a
  sub-path) and `scripts/publish_pages.sh` publishes it to the `gh-pages` branch for GitHub Pages
  (<https://tasty-ramen2010.github.io/hybridock-pep/>). It has no backend, so it is always Demo; real predictions need
  `hybridock-pep serve`. Re-run the script after changing the UI.

## How the live adapter uses the API

| UI action | Calls |
| --- | --- |
| Protein files | bundled examples use the server's own file (`/api/examples`); everything else is sent with `POST /api/upload` |
| Review → exact command, time estimate | `POST /api/preview`, `POST /api/validate` |
| Run prediction / compare / score | `POST /api/run` (`mode` = `dock`, `selectivity`, `crystal`) then `GET /api/jobs/<id>` every second |
| Stop | `POST /api/jobs/<id>/cancel` |
| Results | `GET /api/jobs/<id>/results`; comparison numbers from `selectivity.json` via `/file` |
| Pose in 3D, downloads | `GET /api/jobs/<id>/pose?name=…`, `/file?name=…` |
| Status lamp | `GET /api/env` |

The UI only ever sends the terminal UI's own field keys (`tui.FIELDS`).

## Honesty rules the UI follows

* `delta_g` is the binding strength; `rank_score` only orders poses of one protein and is never shown as a ΔG.
* A single run has no high/low confidence badge; the home card shows the typical error instead.
* "Strong / moderate / weak" is our own rough wording and is labelled as such.
* Compare shows which score the two numbers come from (`selectivity.json: score_field`).
* The server's progress is stage-weighted, so the bar can sit still during sampling; the UI never extrapolates an
  ETA from it, only from the server's own estimate.

## Changing common things

* **Accent colour**: `ACCENTS` in `config.mjs` (default `--accent` in `tokens.css`).
* **Quick / Half / Full pose counts**: `THOROUGHNESS` in `config.mjs` (the setting behind them is `--n-samples`).
* **Add a protein**: add it to `scripts/prepare_web_structures.py` and run that script.

## Tests

```bash
python -m pytest tests/test_web_ui.py tests/test_web_server.py   # contract, asset and packaging checks
node --test "tests/web_js/*.test.mjs"                            # the UI's pure logic (Node 20+)
node tests/web_js/regen_fixtures.mjs                             # after changing what the UI sends
```

`tests/web_e2e/` holds browser end-to-end and stress suites (Playwright): smoke, real Score runs, setup-form fuzzing,
12 tabs at once, a random-click "monkey", axe accessibility audits in light and dark, very large uploads, fault injection
(server frozen / killed, browser offline, reload mid-run), the History cap, keyboard-only use and a phone viewport. See
its README; they run against any server with `BASE=http://host:port`.
