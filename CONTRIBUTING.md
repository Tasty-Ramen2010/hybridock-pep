# Contributing to HybriDock-Pep

Bug reports, fixes and benchmark improvements are welcome. This page is the short version of how
the project works so a change can be reviewed quickly.

## Set up

Python **3.11** is required (`vina` has no wheel for newer versions yet). The full pipeline also
needs the separate RAPiDock environment; [INSTALL.md](INSTALL.md) covers both.

```bash
git clone --recursive https://github.com/Tasty-Ramen2010/hybridock-pep.git
cd hybridock-pep
make install-dev      # editable install with the dev extras
make verify           # fast offline check that the core math holds (no GPU, no external data)
make test             # the full unit-test suite
```

GPU sampling, ADFRsuite and AutoDock4 are not available in CI, so tests that need them skip
themselves. A skipped test is not a passed one — say in your change which of them you ran locally.

## Before you open a pull request

- **Tests pass:** `make test`. Add a test for the behaviour you changed; a bug fix should come with
  a test that fails without it.
- **Artifacts still match:** `python scripts/verify_artifacts.py`. If you *deliberately* changed a
  shipped model, calibration file or benchmark table, regenerate its checksum with
  `python scripts/verify_artifacts.py --write` and say why in the pull request.
- **Style:** the project is configured for `ruff`, `black` and `mypy --strict` at line length 100
  (see `pyproject.toml`). They are in the `dev` extra but are not enforced in CI, so run them on
  the files you touch.
- **Web UI changes:** run `node --test tests/web_js/*.test.mjs`. The browser suites in
  `tests/web_e2e/` are described in [tests/web_e2e/README.md](tests/web_e2e/README.md).

## Changes that touch the science

Scoring, calibration and benchmark changes are held to a higher bar than code changes, because the
numbers in [RESULTS.md](RESULTS.md) are what people rely on.

- **Report leakage-free numbers.** Use leave-cluster-out cross-validation with whole sequence
  clusters held out, as in [RESULTS.md](RESULTS.md). A result from a random split that lets homologs
  into training is not comparable.
- **Lead with MAE/RMSE in kcal/mol.** Pearson *r* is secondary and is bounded for every cheap
  method, not just this one.
- **Show before and after** on the same split, and include the command that reproduces it.
- **Changing the entropy correction?** Re-calibrate α with `scripts/calibrate_alpha.py` and compare
  against the previous values before proposing it.
- **Keep negative results.** If an idea does not work, document it. If a published number turns out
  to be wrong, retract it in RESULTS.md the way the E312 retraction there does.
- **Add a new dependency only if it is needed,** it is OSI-licensed, and you update
  [NOTICE.md](NOTICE.md) with its licence.

## What not to commit

- **Licensed tools or binaries** (ADFRsuite, AutoDock4, PULCHRA). Link to the official download in
  INSTALL.md instead.
- **Datasets you do not have the right to redistribute.** Add a row to
  [DATA_LICENSES.md](DATA_LICENSES.md) for any new data source and say what its terms are. If you
  do not know, mark it **Open** rather than guessing.
- **Large files.** Nothing over about 1 MB belongs in git; the three RAPiDock checkpoints in
  `weights/` are the documented exception ([weights/README.md](weights/README.md)). Put large
  artifacts on Zenodo and describe them in [docs/DATA_ARCHIVE.md](docs/DATA_ARCHIVE.md).
- **Secrets, run outputs, personal paths.** Run directories and caches are gitignored; keep it that way.

## Reporting problems

- **Bugs:** open an issue with the command you ran, the output, your OS and hardware, and the
  contents of `run_metadata.json` if a run produced one.
- **Security issues:** see [SECURITY.md](SECURITY.md) and please do not post details publicly first.
- **A number that looks wrong:** open an issue naming the row in RESULTS.md and the command that
  reproduces it. Those are the reports we most want.

By contributing you agree that your contribution is licensed under the [MIT licence](LICENSE).
