"""Built-in how-to guide: `hybridock-pep guide [COMMAND]`.

Every number quoted here is measured, and each entry says where it came from.
Two distinct kinds appear, and they are labelled so they never get confused:

  * accuracy numbers  -- from the project benchmarks (see README), reproducible
    via `hybridock-pep benchmark`
  * timing numbers    -- measured on one machine (Apple M3, 10-core GPU, 16 GB),
    so they are a reference point for "is my run wildly off", not a promise

Nothing here is estimated. If a figure is not measured it is not printed.
"""

from __future__ import annotations

import shutil
import sys

_MACHINE = "Apple M3 (10-core GPU, 16 GB), macOS 27"


def _rule(ch: str = "─") -> str:
    return ch * min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100)


def _h(title: str) -> str:
    return f"\n{_rule()}\n{title}\n{_rule()}"


OVERVIEW = f"""
HybriDock-Pep — how to use it

  peptide  ->  AI diffusion poses (RAPiDock)  ->  physics rescoring  ->  ΔG (kcal/mol)

Start here if you have just installed:

    hybridock-pep crystal-score \\
        --receptor    data/pdbs/1YCR_mdm2.pdb \\
        --peptide-pdb data/pdbs/1YCR_peptide.pdb \\
        --peptide     ETFSDLWKLLPE

  This scores a known crystal complex and needs no GPU sampling, so it is the
  fastest way to prove the install works end to end.

  Expected: ΔG ≈ -10.07 kcal/mol. The published experimental value for this
  complex is about -8.5 kcal/mol (Kd ≈ 0.6 µM). Anything within a couple of
  kcal/mol means your install is healthy; the exact figure moves by up to
  ~1 kcal/mol with your resolved scikit-learn version.

Then run a real docking job:

    hybridock-pep dock \\
        --peptide ETFSDLWKLLPE \\
        --receptor data/pdbs/1YCR_mdm2.pdb \\
        --site 31.9 17.5 9.5 --box 20 \\
        --n-samples 100 \\
        --output-dir runs/my_first_run

Prefer a UI?  ./launch_ui.sh      (or `hybridock-tui`)
             ./launch_ui.sh --demo   runs a simulated pipeline, no GPU needed
             ./launch_ui.sh --print  builds the command without running it

Per-command detail:   hybridock-pep guide dock       (or prep, selectivity, ...)
Everything at once:   hybridock-pep guide all
"""

COMMANDS = {
    "dock": f"""{_h("dock — the main pipeline")}
Dock one peptide against one receptor and report a calibrated ΔG.

    hybridock-pep dock \\
        --peptide ETFSDLWKLLPE \\
        --receptor data/pdbs/1YCR_mdm2.pdb \\
        --site 31.9 17.5 9.5 --box 20 \\
        --n-samples 100 \\
        --output-dir runs/demo

Stages: RAPiDock diffusion sampling -> clash-relief minimization -> Vina/AD4
rescoring -> RMSD clustering -> calibrated affinity. Add --refine-topk 10 for
MM-GBSA on the top clusters, or --ultra for the full accuracy path.

What you get:
    runs/demo/best_pose.pdb        the top-ranked pose
    runs/demo/scored_poses.csv     every pose with its scores
    and a "Best pose: ΔG = ..." line on stdout

Timing, measured on {_MACHINE}, 12-mer vs a 705-atom receptor:
    sampling, --n-samples 100      ~64 s
    minimization, 100 poses        ~56 s   (~0.55 s/pose)
    --refine-topk 10 (MM-GBSA)     ~210 s on a 705-atom receptor
                                   ~470 s on a 2 550-atom receptor
MM-GBSA cost grows roughly with the square of receptor size, so a big receptor
dominates the run. Leave --refine-topk off unless you need it.

Notes:
  * --site is the box centre in Angstroms; --box the cube edge. Use 30 for
    12-mers and longer.
  * Receptor prep needs ADFRsuite (prepare_receptor / autogrid4) on PATH.
  * Sampling is NOT bit-reproducible on GPU even with --seed. Two identical
    runs differ by ~2.9 A mean pose RMSD. That is expected, not a bug.
""",

    "crystal-score": f"""{_h("crystal-score — validate your install")}
Score an existing complex. No sampling, no GPU, seconds to run.

    hybridock-pep crystal-score \\
        --receptor    data/pdbs/1YCR_mdm2.pdb \\
        --peptide-pdb data/pdbs/1YCR_peptide.pdb \\
        --peptide     ETFSDLWKLLPE

Expected: ΔG ≈ -10.07 kcal/mol  (experimental ≈ -8.5, Kd ≈ 0.6 µM).
Within a couple of kcal/mol means the install is good. The value shifts by up
to ~1 kcal/mol depending on your resolved scikit-learn version, so do not treat
a small difference as a failure.

This is the recommended first command after installing.
""",

    "prep": f"""{_h("prep — receptor preparation")}
Convert a receptor PDB to PDBQT and build AutoDock4 grid maps.

    hybridock-pep prep --receptor data/pdbs/1YCR_mdm2.pdb --output-dir runs/prep

Requires ADFRsuite (prepare_receptor, autogrid4) on PATH. Without it this
command, and the scoring stage of `dock`, cannot run — the tests that cover it
skip rather than fail. `dock` runs prep for you; use this only to inspect the
intermediate files.
""",

    "selectivity": f"""{_h("selectivity — on-target vs off-target ΔΔG")}
Dock the same peptide against two receptors and report ΔΔG.

    hybridock-pep selectivity \\
        --peptide ETFSDLWKLLPE \\
        --target-receptor    data/pdbs/1YCR_mdm2.pdb --target-site    31.9 17.5 9.5 --target-box    20 \\
        --offtarget-receptor data/pdbs/3LNJ_mdm2.pdb --offtarget-site 30.0 16.0 9.0 --offtarget-box 20 \\
        --n-samples 100 --output-dir runs/sel

More negative ΔΔG = more selective for the target. This runs the full dock
pipeline twice, so budget roughly double the `dock` timings above.
""",

    "reproducibility": f"""{_h("reproducibility — run-to-run spread")}
Repeat the same docking several times and report the spread.

    hybridock-pep reproducibility \\
        --peptide ETFSDLWKLLPE --receptor data/pdbs/1YCR_mdm2.pdb \\
        --site 31.9 17.5 9.5 --box 20 --n-samples 100 --output-dir runs/repro

Worth running before you trust a small ΔG difference between two peptides.
GPU sampling is not deterministic: identical seeds give ~2.9 A mean pose RMSD
between runs. If you use --refine-topk, note MM-GBSA adds its own spread of
~4 kcal/mol run-to-run on Apple GPUs, because Apple silicon has no fp64 and the
minimization can settle in a different basin. A ΔG gap smaller than that is not
a real difference.
""",

    "benchmark": f"""{_h("benchmark — accuracy on a held-out set")}
Score a CSV of complexes with known affinities and report error metrics.

    hybridock-pep benchmark --test-csv data/benchmark30.csv --output-dir runs/bench

Published accuracy (from the README; leakage-free 60%-identity clustered CV,
n=865 matched PDBbind peptide-Kd):

    model                          MAE     RMSE    Pearson r   Spearman rho
    HybriDock-Pep (16 feats, GBT)  1.35    1.69    0.352       0.338
    PPI-clone (ProtDCal-3D + SVR)  1.46    1.84    0.210       0.177

At the stricter 30% identity cutoff: MAE 1.39, r 0.32. MAE stays flat
(1.32 -> 1.42) across the whole identity sweep, which is the number the project
stands behind; r is more fragile to the choice of test set.

For context, FEP is typically quoted at ~1.2-2.5 kcal/mol absolute error, at far
higher cost and needing a reference compound.

Bundled sets: data/benchmark30.csv, data/benchmark300.csv,
data/all_binding_affinity.csv
""",

    "calibrate": f"""{_h("calibrate — refit the affinity model")}
Refit the calibration that turns raw scores into kcal/mol.

    hybridock-pep calibrate --training-csv data/all_binding_affinity.csv \\
                            --output-dir runs/calib

Only needed if you change the scoring stack or want to specialise to a family.
The shipped calibration is already fitted; re-running it changes every ΔG the
tool reports, so re-run `benchmark` afterwards and compare against the table in
`hybridock-pep guide benchmark` before trusting the new numbers.
""",
}

TUNING = f"""{_h("performance and tuning (Apple Silicon)")}
All measured on {_MACHINE}.

Environment switches:

  HYBRIDOCK_RAPIDOCK_BATCH=N
      Poses batched per diffusion step. Default is derived from your RAM
      (~15 on a 16 GB machine). Raising it past what fits causes swapping and
      is catastrophic, not gradual: batch 32 measured 398 s against 62 s at
      batch 16 for --n-samples 100. Lower it if you see swapping; raising it
      is rarely worth it.

  HYBRIDOCK_MMGBSA_FAST=1
      Drops H-bond constraints during MM-GBSA minimization: 5.4x faster
      (21.1 -> 4.3 s/pose). NOT on by default -- it shifts ΔG by up to
      ~4.5 kcal/mol and can reorder poses, so re-benchmark before relying on it.

  RAPIDOCK_DISABLE_METAL_TP=1
      Disables the fused Metal tensor-product kernel and falls back to stock
      e3nn. The kernel is ~3.2x faster on Apple GPUs and numerically identical
      (3.6e-07 max relative error); this switch exists for A/B testing.

What is already fast, so you need not tune it:
  * OpenMM runs on the Apple GPU through its OpenCL platform, ~10.7x the CPU
    platform. That is automatic.
  * The e3nn tensor products run a fused Metal kernel at ~76 GB/s, essentially
    the M3's memory bandwidth ceiling.

Known limits on Apple silicon:
  * No fp64 anywhere on the GPU, so MM-GBSA is single-precision only and is not
    reproducible run-to-run (~4 kcal/mol). The CPU platform has fp64 but is
    ~12.5x slower.
  * ADFRsuite is an x86 build; without it, receptor prep and Vina scoring do
    not run.
"""

TESTING = f"""{_h("running the tests")}
    pytest                       whole suite, ~13 s
    pytest tests/test_tui.py     just the terminal UI
    pytest -k mmgbsa             one area

Expect: 549 passed, 65 skipped, 0 failed.

Skips are not failures. Tests that need an external tool skip when it is
absent -- ADFRsuite (prepare_receptor / autogrid4), meeko, or openbabel. If you
install those, the skip count drops and more tests run.

If the suite hangs rather than fails, run it with
`pytest --timeout=120 --timeout-method=thread`, which prints the stack of the
stuck test instead of waiting on it.
"""


def _topics() -> dict:
    d = dict(COMMANDS)
    d["tuning"] = TUNING
    d["testing"] = TESTING
    return d


def print_guide(topic: str | None = None, stream=None) -> int:
    """Print the guide. Returns a process exit code."""
    out = stream if stream is not None else sys.stdout
    topics = _topics()

    if topic in (None, "", "overview"):
        print(OVERVIEW, file=out)
        print(f"Topics: {', '.join(sorted(topics))}, all", file=out)
        return 0

    key = topic.lower()
    if key == "all":
        print(OVERVIEW, file=out)
        for name in list(COMMANDS) + ["tuning", "testing"]:
            print(topics[name], file=out)
        return 0

    if key in topics:
        print(topics[key], file=out)
        return 0

    print(f"No guide topic {topic!r}.", file=out)
    print(f"Available: {', '.join(sorted(topics))}, all", file=out)
    return 1
