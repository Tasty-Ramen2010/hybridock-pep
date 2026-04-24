# Phase 5: CLI & Driver - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-23
**Phase:** 05-cli-driver
**Areas discussed:** --input-poses bypass, Driver Stage 2 scope, calibrate/benchmark/prep depth

---

## --input-poses bypass

| Option | Description | Selected |
|--------|-------------|----------|
| Fully wired | driver.py skips conda run, calls parse_poses() on given dir, proceeds to Stage 2 | ✓ |
| Flag only, stub dispatch | Add flag, raise NotImplementedError in driver.py | |
| No flag yet | Defer entirely to Phase 7 | |

**User's choice:** Fully wired in Phase 5
**Notes:** Required for macOS users who can't run RAPiDock (no CUDA). Per CLAUDE.md §8.

---

## Driver Stage 2 scope

| Option | Description | Selected |
|--------|-------------|----------|
| Wire what exists, stub the rest | Calls sampling→parse→prep→score; logs Phase 6/7 stub message; returns list[ScoredPose] | ✓ |
| Stop at scoring, no stub | Returns scored poses cleanly, no mention of future phases | |
| Include placeholder calls | cluster_poses() and write_output() stubs that raise NotImplementedError | |

**User's choice:** Wire what exists, stub the rest
**Notes:** Clean handoff point — Phase 6 plugs in after scoring; Phase 7 plugs in after clustering.

---

## calibrate / benchmark / prep subcommand depth

| Option | Description | Selected |
|--------|-------------|----------|
| Real args + dispatch to existing logic | calibrate → calibrate_alpha.py; prep → prep/receptor.py; benchmark → NotImplementedError | ✓ |
| Real args, all stub dispatch | All get arg definitions, all raise NotImplementedError | |
| Full stubs only | Keep Phase 1 stubs as-is | |

**User's choice:** Real args + dispatch to existing logic
**Notes:** calibrate_alpha.py is already fully written. prep/receptor.py is fully written. Only benchmark stays as a stub (Phase 8 scope).

---

## Claude's Discretion

- How calibrate_alpha.py is invoked from the calibrate subcommand (import vs subprocess)
- Exact exit code conventions beyond argparse default (code 2)
- Arg name casing for multi-word flags

## Deferred Ideas

- --skip-sampling flag (OPT-02): v2 scope; --input-poses covers the Mac use case
- MM-GBSA --refine-topk execution: flag defined in Phase 5, actual dispatch is v2
- Parallel per-pose scoring: v2 optimization
