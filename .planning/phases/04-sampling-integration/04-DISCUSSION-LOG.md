# Phase 4: Sampling Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 04-sampling-integration
**Areas discussed:** Subprocess streaming, RAPiDock invocation, Pose count tolerance

---

## Subprocess Streaming

| Option | Description | Selected |
|--------|-------------|----------|
| Line-buffered Popen loop | subprocess.Popen + readline() loop. Simple, Python 3.9 compatible. | ✓ |
| asyncio subprocess | Real-time but adds async complexity to synchronous driver. | |
| Separate threads per stream | Two daemon threads, one each for stdout/stderr. | |

**User's choice:** Line-buffered Popen loop (Recommended)

---

## Stderr capture

| Option | Description | Selected |
|--------|-------------|----------|
| Separate stderr thread | Daemon thread captures stderr independently; both emit to logger. | ✓ |
| Merge stderr into stdout | subprocess.STDOUT — simpler but interleaves OOM errors with Vina logs. | |

**User's choice:** Separate stderr thread (Recommended)
**Notes:** GPU OOM goes to stderr — separate thread ensures it surfaces even when stdout is quiet.

---

## RAPiDock Invocation

| Option | Description | Selected |
|--------|-------------|----------|
| Write thin wrapper script in our repo | run_rapidock.py imports RAPiDock API, called via conda run python path/to/script | ✓ |
| python -m rapidock | Module entry point — not available (no __main__.py confirmed) | |
| rapidock CLI entry point | Console script binary — fragile, not confirmed to exist | |
| python path/to/inference.py | Internal script path — brittle, depends on upstream layout | |

**User clarification:** RAPiDock has no __main__.py or CLI entry point — requires installation and API import. Thin wrapper script in our repo is the right approach.

**Wrapper location:**

| Option | Description | Selected |
|--------|-------------|----------|
| src/hybridock_pep/sampling/ | Installed with package, absolute path resolves cleanly via __file__ | ✓ |
| scripts/ | Discoverable but path resolution across conda run boundary is less clean | |

**API entry point:**

| Option | Description | Selected |
|--------|-------------|----------|
| Discover at Phase 4 start | Researcher reads RAPiDock repo and identifies callable before writing code | ✓ |
| It's a script (inference.py) | Hardcode inference script path | |

**Notes:** Do not assume any specific API shape — researcher must inspect the installed RAPiDock repo first.

---

## Pose Count Tolerance

| Option | Description | Selected |
|--------|-------------|----------|
| Warn and continue | Log WARNING, proceed with available poses | ✓ |
| Abort with error | Exception if count < n_samples — harsh for exploratory runs | |
| Configurable threshold | min_poses field in DockConfig | |

**User's choice:** Warn and continue (Recommended)

**Shortfall recording:**

| Option | Description | Selected |
|--------|-------------|----------|
| Record in run_metadata.json | poses_requested + poses_generated fields | ✓ |
| Log only | WARNING log line sufficient | |

**Notes:** Zero poses generated is always a hard failure (RuntimeError), not a shortfall.

---

## Claude's Discretion

- Exact argparse flag names in run_rapidock.py
- threading.Thread vs ThreadPoolExecutor for stderr daemon
- PDB parsing strategy (Biopython vs manual ATOM parsing)
- rapidock_commit_sha discovery strategy

## Deferred Ideas

- Configurable min_poses threshold — deferred to v2
- GPU parallelism — out of scope per CLAUDE.md §7
- Incremental per-pose metadata — two writes (start + end) is sufficient for v1
