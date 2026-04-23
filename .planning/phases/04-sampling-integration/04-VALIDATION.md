---
phase: 4
slug: sampling-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-21
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥7.x |
| **Config file** | `pyproject.toml` (existing) |
| **Quick run command** | `pytest tests/test_sampling.py tests/test_output.py -x` |
| **Full suite command** | `pytest --cov=hybridock_pep` |
| **Estimated runtime** | ~10 seconds (all mocked; no live conda run) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_sampling.py tests/test_output.py -x`
- **After every plan wave:** Run `pytest --cov=hybridock_pep`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | SAMP-01 | T-4-01 | conda run cmd built as list (no shell=True) | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_command_construction -x` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 1 | SAMP-01 | — | RuntimeError on non-zero exit | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_nonzero_exit_raises -x` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 1 | SAMP-01 | — | WARNING (not raise) on shortfall | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_shortfall_warns -x` | ❌ W0 | ⬜ pending |
| 4-01-04 | 01 | 1 | SAMP-01 | — | RuntimeError on zero poses | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_zero_poses_raises -x` | ❌ W0 | ⬜ pending |
| 4-01-05 | 01 | 1 | SAMP-01 | — | rank*.pdb renamed to pose_{i}.pdb | unit | `pytest tests/test_sampling.py::TestRapidockRunner::test_file_rename -x` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | SAMP-01 | — | Valid PDB → PoseRecord with ca_coords shape [n,3] | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_valid_pdb -x` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 1 | SAMP-01 | — | Malformed PDB → PoseFailure(stage="parsing") | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_malformed_pdb -x` | ❌ W0 | ⬜ pending |
| 4-02-03 | 02 | 1 | SAMP-01 | — | Batch invariant: len(results)+len(failures)==len(inputs) | unit | `pytest tests/test_sampling.py::TestPoseIO::test_batch_invariant -x` | ❌ W0 | ⬜ pending |
| 4-02-04 | 03 | 2 | SAMP-01 | — | SEQRES records take priority over ATOM residue names (D-14) | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_seqres_preferred -x` | ❌ W0 | ⬜ pending |
| 4-02-05 | 03 | 2 | SAMP-01 | — | ATOM fallback succeeds when SEQRES absent (D-14, Pitfall 5) | unit | `pytest tests/test_sampling.py::TestPoseIO::test_parse_atom_fallback -x` | ❌ W0 | ⬜ pending |
| 4-03-01 | 03 | 2 | SAMP-02 | — | Skeleton write has status="running" | unit | `pytest tests/test_output.py::TestMetadata::test_skeleton_status_is_running -x` | ❌ W0 | ⬜ pending |
| 4-03-02 | 03 | 2 | SAMP-02 | — | Final write has all 14 required fields | unit | `pytest tests/test_output.py::TestMetadata::test_skeleton_has_required_fields -x` | ❌ W0 | ⬜ pending |
| 4-03-03 | 03 | 2 | SAMP-02 | — | Final write preserves clipped_poses from vina.py | unit | `pytest tests/test_output.py::TestMetadata::test_preserves_clipped_poses -x` | ❌ W0 | ⬜ pending |
| 4-03-04 | 03 | 2 | SAMP-02 | — | get_rapidock_commit_sha() reads direct_url.json | unit | `pytest tests/test_output.py::TestMetadata::test_commit_sha_from_direct_url -x` | ❌ W0 | ⬜ pending |
| 4-03-05 | 04 | 2 | SAMP-02 | — | finalize_metadata() records poses_generated count | unit | `pytest tests/test_output.py::TestMetadata::test_finalize_records_poses_generated -x` | ❌ W0 | ⬜ pending |
| 4-03-06 | 04 | 2 | SAMP-02 | — | finalize_metadata() adds timestamp_end key | unit | `pytest tests/test_output.py::TestMetadata::test_finalize_adds_timestamp_end -x` | ❌ W0 | ⬜ pending |
| 4-03-07 | 04 | 2 | SAMP-02 | — | atomic write uses .tmp intermediate file via os.replace | unit | `pytest tests/test_output.py::TestMetadata::test_atomic_write_uses_tmp_file -x` | ❌ W0 | ⬜ pending |
| 4-03-08 | 04 | 2 | SAMP-02 | — | finalize_metadata() sets status="complete" | unit | `pytest tests/test_output.py::TestMetadata::test_finalize_status_is_complete -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_sampling.py` — stubs for SAMP-01 (rapidock_runner + pose_io tests)
- [ ] `tests/test_output.py` — stubs for SAMP-02 (metadata.py tests)
- [ ] `tests/fixtures/pose_tiny.pdb` — minimal valid PDB with CA atoms (check if exists; create if not)

*Framework already installed; no new pytest install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| fair-esm 2.0.0 imports cleanly in rapidock-env under PyTorch 2.7 | SAMP-01 | Requires live rapidock-env with GPU | `conda run -n rapidock-env python -c "import esm; model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D'); print('OK')"` |
| End-to-end 100 poses generated on RTX 5070 | SAMP-01 | Requires GPU hardware | Run `hybridock-pep dock --peptide ETFSDLWKLLPE --receptor tests/fixtures/mdm2.pdb --site ... --n-samples 5` with `--input-poses` bypass disabled |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
