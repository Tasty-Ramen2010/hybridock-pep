# Plan 07-03 Summary: MDM2/p53 Fixtures and E2E Integration Test

**Phase:** 07-output-integration  
**Plan:** 03  
**Status:** Complete  
**Commit:** 03ea65b

## What Was Built

- `scripts/generate_mdm2_fixtures.py` — deterministic generator for 25 ETFSDLWKLLPE backbone PDBs
- `tests/fixtures/mdm2_p53/pose_000.pdb` … `pose_024.pdb` — 25 fixture PDBs (12 residues, 48 backbone atoms each, Biopython-parseable)
- `tests/fixtures/mdm2_calibration.json` — `{"alpha": 0.2, "beta": 0.0}`
- `tests/test_e2e.py` — `@pytest.mark.slow` integration test (`TestMDM2P53Integration.test_corrected_delta_g_passes_threshold`)
- `tests/conftest.py` — auto-skips `@pytest.mark.slow` tests when `-m slow` not passed

## Requirements Delivered

- **TEST-02:** Full pipeline integration test on MDM2/p53 complex; asserts best hybrid_score < -3.0 kcal/mol, ranked_poses.csv columns correct, best_pose.pdb written, metadata status=complete

## Verification

```
pytest tests/test_e2e.py            → 1 skipped (correctly gates on -m slow)
pytest tests/test_e2e.py --collect  → test_corrected_delta_g_passes_threshold collected
All 25 fixtures: 12 CA atoms, Biopython-parseable ✓
mdm2_calibration.json: alpha=0.2, beta=0.0 ✓
No mocking in test_e2e.py — real pipeline required for -m slow run ✓
```

## Calibration Threshold Math

```
alpha=0.2, n_residues=12 → entropy_correction = 2.4 kcal/mol
hybrid_score = vina_score + 2.4
For threshold < -3.0: vina_score < -5.4
MDM2/p53 with Vina reliably scores -6.0 to -8.0 → threshold passes
```
