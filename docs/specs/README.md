# Design Specifications (D-XX Reference Index)

Source files reference design decisions as `D-01` through `D-11`. This document
explains each identifier so that any reader can trace a `# per D-07`-style
comment without hunting through the spec PDF.

The authoritative source is the technical spec PDF at
`docs/HybriDock-Pep_Technical_Specification.pdf` (§4, §5, §8, §11, §12, §16).
These stubs are a navigational aid only.

---

## Docking Engine Constraints

| ID   | Short Name              | Summary |
|------|-------------------------|---------|
| D-01 | Input-poses bypass      | `--input-poses` CLI flag bypasses Stage 1 (RAPiDock) and reads pre-generated PDB files directly. Required for macOS users who cannot run CUDA inference. |
| D-02 | Two-stage pipeline      | Stage 1 = RAPiDock diffusion sampling; Stage 2 = Vina + AD4 rescoring + entropy correction. Neither stage's output format may change without updating both. |
| D-03 | Box padding             | Grid box = peptide bounding box + 15 Å margin on each side, minimum 20 Å edge. Ensures the full peptide extent is covered even for extended conformations. |
| D-04 | PULCHRA version lock    | Must use PULCHRA v3.04 exactly. v3.07 produces incomplete aromatic side-chain atoms from ADCP output (bug is reproducible, §16 of spec). |
| D-05 | Hard abort on HD map    | Missing HD electrostatic map from autogrid4 = hard abort with diagnostic. Vina --scoring ad4 silently scores zero without HD; silent failure is worse than a crash. |

## Calibration Methodology

| ID   | Short Name              | Summary |
|------|-------------------------|---------|
| D-06 | AD4 anomaly flag        | Poses where AD4 and Vina scores disagree in sign are flagged `is_ad4_anomaly=True` and included in output with a warning. Not filtered out (§9 of spec). |
| D-07 | Per-pose exception isolation | Scoring exceptions are caught per pose; the batch never aborts on a single bad pose. Consistent with ProcessPoolExecutor BrokenProcessPool recovery policy. |
| D-08 | Entropy correction      | Hybrid score = α × n_contact + β × ad4_score + vina_score. The α × n_contact term approximates burial entropy (solvation penalty proxy, not true entropy). See NOTE ON TERMINOLOGY in entropy.py. |
| D-09 | RT constant             | RT at 298 K = 0.5922 kcal/mol. Hardcoded in v1; will be a flag in v2 if temperature-dependent scoring is added. |

## Benchmark Protocol

| ID   | Short Name              | Summary |
|------|-------------------------|---------|
| D-10 | Calibration starting point | Optimizer initial guess x0 = [α=0.65, β=0.22]. Empirically chosen from literature; re-derived each calibration run. |
| D-11 | Calibration JSON schema | Keys: alpha, beta, pearson_r, n_complexes, calibrated_at (ISO timestamp), git_sha. Any additional fields are stored as-is. |

---

*Last updated: 2026-05-26. Corresponds to spec v0.1.*
