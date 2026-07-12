# experiments/ — the research ledger (E0–E37x)

Every exploratory run, ablation, and refuted idea from HybriDock-Pep's development, kept on
the record. This is a **lab notebook**, not the shipping tool — the product is
`src/hybridock_pep/`; the reproducible headline numbers are in [`../RESULTS.md`](../RESULTS.md).

These scripts share a **flat namespace** (they `import` each other by bare `eNNN` name), so
run them **from inside this directory**:

```bash
cd experiments
OMP_NUM_THREADS=1 python e331_ours_vs_ppiclone_clustered.py   # 1.35 vs 1.46 head-to-head
OMP_NUM_THREADS=1 python e330_ours_pdbbind.py                 # full-set leakage-free MAE 1.40
```

Large/external inputs (PDBbind v2020, PPIKB, the PPI-Affinity SI) are gitignored — see
[`../INSTALL.md`](../INSTALL.md) and the reproduce table in [`../RESULTS.md`](../RESULTS.md).
The narrative of what each experiment found (and which ideas were killed) is in
[`../docs/DEVELOPMENT_TIMELINE.md`](../docs/DEVELOPMENT_TIMELINE.md).

Kept public on purpose: a reviewer can see every negative result, not just the wins.
`chain_*.sh` are multi-stage training/eval orchestration scripts from the same campaign.
