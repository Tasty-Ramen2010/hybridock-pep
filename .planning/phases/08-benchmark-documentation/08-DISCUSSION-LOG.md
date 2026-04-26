# Phase 8: Benchmark & Documentation — Discussion Log

**Session:** 2026-04-26
**Areas discussed:** Benchmark strategy, 10-complex dataset, Tutorial notebook, README + architecture docs

---

## Area 1: Benchmark Strategy

**Q: How should we handle the execution gap (ADFRsuite not on PATH in dev env)?**
Options: Write harness + run on RTX machine | Write harness + mock scoring tests | Write harness + pre-run results only
**Selected:** Write harness, run on RTX machine

**Q: What should benchmark.py do when invoked?**
Options: Full pipeline per complex | Score-only mode | Two-mode (full + score-only)
**Selected:** Full pipeline per complex

**Q: What output does benchmark.py produce?**
Options: Markdown report + CSV | CSV only | JSON + Markdown
**Selected:** Markdown report + CSV (benchmark_report.md + benchmark_results.csv)

---

## Area 2: 10-Complex Dataset

**Q: How do we get the list of 10 complexes?**
Options: User provides list | Curate from literature | Reuse training + expand
**Selected:** Curate from literature (researcher agent curates, user reviews)

**Q: Should test complexes overlap with training complexes?**
Options: Fully held-out | Partial overlap OK
**Selected:** Fully held-out (none of 2OY2, 1YCR, 3LNJ appear in test set)

---

## Area 3: Tutorial Notebook

**Q: Pre-run with outputs saved or live-runnable?**
Options: Pre-run with outputs saved | Live-runnable only
**Selected:** Pre-run with outputs saved

**Q: What sections should the tutorial cover?**
Options: Full walkthrough (install → dock → analyze → interpret) | Scoring-only | Minimal
**Selected:** Full walkthrough — uses --input-poses with MDM2/p53 fixtures to bypass GPU Stage 1

---

## Area 4: README + Architecture Docs

**Q: What should README.md prioritize?**
Options: Quick-start focused | Comprehensive user guide | iGEM-pitch style
**Selected:** Comprehensive user guide (all subcommands, all flags, troubleshooting)

**Q: docs/architecture.md format?**
Options: ASCII art + prose | Mermaid diagrams | Prose only
**Selected:** ASCII art + prose (expands CLAUDE.md §3 diagram)

**Q: License audit format?**
Options: docs/licenses.txt | LICENSES-AUDIT.md | Script only
**Selected:** docs/licenses.txt committed (both envs, pip-licenses plain-vertical output)
