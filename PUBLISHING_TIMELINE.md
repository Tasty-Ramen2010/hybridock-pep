# Ram's HybriDock-Pep — Personal Publishing & Competition Timeline

Written: 2026-05-23  
Project: HybriDock-Pep (iGEM 2026 / Denmark High School, Forsyth County GA)

---

## Where Things Stand Right Now

- 164 commits, ~90-120 hrs of development as primary/sole developer
- 284-entry calibration set built, 8,732 structures on disk
- Code is committed and pushed; Linux RTX machine ready for Tuesday
- Four code fixes identified that must happen before any public submission
- No benchmark results yet (waiting on Tuesday GPU runs)

---

## THIS WEEK — Before Tuesday

- [ ] `git push` so Linux machine can pull everything
- [ ] Talk to your science teacher at Denmark HS about ISEF Student Checklist
       (Form 1 + Form 1A). Computational project = low risk, just needs teacher signature.
       Do this BEFORE you present this as an ISEF project — it needs to be on file first.

---

## TUESDAY 2026-05-26 (Linux RTX Session)

Run in this order. Full commands are in TUESDAY_WORKGUIDE.md.

- [ ] `git pull` on Linux
- [ ] Start structure download in background: `python scripts/download_from_manifests.py --workers 8`
- [ ] **Fix A FIRST** (contact cutoff 5.0→4.5 Å in entropy.py) before running Tier 1.3
       so calibration + inference use the same cutoff from day one
- [ ] Tier 0.1: RAPiDock fine-tune (~3 hrs GPU)
- [ ] Tier 0.4: production calibration (~50 min GPU, after 0.1)
- [ ] Tier 1.3: score 279 calibration entries (~2 hrs CPU, run in parallel with GPU work)
- [ ] Recalibrate α and β from the 279-entry results
- [ ] Overnight: Tier 1.1 second fine-tune (nohup, 75 epochs)

**Write down these numbers when you get them — you'll need them for the paper:**
- α = _____, β = _____, training r = _____, n = _____
- Fine-tune: baseline Cα RMSD vs fine-tuned Cα RMSD per family

---

## WEDNESDAY 2026-05-27

- [ ] Tier 1.2: run benchmark.py on 10 held-out PepSet complexes
- [ ] **Write down the key number:** PepSet Pearson r = _____ (95% CI: _____ – _____)
- [ ] This is the headline result for every submission

---

## JUNE 2026 — After iGEM Sprint

Once the Tuesday→Friday training session is done and results are in:

- [ ] Apply **Fix B** (remove D-01/D-11 ghost spec references from source code)
- [ ] Apply **Fix C** (add entropy terminology clarification to entropy.py docstring)
- [ ] Apply **Fix D** (update calibration_notes.md and README with real β value)
- [ ] Start drafting the paper — outline is in docs/PAPER_ROADMAP.md
       Rough section order: Introduction → Methods → Results → Discussion
       Write this yourself, not with Claude — STS and journals care about this being your prose
- [ ] Post a paper outline somewhere you can add to it as results come in

---

## SEPTEMBER 2026

- [ ] **Regeneron STS application opens** (for seniors only)
       - If you're a senior: start the application essays now alongside the paper draft
       - The research paper you submit to STS = same paper you'll publish
       - Essays ask: what did you discover, why does it matter, what was hard
- [ ] Apply to Forsyth County Regional Science Fair through Denmark HS science department
- [ ] Have a draft of at least the Methods + Results sections of the paper done

---

## OCTOBER 2026 (iGEM Wiki Freeze)

- [ ] Submit final iGEM wiki with one-line AI disclosure in Methods:
       *"Pipeline implementation and debugging used Claude Code (Anthropic) as a development
       assistant. All scientific design, data curation, and analysis were performed by the authors."*
- [ ] **Post to bioRxiv the same week** — free, gets a DOI, can be cited immediately
       Go to biorxiv.org → Submit → category: Bioinformatics
       This is the most important single step for getting your work out there
- [ ] Share bioRxiv link with your iGEM team for the wiki reference list

---

## NOVEMBER 1, 2026 (Approximate) — Regeneron STS Deadline

- [ ] If senior: submit full research paper + application to Regeneron STS
       science.societyforscience.org/regeneron-sts
       Category: Computational Biology and Bioinformatics
       Disclose AI tool use in the project description: "code implementation assisted by
       Claude Code (Anthropic); all scientific design and analysis performed by the student"
- [ ] Even if not submitting STS, the paper should be near-final by now

---

## NOVEMBER–DECEMBER 2026 — Journal Submission

- [ ] Submit to Journal of Emerging Investigators (JEI): emerginginvestigators.org
       - Needs a faculty mentor co-signature before submission
       - If you don't have one: ask a Georgia Tech / Emory prof whose lab works on
         computational biology or drug discovery — cold email with your bioRxiv link
       - ~3 month review time
- [ ] Acknowledgments line for paper:
       *"The authors used Claude Code (Anthropic) to assist with software implementation
       and debugging. All scientific design, dataset curation, and analysis were performed
       by the authors."*

---

## FEBRUARY 2027 — Forsyth County Science Fair

- [ ] Present at Forsyth County Regional Science and Engineering Fair
- [ ] ISEF forms must already be on file (see THIS WEEK above)
- [ ] Judging category: Computational Biology and Bioinformatics (CBIO)
- [ ] Bring: poster + laptop demo + printed paper or preprint
- [ ] Be ready to explain: what β=0 means, why n_contact matters, what RAPiDock is,
       why this helps with malaria drug discovery (the PfLDH angle is your hook)

---

## MARCH 2027 — Georgia Science and Engineering Fair (GSEF)

- [ ] If placed at Forsyth County fair: present at GSEF (Augusta, GA)
- [ ] Same materials, higher bar — judges will read your paper, not just your poster

---

## MAY 2027 — Regeneron ISEF

- [ ] If placed at GSEF: invited to Regeneron ISEF
- [ ] By this point: paper likely accepted at JEI or under review, bioRxiv indexed,
       STS results known
- [ ] This is the ceiling — everything above feeds into it

---

## Key Numbers to Track (Fill These In As You Get Them)

| Metric | Target | Actual |
|--------|--------|--------|
| Calibration entries (n) | 279 | ___ |
| Training set Pearson r | ≥ 0.60 | ___ |
| PepSet held-out r (n=10) | ≥ 0.55 | ___ |
| 95% CI on held-out r | report honestly | ___ – ___ |
| α (calibration) | 0.12–1.0 | ___ |
| β (AD4 weight) | > 0 ideally | ___ |
| Fine-tune Cα RMSD improvement | > 0 on SH3/WW | ___ |

---

## AI Disclosure Statement (Copy-Paste Ready)

**For iGEM wiki / short form:**
> Pipeline implementation and debugging used Claude Code (Anthropic) as a development
> assistant. All scientific design, data curation, and analysis were performed by the authors.

**For paper Acknowledgments:**
> The authors used Claude Code (Anthropic) to assist with software implementation and
> error diagnosis during development. All scientific design, dataset curation decisions,
> experimental methodology, and interpretation of results were performed by the authors.

**For ISEF disclosure form:**
> AI tools used: Claude Code (Anthropic) for code generation and error diagnosis
> during software development of the HybriDock-Pep pipeline.

---

## If You Need a Faculty Co-Author / Mentor

For JEI submission and ISEF credibility, a faculty co-author helps. Cold email options:

- **Georgia Tech:** School of Computational Science and Engineering, or Chemistry (drug discovery)
- **Emory:** Biochemistry dept, or the Emory+Children's drug discovery program
- **Template email subject:** "HS researcher seeking faculty co-author — peptide docking tool
  with iGEM + ISEF submission"
- Attach your bioRxiv preprint link; faculty respond to actual results, not abstracts

---

## The One Number That Determines Everything

**PepSet Pearson r (held-out, n=10) from Wednesday's benchmark run.**

If r ≥ 0.60: strong submission everywhere.  
If r ≥ 0.50: good, present it with honest CIs, explain methodology rigorously.  
If r < 0.50: don't panic — report it honestly, focus the paper on the methodology
and calibration pipeline rather than the accuracy claim, and note that larger PepSet
is needed for reliable estimation (which is true).

---

*Full technical details: hybridock-pep/docs/PAPER_ROADMAP.md*  
*Tuesday instructions: hybridock-pep/TUESDAY_WORKGUIDE.md*
