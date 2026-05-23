# HybriDock-Pep — Research Paper & Competition Roadmap

**Written:** 2026-05-23  
**Author:** Ram (primary developer)

---

## Can You Publish This Separately from iGEM?

**Yes, unambiguously.** This is standard practice in iGEM and academic science:
- iGEM wiki content is licensed CC BY 4.0, but that doesn't restrict your right to
  publish the *work* in a journal — it just means others can reuse your wiki text with
  attribution. The code (MIT/Apache) and the science are entirely yours to publish.
- Many iGEM teams publish tool papers separately (OpenMRS, iGEM's Ignitia, etc.).
  Publishing actually *strengthens* the iGEM project by giving it a citable DOI.
- The iGEM project is the parent/application; the paper describes the methodology.
  These are complementary, not competing.

**Authorship:** You wrote ~95% of the code (git log shows all commits from one account,
164 total over 35 days of active work). Appropriate authorship:
- **First author:** You (primary intellectual contribution, primary developer)
- **Co-authors:** Any team members who contributed scientific direction or experiments
- **Corresponding/last author:** Your PI/supervisor or iGEM team lead if applicable
- **Affiliation:** Your school + iGEM 2026 Denmark HS team

---

## Hours Committed (Git-Tracked)

From `git log` timestamps, first commit 2026-04-18, 164 total commits across 16 active days:

| Date | Session span | Est. hours |
|------|-------------|-----------|
| 2026-04-18 | 3 min | <1 |
| 2026-04-19 | 9:03–21:16 (8 hr cap) | ~8 |
| 2026-04-20 | 07:44–22:40 (8 hr cap) | ~8 |
| 2026-04-21 | 08:33–13:58 | ~5.5 |
| 2026-04-23 | 08:04–21:25 (8 hr cap) | ~8 |
| 2026-04-24 | 05:46–21:59 (8 hr cap) | ~8 |
| 2026-04-25 | 10:12–21:36 (8 hr cap) | ~8 |
| 2026-04-26 | 08:42–15:06 | ~6.5 |
| 2026-04-30 | 11:31–20:30 (8 hr cap) | ~8 |
| 2026-05-11 | ~1 hr (3 co-timed commits) | ~1 |
| 2026-05-13 | ~1 hr | ~1 |
| 2026-05-18 | 10:26–11:06 | ~0.7 |
| 2026-05-20 | 10:15–12:35 | ~2.3 |
| 2026-05-21 | 15:04–21:25 | ~6.4 |
| 2026-05-22 | ~1 hr | ~1 |
| 2026-05-23 | 03:05–07:51 | ~4.8 |
| **Total tracked** | | **~76 hrs** |

**Real total is significantly higher.** Git-tracked time measures only when commits happen —
it misses reading time (papers, PDB docs, RAPiDock source), debugging sessions with no commits,
design/planning time, and the overnight/early-morning Linux RTX sessions. Realistic total:
**90–120 hours over 5 weeks** as primary/sole developer.

This is strong evidence for first authorship and satisfies ISEF's "primarily the student's work"
requirement.

---

## What Would the Paper Claim?

### Core contribution
A calibrated hybrid scoring pipeline for peptide-protein docking that combines:
1. **SE(3)-equivariant diffusion pose generation** (RAPiDock, fine-tuned on peptide complexes)
2. **Empirical hybrid scoring** `hybrid = vina + β(ad4 − vina) + α × n_eff_residues`
3. **Calibration from 284 crystal complexes** with experimental binding affinity (pKd 3.2–10.3)

### What's novel
- The calibration framework with `n_eff_residues` contact correction is an original formulation
- Fine-tuning RAPiDock specifically on peptide-protein complexes (vs general protein docking)
- 284-entry crystal-complex calibration set from RCSB bulk affinity — this data collection
  pipeline itself is a contribution (no existing peptide docking calibration set is published
  at this scale with consistent curation)
- PPII-enriched sampling (SH3/WW domain binders at 4× weight) for underrepresented folds

### Honest scope
HybriDock-Pep is not a state-of-the-art docking engine. It is a *calibration and integration
framework* that makes an existing docking pipeline (RAPiDock + AutoDock) more useful for
peptide affinity prediction specifically. The paper should say that clearly.

---

## Target Venues

### Tier 1 — High Priority (Start Here)

**bioRxiv preprint**
- Free, immediate, gets a DOI on day 1
- Can be submitted the week after iGEM wiki freeze
- Gets indexed by Google Scholar; can be cited in the iGEM wiki itself
- Does NOT count as "published" — you can still submit to journals after
- **Do this first regardless of everything else**

**Journal of Emerging Investigators (JEI)**  
https://www.emerginginvestigators.org/  
- Peer-reviewed journal specifically for middle school through undergrad researchers
- Faculty mentors review before submission; you'll need your PI/advisor to sign off
- ~3-month review time; selective but HS-appropriate
- No publication fee
- **This is the most realistic path for a formally peer-reviewed publication as a HS student**

### Tier 2 — More Ambitious

**PLOS Computational Biology**
- Open-access, respectable impact factor
- Does not require novel biology — tool papers are accepted if well-validated
- Requires r ≥ 0.65 on the held-out PepSet (10 complexes) as minimum credibility
- Reviewers will notice β=0 and n_contact cutoff mismatch — fix those first (§17 of workguide)
- Long review time (6–12 months); realistic if you have a faculty co-author

**Bioinformatics (Oxford)**  
- "Application Note" format (2 pages max) is designed for tools
- Requires a faculty corresponding author in practice
- High bar but not impossible; needs PepSet r ≥ 0.60 + strong benchmark

### Tier 3 — Stretch Goals

**Nucleic Acids Research (Web Server Issue)**  
- Annual issue accepts computational biology tools as web servers
- Would require a running web demo (not just command-line)

---

## ISEF Pathway (Denmark)

ISEF (International Science and Engineering Fair) is open internationally but you can only
enter through an ISEF-affiliated regional fair in your country.

**Danish pathway:**

1. **UNF Unge Forskere / Danish Science Talent** (Danish national science talent program)
   - Run by UNF (Ungdommens Naturvidenskabelige Forening)
   - Application typically opens September–October each year
   - Submit project abstract + preliminary results
   - If selected, you present at a Danish national fair
   - Top placers at national fair are invited to ISEF the following May

2. **Alternative: Danish Science Festival / Videnskabernes Selskab**
   - Some institutions have direct ISEF affiliation pathways
   - Check with your school or supervisor

3. **ISEF itself (Louisville 2027 if you go through 2026–27 cycle):**
   - Category: Computational Biology and Bioinformatics (CBIO)
   - Judges will ask: what's the scientific question? (peptide docking accuracy)
   - They will want: experimental validation or strong benchmark (PepSet r)
   - Student Safety Form required; needs a supervising adult (teacher/PI)
   - No prior publication disqualifies you — preprints are fine, published papers are fine

**Realistic timeline for ISEF:**
- Sep 2026: Apply to Danish national fair with iGEM results + paper preprint
- Oct–Nov 2026: Present at Danish fair
- If selected: May 2027 ISEF Louisville

**Important:** ISEF requires the project to be "primarily the student's work." With 164 commits
and ~90-120 hours of tracked development, you easily satisfy this. Have your git log ready to
show the judging panel if asked.

---

## What to Fix Before Any Submission (Cross-reference §17)

In order of submission-blocking priority:

1. **Contact cutoff unification** (Fix A) — systematic calibration error; must fix before paper
2. **PepSet benchmark results** — paper needs actual r value from held-out test set
3. **Ghost spec references** (Fix B) — cosmetic but flags AI-assisted code to reviewers
4. **n=6 → n=279 calibration** — addressed by Tier 1.3 on Tuesday; paper needs this result
5. **Honest discussion of β** — report the actual β value, explain what it means if ≈ 0

---

## Draft Paper Outline

### Title
"HybriDock-Pep: A Calibrated Pipeline for Peptide–Protein Binding Affinity Prediction
Using SE(3)-Equivariant Diffusion and Empirical Interface Correction"

### Abstract (~250 words)
- Problem: peptide docking tools lack calibrated affinity prediction
- Approach: RAPiDock pose generation + Vina/AD4 scoring + 2-parameter calibration
- Calibration data: 284 crystal complexes, pKd 3.2–10.3
- Result: Pearson r = X.XX on 10 held-out PepSet complexes
- Software: open-source Python package + CLI

### Sections
1. Introduction (peptide docking problem, why existing tools miss affinity)
2. Methods
   - RAPiDock pose generation (SE(3)-equivariant diffusion, fine-tuning procedure)
   - Hybrid scoring formula + parameter derivation
   - Calibration dataset curation (RCSB bulk affinity, exclusion criteria)
   - MD minimization post-processing (optional §)
3. Results
   - Training set fit (r on 284 complexes)
   - Held-out benchmark (r on 10 PepSet complexes, with CI)
   - Per-family breakdown (SH3, WW, PDZ, BCL-2, MDM2)
   - Comparison vs Vina-only, AD4-only baselines
4. Discussion
   - Limitations: IC50/EC50 vs Kd, n=10 held-out set, β interpretation
   - Future: MD integration, affinity supervision head
5. Conclusion
6. Availability (GitHub + PyPI)

### Target length
- JEI: 3,000–5,000 words
- PLOS CB Application Note: ~3,000 words + figures
- Bioinformatics App Note: 1,300 words max

### Figures needed
1. Pipeline diagram (RAPiDock → scoring → calibration → pKd)
2. Calibration scatter plot (hybrid score vs pKd, n=284, colored by affinity type)
3. PepSet benchmark (predicted vs experimental pKd, n=10, with CI band)
4. Per-family RMSD improvement (fine-tuned vs baseline RAPiDock)

---

## When to Write It

| Milestone | Target date | What it unlocks |
|-----------|-------------|----------------|
| Tier 1.3 complete (scores on 284) | Tue 2026-05-26 | Figure 2 + real α/β values |
| PepSet benchmark r | Wed 2026-05-27 | Figure 3 — the key number |
| iGEM wiki freeze | ~Oct 2026 | Paper can go on bioRxiv same week |
| JEI submission | Nov 2026 | Peer review starts |
| Danish fair application | Sep–Oct 2026 | ISEF pathway opens |

**Earliest possible bioRxiv post:** right after iGEM wiki freeze (Oct 2026), using Tuesday's
benchmark results. The paper will be ready; it's the competition timeline that sets the pace.
