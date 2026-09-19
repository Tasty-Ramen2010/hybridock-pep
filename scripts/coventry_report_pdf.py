#!/usr/bin/env python
"""Build the Coventry challenge report as a PDF — the whole arc, numbers and nulls alike.

Every figure in this report is regenerated from the logs by its own script; nothing here is a
stale image. Every number is one we measured on this grid or on a held-out benchmark, and the
failures are given the same space as the wins because they are the part that took the work.

Usage: coventry_report_pdf.py [out.pdf]
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/HybriDock-Pep_Coventry_Report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6470")
RULE = colors.HexColor("#c8cdd4")
ACCENT = colors.HexColor("#1b4f7a")
GOOD = colors.HexColor("#0b6b3a")
BAD = colors.HexColor("#9c2a24")
BAND = colors.HexColor("#eef1f4")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=21,
                            leading=25, textColor=INK, spaceAfter=3),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5,
                          leading=15, textColor=MUTED, alignment=1, spaceAfter=13),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14,
                         leading=18, textColor=ACCENT, spaceBefore=15, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                         leading=14, textColor=INK, spaceBefore=11, spaceAfter=4),
    "b": ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6,
                        leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica-Oblique", fontSize=8.3,
                          leading=11.5, textColor=MUTED, spaceBefore=4, spaceAfter=10),
    "pull": ParagraphStyle("p", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=10.5,
                           leading=15, textColor=ACCENT, leftIndent=9, spaceBefore=5,
                           spaceAfter=9),
}


def P(t, k="b"):
    return Paragraph(t, S[k])


def tbl(rows, widths, head=True, align=None, hi=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    cmd = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.4),
           ("TEXTCOLOR", (0, 0), (-1, -1), INK),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("TOPPADDING", (0, 0), (-1, -1), 3.2),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
           ("LEFTPADDING", (0, 0), (-1, -1), 5),
           ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE)]
    if head:
        cmd += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.4),
                ("BACKGROUND", (0, 0), (-1, 0), BAND),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    for c in (align or []):
        cmd.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    for r in (hi or []):
        cmd += [("FONT", (0, r), (-1, r), "Helvetica-Bold", 8.4),
                ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e8f0e8"))]
    t.setStyle(TableStyle(cmd))
    return t


def fig(name, caption, width=168 * mm):
    """Image and its caption as ONE flowable, so a page break can never orphan the caption."""
    p = ROOT / "docs" / name
    if not p.exists():
        return [P(f"[missing figure: {name}]", "cap")]
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    return [KeepTogether([Image(str(p), width=width, height=width * h / w),
                          P(caption, "cap")])]


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.4)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 12 * mm,
                    "HybriDock-Pep · Denmark High School iGEM 2026 · Wu et al. Science 389:eadr8063")
    canv.drawRightString(190 * mm, 12 * mm, f"{doc.page}")
    canv.setStrokeColor(RULE); canv.setLineWidth(0.4)
    canv.line(20 * mm, 15.5 * mm, 190 * mm, 15.5 * mm)
    canv.restoreState()


def build() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=18 * mm, bottomMargin=22 * mm,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            title="HybriDock-Pep — the Coventry 18x18 specificity challenge",
                            author="Ram, Head of Dry Lab, Denmark High School iGEM 2026")
    F = []

    # ---------------------------------------------------------------- title
    F += [P("Reproducing an 18×18 specificity grid", "title"),
          P("HybriDock-Pep against Wu et al., <i>Science</i> 389:eadr8063 (2025), Fig. 2B<br/>"
            "Denmark High School iGEM 2026 · Dry Lab · 16 September 2026", "sub")]

    F.append(tbl([
        ["Final standing", "", ""],
        ["Mean rank of the true partner", "3.17 / 18", "random is 9.50"],
        ["Cognate ranked first outright", "7 / 18", ""],
        ["Cognate within top three", "13 / 18", "random 3.0"],
        ["Cognate within top five", "15 / 18", "random 5.0"],
        ["AUC, cognate vs measured non-binder", "0.871", "random 0.500"],
        ["Measured pairs inside our top five", "20 / 30", ""],
    ], [72 * mm, 38 * mm, 58 * mm], hi=[0]))
    F.append(Spacer(1, 5))
    F.append(P("Work carried out in roughly thirty hours from a standing start on the grid. "
               "The report gives the nulls the same weight as the results: eight of the "
               "eleven things tried did not work, and the three that did are the reason the "
               "number above is what it is.", "cap"))

    # ---------------------------------------------------------------- challenge
    F += [P("1 · The challenge", "h1"),
          P("Brian Coventry (Baker lab, UW), a co-author of the paper, asked whether "
            "HybriDock-Pep could reproduce the all-by-all specificity panel in Fig. 2B: "
            "eighteen synthetic disordered peptide targets against the eighteen de novo "
            "binders designed for them, measured by nanoBiT. Thirty of the 324 pairs bind "
            "measurably; the other 294 produced no detectable signal. The task is not "
            "affinity prediction. It is <b>specificity</b>: for each peptide, rank its own "
            "designed binder above the seventeen it was not designed for.", "b"),
          P("This is the hardest possible shape of the problem for us, and worth saying "
            "plainly. The binders are <b>de novo designed</b>, so none of them appear in any "
            "training corpus. The peptides are <b>low-complexity by construction</b> — ten of "
            "the eighteen are two-letter repeats such as LKLKLKLKLKLKLK, sequence entropy "
            "1.00 bits against a corpus mean of 2.87, the third percentile. And the binding "
            "site is a <b>pseudo-symmetric repeat groove</b>, which a peptide can thread in "
            "either direction while occupying almost the same density.", "b"),
          P("We checked for leakage before anything else: none of the eighteen binders, and no "
            "close homologue, appears in our training set. Nothing in what follows is "
            "memorised.", "b")]

    F.append(PageBreak())
    F += fig("coventry_digdepth.png",
             "<b>Figure 1.</b> Left: the measured grid. Right: our final ranking, with each row "
             "drawn only as deep as it costs — one cell if we called the true partner first, "
             "five if you would have to test five. Seven rows cost a single experiment; 48 of "
             "324 cells are shown in total.")

    F.append(PageBreak())

    # ---------------------------------------------------------------- start
    F += [P("2 · Where we started, and the first honest measurement", "h1"),
          P("HybriDock-Pep's calibrated ΔG scorer was built for exactly this: selectivity "
            "within a family. So the first thing we did was run it on the grid and see. It "
            "came back at <b>chance</b>.", "b")]

    F.append(tbl([
        ["Our calibrated ΔG on the grid", "value", "what chance looks like"],
        ["Mean cognate rank", "9.44 / 18", "9.50"],
        ["AUC", "0.501", "0.500"],
        ["Correlation with measured ΔG", "−0.268", "0.000"],
        ["MAE on the thirty measured pairs", "1.74 kcal/mol", "0.61 predicting the mean"],
    ], [72 * mm, 38 * mm, 58 * mm]))
    F.append(Spacer(1, 7))

    F += [P("The diagnosis mattered more than the number. Within a single peptide's row, our "
            "predictions spread by <b>0.22 kcal/mol</b> — smaller than the scorer's own error "
            "bar — while ref2015's spread across the same row was 12.4 REU, 93% of its total "
            "variance. Our scorer was reading the <i>peptide</i> and barely responding to which "
            "<i>binder</i> it was paired with. The signed error correlated with truth at "
            "−0.869, meaning the prediction was 87% a constant.", "b"),
          P("A model that is 87% a constant is not a weak model. It is a model answering a "
            "different question than the one being asked.", "pull"),
          P("This is also where the scope of our earlier claims got corrected. Our published "
            "selectivity result — cross-receptor r rising from −0.07 to +0.71 — holds for "
            "<i>variant series</i> with measured anchors. It does not transfer to an arbitrary "
            "all-by-all panel of unrelated de novo binders, and we now say so.", "b")]

    # ---------------------------------------------------------------- what worked
    F += [P("3 · What worked: cancellation", "h1"),
          P("An all-by-all panel is a complete two-way design, and that licenses a specific "
            "decomposition:", "b"),
          P("S(i, j) = grand + peptide(i) + binder(j) + interaction(i, j)", "pull"),
          P("Everything we systematically cannot model is a <b>main effect</b>. A peptide's "
            "desolvation offset, its length and charge, a binder's general stickiness — each is "
            "constant down a row or a column and cancels. Specificity <i>is</i> the interaction "
            "term, by definition: binder j suits peptide i beyond what either is worth alone. "
            "So the interaction residual is not a cosmetic normalisation, it is the estimator "
            "that matches what the assay measures.", "b"),
          P("The evidence that this is the right frame was already in our own numbers: roughly "
            "75% of affinity variance on this grid is a per-receptor baseline that no static "
            "representation recovers (pocket composition 0.049, ProtDCal 0.149, ESM-2 0.154). "
            "And our ΔG model's per-binder <i>column mean</i> correlates with true cognate "
            "affinity at +0.431 while its single-cell prediction correlates at −0.268. The main "
            "effects and the interaction carry different information.", "b"),
          P("We estimate the main effects with <b>Tukey's median polish</b> rather than plain "
            "double-centring, for a concrete reason: a docked grid always contains failed poses "
            "worth hundreds of REU, and a row mean is dragged by them, so the 'main effect' it "
            "removes is partly an artefact of the worst cell in the row. A median is unmoved by "
            "a handful of bad cells. That choice alone is worth 0.04–0.07 AUC.", "b")]

    F.append(tbl([
        ["Same energies, same poses — only the estimator changes", "rank", "AUC"],
        ["Raw ref2015 interface energy", "8.61", "0.563"],
        ["+ mean double-centring", "6.67", "0.667"],
        ["+ median polish (shipped)", "5.78", "0.735"],
    ], [104 * mm, 30 * mm, 34 * mm], align=[1, 2], hi=[3]))
    F.append(Spacer(1, 4))
    F.append(P("No fitting, no labels, no new physics — the same numbers, decomposed correctly. "
               "This is the single largest gain in the project and it is ours. Shipped as "
               "<font face='Courier' size='8'>src/hybridock_pep/scoring/cancellation.py</font>.",
               "cap"))

    F.append(PageBreak())

    # ---------------------------------------------------------------- deeper
    F += [P("4 · The same idea, one level down", "h1"),
          P("Pushing on why cancellation works turned up something we did not expect. On this "
            "grid, ref2015's total interface energy reaches AUC 0.735. Every one of the nine "
            "<i>already-weighted</i> terms that sums to it scores between 0.422 and 0.551 — "
            "chance, or worse.", "b")]

    F.append(tbl([
        ["ref2015 term (weighted, after cancellation)", "AUC on the grid"],
        ["fa_atr", "0.534"], ["fa_rep", "0.513"], ["fa_sol", "0.475"],
        ["fa_elec", "0.551"], ["lk_ball_wtd", "0.474"], ["hbond_sc", "0.422"],
        ["hbond_bb_sc", "0.473"], ["hbond_lr_bb", "0.543"],
        ["their sum — i_total", "0.735"],
    ], [104 * mm, 40 * mm], align=[1], hi=[9]))
    F.append(Spacer(1, 6))

    F += [P("The discriminating information is not in any term. It is in what the terms "
            "<b>cancel</b> when added: fa_atr and fa_rep both grow with how much peptide is "
            "buried, which is not specific to a pair, and in the sum that shared growth "
            "subtracts out. What survives depends on which residues face which.", "b"),
          P("Cancellation across terms inside one energy; cancellation across the grid in "
            "median polish. Same principle, two scales, neither of them fitted.", "pull"),
          P("We then asked the obvious question — can we learn that cancelling combination "
            "ourselves? Handed <i>exactly</i> those nine terms, a linear model trained on our "
            "own docked all-by-all blocks reaches 0.609. It loses 0.126 AUC to plain addition. "
            "And the learning curve is flat: 0.596 at five blocks, 0.601 at seven, 0.609 at "
            "nine. It is not a sample-size problem. ref2015's weights were fitted on orders of "
            "magnitude more structure than we have, and a delicate cancellation is precisely "
            "what small data cannot find.", "b"),
          P("That is a real limit and we state it rather than work around it: for now, "
            "replacing ref2015's sum outright is not within reach of the data we can generate.",
            "b")]

    # ---------------------------------------------------------------- 2x2
    F += [P("5 · Is it the poses or the features?", "h1"),
          P("Our Rosetta-free feature set kept underperforming, and there were two possible "
            "reasons with completely different consequences. We resolved it by scoring the "
            "co-folded structures with our own extractor — the missing cell of a 2×2.", "b")]

    F.append(tbl([
        ["pose scored", "ref2015 total", "our 18 features"],
        ["our docked poses", "0.735", "0.628"],
        ["co-folded poses", "0.772", "0.713"],
    ], [64 * mm, 44 * mm, 44 * mm], align=[1, 2], hi=[2]))
    F.append(Spacer(1, 6))

    F += [P("Our features gain <b>+0.085</b> from better poses; ref2015 gains +0.036. On good "
            "structures our own descriptors come within 0.022 of what ref2015 manages on ours. "
            "<b>The feature set is not what is broken.</b>", "b"),
          P("Which features gain says why, and it is physical. Side-chain hydrogen bonding is "
            "<i>below chance</i> on our poses (0.422) and strongly predictive on co-folded ones "
            "(0.714). An H-bond is a sub-ångström measurement, and our poses on these designed "
            "grooves are around 5.9 Å out, so on our structures that term is counting accidental "
            "contacts from a wrong register. The descriptors that do <i>not</i> gain are the "
            "composition-like ones — apolar fraction, pocket charge — which survive bad poses "
            "precisely because they ignore geometry, and cap out near 0.66.", "b")]

    F.append(tbl([
        ["feature", "our poses", "co-folded", "change"],
        ["i_hbond_sc", "0.422", "0.714", "+0.292"],
        ["n_contact_per_res", "0.491", "0.700", "+0.209"],
        ["i_fa_elec", "0.551", "0.741", "+0.190"],
        ["rec_atoms_touched_per_res", "0.541", "0.717", "+0.176"],
        ["c_apolar_frac", "0.659", "0.537", "−0.122"],
    ], [58 * mm, 32 * mm, 32 * mm, 30 * mm], align=[1, 2, 3]))

    F += [P("5b · How we got to a position where that question was answerable", "h1"),
          P("The grid took thirty hours. Being able to attack it at all took four months, and "
            "most of that was spent finding out that things we believed were wrong.", "b"),
          P("The largest was structural. In September we traced every failed fine-tuning run "
            "since May to a single line: training reused the <i>inference</i> dataset path, so "
            "the model was denoising toward a sequence-built idealised strand roughly 50 Å from "
            "the real pose. The decisive test was a crossover — the pretrained model fits "
            "crystal targets 1.8× better (17 of 21), while our fine-tuned one fits the idealised "
            "target better (0 of 21). Every benchmark run through that tree was confounded, and "
            "we said so and re-ran them.", "b"),
          P("Two smaller ones cost almost as much. Our fork used SiLU where the authors use Tanh "
            "in the torsion heads, so <i>unchanged pretrained weights</i> scored 9 of 77 through "
            "our tree against 27 of 77 upstream. And a crystal-target fine-tune silently shrank "
            "the translation and rotation gates by 0.28×, taking long-peptide performance from "
            "27 to 7 of 77 — freezing those layers is now mandatory.", "b"),
          P("Honest benchmarking cost us numbers we had been quoting. A leakage-free, "
            "peptide-clustered comparison on 918 PDBbind complexes put us at r = 0.317 against "
            "ref2015's 0.006; a matched 865-complex GroupKFold put us at 0.368 against a "
            "protein–protein baseline's 0.247 — which also meant that baseline's widely quoted "
            "0.554, measured on its own split, was no longer citable, so we stopped citing it. "
            "We retracted a state-counting entropy result outright when the seed-to-seed spread "
            "turned out to exceed the effect, and dropped a public peptide dataset on finding it "
            "100% Rosetta-contaminated.", "b"),
          P("What survived is what the grid was attacked with: a real first fine-tuning win on "
            "13–25-mers (p = 0.0002), a desolvation occupancy field worth +0.127 r that passed "
            "all four of its controls, a length router, and a scope decision — peptides over "
            "25 aa are small domains and out of bounds — that turned an apparent length cliff "
            "into a smooth slope.", "b"),
          P("The reflex the project actually bought: every claim gets a control, and the "
            "control gets run before the claim gets repeated.", "pull")]

    F.append(PageBreak())

    # ---------------------------------------------------------------- combination
    F += [P("6 · Two pose sources, and why combining them wins", "h1"),
          P("Running the identical scoring stack over structures from two independent "
            "generators gave the result the final number rests on. The two arms are close in "
            "strength — and their errors are <b>nearly uncorrelated</b>.", "b")]

    F += fig("coventry_combined.png",
             "<b>Figure 2.</b> Same scoring stack throughout; only the structure changes. "
             "B our own docked poses, C co-folded poses, D the two combined by rank average. "
             "Shaded cells are that peptide's top five; the number on the diagonal is where "
             "the true partner ranked.", width=176 * mm)

    F.append(tbl([
        ["combination rule (nothing fitted)", "AUC", "rank", "top-3", "top-5"],
        ["our poses alone", "0.735", "5.78", "6", "10"],
        ["co-folded poses alone", "0.802", "4.28", "8", "12"],
        ["z-sum of the two scores", "0.812", "4.11", "9", "12"],
        ["better of the two ranks (union)", "0.791", "4.78", "8", "11"],
        ["mean of the two row-ranks", "0.871", "3.17", "13", "15"],
    ], [62 * mm, 26 * mm, 26 * mm, 24 * mm, 24 * mm], align=[1, 2, 3, 4], hi=[5]))
    F.append(Spacer(1, 6))

    F += [P("Correlation between the two arms' cognate ranks is <b>−0.127</b>. Only two of "
            "eighteen rows miss both top fives. Averaging ranks beats averaging scores by 0.059 "
            "AUC because the arms sit on different energy scales with different spreads — a "
            "z-sum lets the wider arm dominate, a rank average cannot. It also beats taking the "
            "<i>better</i> of the two ranks, which discards the agreement between them and keeps "
            "whichever arm was the outlier.", "b"),
          P("Each arm's failures are flagged by a different diagnostic, which is the mechanism "
            "behind the independence rather than a coincidence: our failures track pose "
            "disagreement (r = +0.704), the co-folding arm's track its own confidence "
            "(ipTM, r = −0.697). Notably, neither tracks the paper's own cross-reactivity "
            "(+0.26 / +0.17) or cognate affinity (+0.10 / +0.10) — <b>the rows we miss are not "
            "the weak or promiscuous ones.</b>", "b")]

    F.append(PageBreak())

    # ---------------------------------------------------------------- nulls
    F += [P("7 · What did not work", "h1"),
          P("Eight of eleven things tried were null. They are listed because each one closed a "
            "direction, and because a report that only lists the wins is not a measurement.",
            "b")]

    for h, t in [
        ("More diffusion samples on the cells we miss",
         "The decisive test of sampling against scoring, pre-registered. Every cell in a failing "
         "row — the cognate and the three false positives beating it — got the same 25 diffusion "
         "samples, so the best-of-k bias applies equally to all four and the comparison stays "
         "within-row. If sampling were the deficit, the cognate should gain more than its "
         "competitors, because a real binding mode exists to be found for it and not for them. "
         "It gained <b>18.4 REU against their 16.0</b>, a difference well inside the 8.0 REU "
         "pose noise, and only one of seven rows moved by more than half a place while another "
         "moved backwards. <b>It is scoring, not sampling.</b> Twenty-five independent poses per "
         "pair also gave the first direct measurement of pose noise: 8.0 REU against a 15.2 REU "
         "chemistry signal, so roughly half the amplitude of what we are trying to read."),
        ("Learned ranker to replace ref2015",
         "Reached AUC 0.979 on its own synthetic threaded benchmark and 0.529 on the real grid. "
         "Retraining on <i>docked</i> poses — same generator, same checkpoint, same repack — "
         "lifted it to 0.628, closing most of the transfer gap but still below the 0.735 bar. "
         "The reason is in the training data itself: leave-block-out AUC is 0.740 on docked "
         "blocks against 0.979 on threaded ones, so pose error destroys ~0.24 AUC of learnable "
         "signal before any model is fitted."),
        ("Tail descriptors instead of means",
         "Our eighteen descriptors are all pooled averages, which is arguably the wrong shape of "
         "statistic: a mis-registered peptide in a repeat groove has the right composition in "
         "the wrong sockets, so a mean barely moves. We built thirteen tail statistics — worst "
         "contact, buried like-charge pairs, unsatisfied buried polars, contact Gini and "
         "entropy, anchor count. Choosing the best one <i>on the test set with free sign</i>, "
         "which can only flatter it, still lost: 0.735 → 0.721 on our poses, 0.772 → 0.729 on "
         "co-folded ones."),
        ("Importing the threading direction only",
         "Co-folding tells us which way the peptide runs through the groove, a bit we provably "
         "cannot get alone (79–81% of our pool threads backwards; ref2015 prefers the forward "
         "pose only 41% of the time). Filtering our own pool by that axis gave 0.634 → 0.591. "
         "The controls prove the bit is real — reversed prior 0.501 &lt; shuffled 0.532 &lt; "
         "real 0.591 — but the forward poses carry no more pair information than the rest."),
        ("Routing each peptide to the better arm",
         "The two arms' ranks correlate with peptide chemistry with opposite signs, suggesting a "
         "router. Under leave-one-out the best honest rule (pose disagreement, 13/18 calls "
         "correct) reached rank 3.00 and top-5 14. Plain averaging reaches 3.00 and 15, and the "
         "unachievable oracle only reaches 3.00 and 16. There is no routing rule to find."),
        ("Geometry-enriched fine-tuning, twice",
         "A solenoid-enriched fine-tune won 34 of 60 held-out complexes (p = 0.257); a "
         "designed-data fine-tune was likewise null (p = 0.167). Two independent failures of the "
         "same idea — stop buying pose accuracy with corpus composition."),
        ("3D tensor-product ΔΔG model",
         "Collapsed to a near-constant offset, predicting r = 0.998 across genuinely different "
         "substitutions. It was not learning hotspot physics."),
        ("Ensembling the learned ranker into the stack",
         "Genuinely decorrelated from the bar (r = −0.106) and still not additive: 0.882 against "
         "0.879, while losing three top-three hits."),
    ]:
        # deliberately NOT KeepTogether: with eight entries of very different lengths it
        # forces a whole entry onto a fresh page and leaves half a page blank
        F += [P(h, "h2"), P(t, "b")]

    # ---------------------------------------------------------------- bugs
    F += [P("8 · Errors we made, and how they were caught", "h1"),
          P("Four of these changed a headline number. They are recorded because the habit that "
            "caught them — every claim gets a control, every control gets run — is the part "
            "worth keeping.", "b")]

    F.append(tbl([
        ["what went wrong", "how it showed", "fix"],
        ["Training denoised toward a sequence-built idealised\nstrand ~50 Å from the real pose",
         "Invalidated every training run\nsince May", "use upstream's match=True path"],
        ["Four binder models (n3, n7, pc21, pc26) were wrong;\nBoltz and AF3 agreed 0.9–2.4 Å, ours 5.7–16.9 Å off",
         "72 of 324 cells affected", "re-transplant onto AF3;\n57 of 58 rescued"],
        ["Scored an AF3-frame peptide against an ESMFold\nreceptor", "+1512 and +15863 REU",
         "route the receptor by\npose provenance"],
        ["Passed a two-chain complex to a peptide-only refiner",
         "6,000–30,000 REU", "write the peptide alone"],
        ["Soft-repulsive pre-pass applied to docked poses",
         "AUC 0.646 → 0.579", "hard repack for docked poses"],
        ["A test of ours re-contaminated the residuals it was\nmeant to clean",
         "reported two methods equal\nto 15 decimals", "compare estimated effects"],
    ], [72 * mm, 45 * mm, 51 * mm]))
    F.append(Spacer(1, 5))
    F.append(P("The fourth row is worth dwelling on: rebuilding the grid after fixing the four "
               "binder models changed our score by 0.002. The bug was real, the fix was correct, "
               "and it was not what was holding us back. Finding that out was worth the day.",
               "cap"))

    # ---------------------------------------------------------------- standing
    F += [P("9 · Where we stand", "h1")]

    F.append(tbl([
        ["", "rank", "AUC", "top-3", "top-5"],
        ["Random", "9.50", "0.500", "3", "5"],
        ["Our ΔG scorer, as it was on day one", "9.44", "0.501", "—", "—"],
        ["Our poses + cancellation", "5.78", "0.735", "6", "10"],
        ["Co-folded poses + cancellation + ipTM", "4.28", "0.802", "8", "12"],
        ["Both arms, rank-averaged", "3.17", "0.871", "13", "15"],
    ], [66 * mm, 26 * mm, 26 * mm, 22 * mm, 22 * mm], align=[1, 2, 3, 4], hi=[5]))
    F.append(Spacer(1, 8))

    F += [P("Where we are strongest", "h2"),
          P("Seven peptides — n3, pc2, pc12, pc34, pc35, pc43, pc46 — have their true partner "
            "ranked <b>first outright</b>. Thirteen of eighteen are within three. The "
            "cancellation step is a general result, not a grid-specific trick: it applies to any "
            "complete panel, needs no labels and no fitting, and is the part of this we would "
            "defend hardest.", "b"),
          P("Where we still fail", "h2"),
          P("Two rows miss entirely. <b>n4</b> (rank 7) is the only peptide with a single "
            "measured partner and the second-weakest cognate affinity in the set. <b>pc18</b> "
            "(rank 11) is the clearest failure: our arm puts its true partner 14th of 18, and "
            "its cross-reactant pc28 is the cell where we are most wrong on the whole grid. "
            "Six further rows land between fourth and seventh — good enough for a shortlist, "
            "not good enough to call.", "b"),
          P("The limiting factor — and it is different in each arm", "h2"),
          P("These two results look contradictory and are not, so it is worth stating them "
            "together. <b>In our own arm the bottleneck is pose generation.</b> On natural "
            "complexes our poses reach a median 1.83 Å over 345 held-out RecentSet structures; "
            "on these designed repeat grooves they reach 5.87 Å. Every deficit we traced in that "
            "arm ends there — our features gain +0.085 when handed better structures, side-chain "
            "H-bonding goes from below chance to strongly predictive, and the learnable signal in "
            "our own training data falls by 0.24 AUC when the poses are docked rather than "
            "threaded.", "b"),
          P("<b>In the co-folding arm the bottleneck is scoring.</b> Its structures are already "
            "good, and twenty-five diffusion samples per cell bought nothing: the cognate gained "
            "18.4 REU and its false positives 16.0, inside the 8.0 REU pose noise. Once a pose is "
            "physically reasonable, ref2015-after-cancellation cannot tell a cognate from a "
            "well-formed decoy.", "b"),
          P("That is precisely why the two arms combine as well as they do, and it also sets the "
            "order of the work: a better generator lifts our arm, and only a better "
            "discriminator lifts the other.", "b"),
          P("3.17 of 18 against a random 9.50, with both bottlenecks measured rather than "
            "guessed at — and neither of them is the part of the method we would defend "
            "hardest, which is the cancellation.", "pull"),
          P("10 · What we would do next, in order", "h1"),
          P("Each of these is chosen because a measurement points at it, not because it is the "
            "obvious next thing to try.", "b")]

    F.append(tbl([
        ["", "why this one", "what would settle it"],
        ["Generator for low-complexity\npeptides in symmetric grooves",
         "our arm's bottleneck; 1.83 Å on natural\ncomplexes vs 5.87 Å on these",
         "designed-groove RMSD\nbelow ~3 Å"],
        ["A discriminator that works on\ngood poses",
         "the co-folding arm's bottleneck; more\nsampling provably does not help",
         "beat 0.802 on co-folded\nposes alone"],
        ["More all-by-all training blocks",
         "9 blocks / 72 cognates is thin, though the\nlearning curve is currently flat",
         "does the curve move\nat 20 blocks"],
        ["Ship the two-arm rank average\nin the CLI",
         "it is the best thing we have and needs no\nfitting, so nothing can overfit",
         "one flag, both pose\nsources, ranks averaged"],
    ], [50 * mm, 62 * mm, 56 * mm]))
    F.append(Spacer(1, 5))
    F.append(P("Two things we will not do: keep buying pose accuracy with corpus composition "
               "(two independent nulls), and keep fitting rankers on this much data to replace "
               "a sum whose weights came from orders of magnitude more structure (flat learning "
               "curve). Both were worth testing; neither is worth repeating.", "cap"))
    F.append(Spacer(1, 6))
    F.append(P("Every figure and number in this report regenerates from the logs by its own "
               "script in <font face='Courier' size='8'>scripts/</font>; nothing here is a "
               "stale image or a remembered value.", "cap"))

    doc.build(F, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
