"""Walkthrough and help text for the terminal UI.

Kept out of ``tui.py`` deliberately. This is the material a first-time user
actually reads, it is the part most likely to need editing by someone who is
not comfortable inside a prompt_toolkit layout, and holding it here means it
can be unit-tested (and rendered to plain stdout) without importing
prompt_toolkit at all.

Audience: people who are not computational scientists. Every topic answers
"what do I type and what does the answer mean", not "how is it implemented".
No jargon is used before it is defined.
"""

from __future__ import annotations

LOGO = r"""
    _   _       _         _ ____             _
   | | | |_   _| |__  _ _(_)  _ \  ___   ___| |__
   | |_| | | | | '_ \| '_| | | | |/ _ \ / __| / /
   |  _  | |_| | |_) | | | | |_| | (_) | (__|  \
   |_| |_|\__, |_.__/|_| |_|____/ \___/ \___|_|\_\
          |___/            P e p t i d e   E d i t i o n
"""


WELCOME = """\
 WELCOME — this screen appears the first time you open the UI.
 You can bring it back any time with Ctrl-W.

 WHAT THIS TOOL DOES, IN ONE LINE
   You give it a peptide and a protein. It predicts how tightly they stick
   together, as a number in kcal/mol. More negative = binds more tightly.

 THE FIVE-STEP WALKTHROUGH
   ┌───────────────────────────────────────────────────────────────────┐
   │ 1. PEPTIDE     Type the sequence, e.g.  ETFSDLWKLLPE              │
   │                One letter per amino acid. 3-30 letters.           │
   ├───────────────────────────────────────────────────────────────────┤
   │ 2. RECEPTOR    The protein file (.pdb). Drag the file onto the    │
   │                field, or press Ctrl-B to browse for it.           │
   ├───────────────────────────────────────────────────────────────────┤
   │ 3. SITE        Where on the protein to look — three numbers       │
   │                (x y z). If you do not know them, see Help topic 8.│
   ├───────────────────────────────────────────────────────────────────┤
   │ 4. BOX         How big a region to search, in angstroms.          │
   │                Use 30 for peptides of 12 letters or more.         │
   ├───────────────────────────────────────────────────────────────────┤
   │ 5. PRESS RUN   Start with  Demo ▷  — it simulates a full run in   │
   │                a few seconds and needs no GPU. Then try  Quick.   │
   └───────────────────────────────────────────────────────────────────┘

 IF YOU ONLY REMEMBER ONE THING
   Press  Demo ▷  first. Nothing is computed, nothing can break, and you
   see exactly what a real run looks like.

 WHEN IT IS RUNNING
   You will see progress bars filling up, like this:

     [############################] 100.0%  100/100 poses scored
     [##############--------------]  50.0%  8/16 denoising steps

   If a bar is moving, it is working. Some stages are slow and quiet —
   MM-GBSA takes several seconds per pose. That is normal.

   Enter  start filling the form        Ctrl-G  open help (9 topics)
"""


# (hotkey, short title, body). Order is the order shown in the menu.
TOPICS: list[tuple[str, str, str]] = [
    ("1", "Getting started", """\
 GETTING STARTED

 The fastest safe path, in order:

   1. Press  Demo ▷  (or Ctrl-T).
      Simulated run. No GPU, no files needed, finishes in seconds.
      Use it to learn what the screen does.

   2. Press  Quick  with the pre-filled example values.
      A real but small run (20 poses). A couple of minutes.

   3. Press  Full ▶  when you trust the setup.
      100 poses + MM-GBSA refinement. This is the real thing.

 The form already contains a working example, so you can press Demo or
 Quick right now without typing anything.

 Nothing you do here can damage your data. Every run writes into its own
 new folder under  runs/  and never edits your input files.

 CONTROLS — all Ctrl keys, so they work identically on every OS
   Ctrl-R  Full run      Ctrl-T  Demo run     Ctrl-B  Browse for a file
   Ctrl-P  Print command Ctrl-L  Clear output Ctrl-G  Help (this screen)
   Ctrl-W  Welcome tour  Ctrl-Q  Quit         Tab     Next field

 You can also just click any button with the mouse.

 FILE BROWSER (Ctrl-B)
   ↑/↓ move · Enter opens a folder or picks a file · U uses the current
   folder · Esc cancels. Dragging a file from Finder onto a path field
   works too.
"""),

    ("2", "The form fields", """\
 THE FORM FIELDS

 Peptide sequence
   One letter per amino acid, e.g. ETFSDLWKLLPE. Between 3 and 30 letters.
   Only the 20 standard letters are accepted. A ✗ appears if a letter is
   not valid.

 Target receptor PDB
   The protein, as a .pdb file. Three ways to fill it:
     • drag the file from Finder onto the field
     • press Ctrl-B and browse
     • type the path

 Target site (x y z)
   The centre of the pocket you want to search, in angstroms. Three
   numbers separated by spaces. See topic 8 if you do not know them.

 Target box
   The width of the search cube, in angstroms. 20 is fine for short
   peptides; use 30 for 12 letters or more. Too small and the peptide
   will not fit; too large wastes time.

 N samples
   How many candidate poses to generate. More = slower but more thorough.
   20 = quick look, 100 = standard.

 Scoring
   Which physics engines to run. Leave as-is unless you have a reason.

 Refine top-K (MM-GBSA)
   Runs a slower, more accurate energy calculation on the best K poses.
   0 turns it off. 10 is a good value. Each pose adds several seconds.

 A ✓ or ✗ appears beside every field as you type. You cannot start a run
 while any field shows ✗.
"""),

    ("3", "Run modes", """\
 RUN MODES  (the buttons along the bottom)

   Demo ▷        Simulated run. No GPU, no real computation, seconds.
                 Always safe. Start here.

   Quick         20 poses, Vina scoring only. Fastest real answer.
                 Good for "is this peptide worth looking at?"

   Half          50 poses, Vina + AD4. A middle option.

   Full ▶        100 poses, Vina + AD4, MM-GBSA on the top 10.
                 The standard production run.

   Selectivity ⚖ Docks against TWO proteins and compares them.
                 See topic 5.

 Rough timings on a laptop (Apple M3): Quick about a minute, Full about
 two minutes, plus roughly 3.5 minutes if MM-GBSA refinement is on.
 A slower machine, or a large protein, can take considerably longer.
"""),

    ("4", "Reading the results", """\
 READING THE RESULTS

 The headline number is ΔG (delta G), in kcal/mol.

   More negative  =  binds more tightly.

   about  -6      weak binding
   about  -8      moderate
   -10 or lower   strong

 For reference, the built-in example (the p53 peptide against MDM2) is a
 genuinely strong binder and scores about -9.3.

 Where the number comes from
   The tool generates many candidate poses, scores them with physics,
   groups similar ones together, and reports the best group. The ΔG you
   see comes from a model trained on 865 real measured complexes.

 How much should you trust a difference?
   Do NOT trust a gap smaller than about 1 kcal/mol between two peptides.
   Pose generation is random, so two identical runs differ slightly. If a
   difference matters to you, run it more than once (see topic 9).

 Files written into your output folder
   best_pose.pdb      the winning pose — open in PyMOL or ChimeraX
   ranked_poses.csv   every pose with all its scores
"""),

    ("5", "Selectivity (two proteins)", """\
 SELECTIVITY — does my peptide prefer protein A over protein B?

 This is the question "will my peptide hit the thing I want, and leave
 the similar-looking thing alone?"

 How to run it
   1. Fill in the normal target fields (protein A).
   2. Fill in the three Off-target fields (protein B): receptor, site, box.
   3. Press  Selectivity ⚖.

 It docks the peptide twice, once against each protein, and reports:

     ΔΔG  =  ΔG(target)  -  ΔG(off-target)

   Negative ΔΔG  =  prefers your target. Good.
   Near zero     =  no real preference.
   Positive      =  prefers the off-target. Bad.

 It also reports a confidence interval. If that interval crosses zero,
 the preference is not established — treat it as "no difference found".

 This runs the whole pipeline twice, so budget double the time.
"""),

    ("6", "Crystal scoring", """\
 CRYSTAL SCORING — I already have the structure

 Use this when you already have an experimentally determined structure of
 the peptide bound to the protein (from the PDB, or a very high quality
 model), and you only want the binding energy.

 It skips pose generation entirely, so it takes seconds instead of
 minutes, and it uses a model tuned specifically for crystal-quality
 structures.

 From the command line:

   hybridock-pep crystal-score \\
       --receptor    data/pdbs/1YCR_mdm2.pdb \\
       --peptide-pdb data/pdbs/1YCR_peptide.pdb \\
       --peptide     ETFSDLWKLLPE

 Expect about -9.3 kcal/mol for that example. Anything from -8 to -11
 means your installation is healthy.

 Important: do NOT use crystal scoring on a pose that came out of a
 docking run. It assumes the geometry is essentially perfect, and a
 docked pose is not. For docked poses the normal pipeline already picks
 the right model automatically.
"""),

    ("7", "AI scoring vs physics", """\
 AI SCORING vs PHYSICS SCORING — which number is the answer?

 You will see several numbers in the output CSV. Only one is the answer.

 The answer
   The headline ΔG comes from the AI affinity model. It reads the shape
   and chemistry of the contact between peptide and protein and predicts
   the measured binding energy. It was trained on real experimental data.

 Vina score
   A fast physics energy. The tool uses it to clean up poses that are
   physically impossible (atoms overlapping), and reports it as raw
   information. It is NOT the binding energy and is not calibrated
   against experiment. Do not quote it.

 AD4 score
   A second physics engine, off by default. Extra information only.

 MM-GBSA
   A much slower and more careful physics calculation, run only on the
   best few poses when you turn on "Refine top-K". Useful for ranking a
   handful of finalists against each other.

 Rule of thumb: quote ΔG. Everything else is diagnostic.
"""),

    ("8", "Finding the site (x y z)", """\
 FINDING THE BINDING SITE COORDINATES

 The site is just the centre of the pocket, in angstroms. Getting it
 wrong is the single most common mistake: if the box misses the pocket,
 the run completes but the answer is meaningless.

 If you already know where the peptide binds
   Open the structure in PyMOL and type:

     select pocket, chain B
     print(cmd.centerofmass("pocket"))

   Use those three numbers. (Chain B here means "the peptide chain" —
   change the letter to whatever your peptide chain is called.)

 If a known peptide is already bound in your PDB file
   Its centre IS the site. That is the most reliable source.

 If you have no idea
   Use a pocket finder such as fpocket or PrankWeb, or look up the
   protein in the literature — most well-studied targets have a named,
   documented binding site.

 Sanity check
   After a run, open best_pose.pdb next to your receptor. If the peptide
   is sitting in a groove, the site was right. If it is floating in empty
   space beside the protein, your coordinates were wrong.

 Example values that are known-good
   The UI form is pre-filled with a working pair, so you can press Demo
   or Quick immediately without knowing any of this:

     data/pdbs/1T2D_receptor.pdb   →  site 31.9 17.5 9.5    box 20

   The other bundled example, used by the command-line guide:

     data/pdbs/1YCR_mdm2.pdb       →  site 25.20 -25.61 -7.97  box 30

   These are not interchangeable. Each site belongs to its own receptor —
   using one protein's coordinates with another protein is exactly the
   mistake this topic is warning you about.
"""),

    ("9", "Calibration & reproducibility", """\
 CALIBRATION AND REPRODUCIBILITY

 Reproducibility — why two runs differ
   Pose generation is a random process, and on a GPU it is not exactly
   repeatable even with the same seed. Two identical runs typically give
   poses about 2.9 A apart on average.

   What this means for you: a ΔG difference under roughly 1 kcal/mol is
   noise, not signal. If MM-GBSA refinement is on, that noise is larger
   still (about 4 kcal/mol on Apple GPUs, which have no double-precision
   arithmetic).

   To measure it for your own system, from the command line:

     hybridock-pep reproducibility --peptide ... --receptor ...

   It repeats the run and reports the spread, so you know your own noise
   floor before trusting a small difference.

 Calibration — turning raw scores into kcal/mol
   The tool ships already calibrated on 865 measured peptide complexes.
   You do not need to do anything.

   You would only recalibrate if you had your own measured binding data
   for a family of peptides and wanted to specialise the model to it:

     hybridock-pep calibrate --training-csv my_data.csv

   Be careful: recalibrating changes every ΔG the tool reports. Always
   re-run  hybridock-pep benchmark  afterwards and compare against the
   published accuracy table before trusting the new numbers.
"""),

    ("0", "Troubleshooting", """\
 TROUBLESHOOTING

 "It looks frozen."
   Check for a progress bar. Pose generation and MM-GBSA are slow and
   quiet. If a bar is moving, it is working. A full run with refinement
   can take five minutes or more on a laptop.

 A warning about "mk_prepare_receptor.py failed / No template matched"
   Expected and harmless. The tool automatically falls back to a second
   receptor-preparation program. Both give the same scores.

 "ligand is outside the grid box"
   Your site coordinates or box size are wrong. See topic 8.

 The ΔG looks absurd (very large positive number)
   Almost always the same cause: the box is not on the pocket. Topic 8.

 A run fails partway
   Every run writes into its own folder under runs/. Nothing is lost and
   nothing is overwritten; just fix the input and run again.

 Getting the same help outside the UI
   Everything here is also available on the command line:

     hybridock-pep guide
     hybridock-pep guide dock
     hybridock-pep guide all
"""),
]


def topic_titles() -> list[tuple[str, str]]:
    """(hotkey, title) pairs, in menu order."""
    return [(key, title) for key, title, _ in TOPICS]


def topic_body(key: str) -> str | None:
    """Body text for a hotkey, or None if there is no such topic."""
    for k, _title, body in TOPICS:
        if k == key:
            return body
    return None


def menu_line() -> str:
    """One-line topic index for the help footer."""
    return "  ".join(f"{k} {title}" for k, title in topic_titles())


def render(key: str = "1") -> str:
    """Full help screen for a topic: body, then the menu of other topics."""
    body = topic_body(key) or topic_body("1") or ""
    items = []
    for k, title in topic_titles():
        marker = ">" if k == key else " "
        items.append(f"  {marker}{k}  {title}")
    return (
        body
        + "\n ─────────────────────────────────────────────────────────────────\n"
        + " PRESS A NUMBER TO SWITCH TOPIC:\n"
        + "\n".join(items)
        + "\n\n Esc or Ctrl-G to close help.   Ctrl-W reopens the welcome tour.\n"
    )
