#!/usr/bin/env python
"""Emit a copy-paste sheet of the 18 designed binder sequences for AlphaFold Server.

Ram runs these through alphafoldserver.com (real AF3) by hand; the resulting
structures drop straight into the docking grid in place of the local ESMFold models.

The sequence printed for submission is TAG-STRIPPED: the C-terminal GS-His6 is a
purification handle, is disordered, and folding it only grows a floppy tail next to
the groove we care about.  The removed suffix is printed alongside so nothing is
silently dropped, and the full published sequence stays in data/coventry_binders.csv.

Usage: coventry_seq_pdf.py [out.pdf]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path("/home/igem/unknown_software")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/coventry_binder_sequences.pdf"
WRAP = 60
MONO = {"family": "DejaVu Sans Mono"}
INK, MUTED, ACCENT = "#111111", "#666666", "#0b5394"


def strip_tag(seq: str) -> tuple[str, str]:
    m = re.search(r"(GS)?H{5,}$", seq.rstrip("*"))
    return (seq[:m.start()].rstrip("GS"), seq[m.start():]) if m else (seq, "")


def main() -> None:
    binders = list(csv.DictReader(open(ROOT / "data/coventry_binders.csv")))
    targets = {r["cognate_binder"]: r for r in
               csv.DictReader(open(ROOT / "data/coventry_targets.csv"))}
    fold = {}
    f = ROOT / "logs/coventry_fold.json"
    if f.exists():
        fold = json.loads(f.read_text())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        # ---------------- page 1: instructions + index ----------------
        fig = plt.figure(figsize=(8.5, 11))
        y = 0.955
        fig.text(0.07, y, "Designed binder sequences for AlphaFold Server",
                 fontsize=16, weight="bold", color=INK); y -= 0.028
        fig.text(0.07, y, "Wu et al., Science 389, eadr8063 (2025) - Table S3A - "
                 "the 18 binders of the Fig. 2B all-by-all grid",
                 fontsize=8.5, color=MUTED); y -= 0.038

        for line, style in [
            ("How to run these", {"fontsize": 11, "weight": "bold", "color": ACCENT}),
            ("1.  alphafoldserver.com  ->  sign in with Google.", {}),
            ("2.  One job per binder: Add entity -> Protein -> paste the SUBMIT sequence below.", {}),
            ("3.  Name the job exactly as the binder is named here (n1_1b1, n2_1b1, ...).", {}),
            ("4.  Single chain per job. No ligands, no peptide - we dock the peptide ourselves.", {}),
            ("5.  Daily quota is about 20-30 jobs, so all 18 fit in one day.", {}),
            ("6.  Download each result and drop the .cif into:", {}),
            ("        datasets/coventry/binders_af3/<binder_name>.cif", {"color": ACCENT}),
            ("", {}),
            ("The sequences below are tag-stripped (C-terminal GS-His6 removed): the tag is a", {}),
            ("purification handle, is disordered, and would only add a floppy tail beside the", {}),
            ("peptide groove. The removed suffix is shown for each entry.", {}),
            ("", {}),
            ("Priority if you run out of quota", {"fontsize": 11, "weight": "bold", "color": ACCENT}),
            ("n3_1b1, pc21_1b1 and n7_1b1 came out of ESMFold at pLDDT 75.6 / 75.8 / 79.4 -", {}),
            ("those three are the ones where a real AF3 structure would change the answer.", {}),
        ]:
            st = {"fontsize": 9, "color": INK, **style}
            fig.text(0.07, y, line, **st)
            y -= 0.023 if st.get("weight") == "bold" else 0.019
        y -= 0.012

        fig.text(0.07, y, "Index", fontsize=11, weight="bold", color=ACCENT); y -= 0.025
        hdr = f"{'binder':<11}{'grid':<7}{'cognate peptide':<26}{'aa':>4}{'pLDDT':>8}"
        fig.text(0.07, y, hdr, fontsize=7.6, weight="bold", color=INK, **MONO); y -= 0.017
        fig.text(0.07, y, "-" * 56, fontsize=7.6, color=MUTED, **MONO); y -= 0.017
        for r in binders:
            t = targets.get(r["name"], {})
            p = fold.get(r["name"], {}).get("plddt")
            seq, _ = strip_tag(r["sequence"])
            flag = "  <- low" if p is not None and p < 85 else ""
            row = (f"{r['name']:<11}{r['grid_label']:<7}"
                   f"{t.get('name', '?') + ' ' + t.get('sequence', ''):<26}"
                   f"{len(seq):>4}{(f'{p:.1f}' if p else '-'):>8}{flag}")
            fig.text(0.07, y, row, fontsize=7.6, color=INK, **MONO); y -= 0.0168
        pdf.savefig(fig); plt.close(fig)

        # ---------------- sequence blocks ----------------
        per_page = 4
        for start in range(0, len(binders), per_page):
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.96
            for r in binders[start:start + per_page]:
                seq, tag = strip_tag(r["sequence"])
                t = targets.get(r["name"], {})
                p = fold.get(r["name"], {}).get("plddt")
                fig.text(0.07, y, r["name"], fontsize=12, weight="bold", color=ACCENT)
                fig.text(0.28, y, f"grid {r['grid_label']}   cognate {t.get('name','?')} = "
                         f"{t.get('sequence','')}", fontsize=8.5, color=INK)
                y -= 0.019
                meta = f"{len(seq)} aa to submit   removed tag: {tag or '(none)'}"
                if p:
                    meta += f"   ESMFold pLDDT {p:.1f}"
                fig.text(0.07, y, meta, fontsize=7.5, color=MUTED); y -= 0.018
                for i in range(0, len(seq), WRAP):
                    fig.text(0.07, y, f"{i + 1:>4}  {seq[i:i + WRAP]}",
                             fontsize=7.6, color=INK, **MONO)
                    y -= 0.0145
                y -= 0.017
            pdf.savefig(fig); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Coventry challenge - 18 designed binder sequences"
        d["Subject"] = "AlphaFold Server submission sheet (Wu et al. Science 2025, adr8063)"

    print(f"wrote {OUT}")
    print(f"  {len(binders)} binders, "
          f"{sum(len(strip_tag(r['sequence'])[0]) for r in binders)} residues total")


if __name__ == "__main__":
    main()
