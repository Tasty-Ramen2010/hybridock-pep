#!/usr/bin/env python
"""Build the Coventry challenge benchmark from Wu et al. Science 2025 (adr8063).

Brian Coventry (UW, co-author) challenged us to reproduce Fig. 2B of
"Design of intrinsically disordered region binding proteins" -- an all-by-all
18x18 nanoBiT titration of 18 synthetic peptide targets against their 18 de novo
designed binders.  Ground truth:

  Table S1  (SI p.32)  the 18x18 Kd matrix itself. 30 of 324 cells have a fitted
                       Kd; the other 294 are "NA" = no measurable binding.
  Table S3A (SI p.34+) target sequences and binder sequences.

LABEL CAVEAT, pc17/pc18: Table S3A prints target `pc17` (DYDYDYDYVPVPVPVP) on the
same row as binder `pc18_1b1`, and target `pc18` (DYDYDYDYPVPVPVPV) on the row with
binder `pc17_1b1`.  Verified against the rendered page, so it is in the paper as
printed, not an extraction artifact.  We therefore key cognate pairs off the S3A
ROW (sequence-to-sequence), which is label-independent, and record the alternative
labelling so the two affected rows can be scored both ways.

Writes data/coventry_{targets,binders,pairs}.csv.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
RT = 0.0019872041 * 298.15  # kcal/mol at 25 C

# --- Table S3A: (target name, target sequence, binder name as printed) ---------
# Binder sequences are filled in from the extracted PDF table (s3_rows.json).
TARGETS: list[tuple[str, str, str]] = [
    ("n1",   "LKLKLKLKLKLKLK",     "n1_1b1"),
    ("n2",   "PVPVPVPVPVPVPV",     "n2_1b1"),
    ("n3",   "YDYDYDYDYDYDYD",     "n3_1b1"),
    ("n4",   "GAGAGAGAGAGAGA",     "n4_1b1"),
    ("n7",   "RTRTRTRTRTRTRT",     "n7_1b1"),
    ("pc2",  "LKLKLKLKYDYDYDYD",   "pc2_1b1"),
    ("pc11", "LKLKLKLKPVPVPVPV",   "pc11_1b1"),
    ("pc12", "LKLKLKLKVPVPVPVP",   "pc12_1b1"),
    ("pc17", "DYDYDYDYVPVPVPVP",   "pc18_1b1"),   # printed pairing, see caveat
    ("pc18", "DYDYDYDYPVPVPVPV",   "pc17_1b1"),   # printed pairing, see caveat
    ("pc21", "LKLKLKPVPVPVDYDYDY", "pc21_1b1"),
    ("pc26", "LKLKGAPVTQYDYDRTRT", "pc26_1b1"),
    ("pc28", "DANIELSILVA",        "pc28_1b1"),
    ("pc34", "HIALIENFRIEND",      "pc34_1b1"),
    ("pc35", "DERRICKHICKS",       "pc35_1b1"),
    ("pc43", "DAVIDLIKESPEPTIDE",  "pc43_1b1"),
    ("pc44", "DAVIDITISTIME",      "pc44_1b1"),
    ("pc46", "WILLCHEN",           "pc46_1b1"),
]

# --- Table S1: measured Kd in molar.  Key = (row label, column label). ---------
# Everything not listed is NA (no measurable binding) -- 294 of the 324 cells.
KD: dict[tuple[str, str], float] = {
    ("n1", "n1"): 2.2e-9,    ("n1", "n2"): 1.02e-6,  ("n1", "pc2"): 1.5e-7,
    ("n2", "n2"): 4.64e-9,   ("n2", "pc18"): 2.6e-7, ("n2", "pc17"): 2.5e-7,
    ("n3", "n3"): 9.42e-8,   ("n3", "pc28"): 3.84e-5,
    ("n4", "n4"): 1.9e-7,
    ("n7", "n7"): 2.6e-8,    ("n7", "pc35"): 1.5e-6,
    ("pc2", "pc2"): 1.2e-7,
    ("pc11", "n7"): 2.7e-7,  ("pc11", "pc11"): 5.6e-8,
    ("pc12", "pc11"): 2.1e-7, ("pc12", "pc12"): 5e-8,
    ("pc18", "pc18"): 2.2e-8, ("pc18", "pc28"): 7.6e-7,
    ("pc17", "pc17"): 5e-9,
    ("pc21", "pc21"): 6.3e-8,
    ("pc26", "pc26"): 2.1e-7,
    ("pc28", "pc18"): 7.8e-6, ("pc28", "pc28"): 2.7e-8,
    ("pc34", "pc34"): 2.3e-7,
    ("pc35", "pc35"): 1e-7,
    ("pc43", "pc43"): 6.2e-8,
    ("pc44", "n1"): 7e-6,    ("pc44", "n2"): 7.5e-6, ("pc44", "pc44"): 2.7e-8,
    ("pc46", "pc46"): 3.1e-8,
}

ORDER = ["n1", "n2", "n3", "n4", "n7", "pc2", "pc11", "pc12", "pc18", "pc17",
         "pc21", "pc26", "pc28", "pc34", "pc35", "pc43", "pc44", "pc46"]


def main() -> None:
    import json
    rows = json.load(open("/tmp/claude-1000/adr8063/s3_rows.json"))
    binder_seq = {r[2]: r[3] for r in rows if len(r) > 3 and r[2].endswith(("1b1", "1b2"))}

    with (ROOT / "data/coventry_targets.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "length", "cognate_binder"])
        for n, s, b in TARGETS:
            w.writerow([n, s, len(s), b])

    with (ROOT / "data/coventry_binders.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "grid_label", "sequence", "length"])
        for n, s, b in TARGETS:
            seq = binder_seq.get(b, "")
            w.writerow([b, n + "B", seq, len(seq)])

    # The full 324-cell grid, with measured Kd where one exists.
    with (ROOT / "data/coventry_pairs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["peptide", "binder_label", "binder_name", "peptide_seq",
                    "cognate", "kd_M", "dG_kcal_mol", "measured"])
        by_name = {n: (s, b) for n, s, b in TARGETS}
        for pep in ORDER:
            pseq, _ = by_name[pep]
            for bl in ORDER:
                _, bname = by_name[bl]
                kd = KD.get((pep, bl))
                dg = RT * math.log(kd) if kd else ""
                w.writerow([pep, bl + "B", bname, pseq, int(pep == bl),
                            kd if kd else "", f"{dg:.3f}" if dg else "", int(kd is not None)])

    n_meas = len(KD)
    print(f"targets   : {len(TARGETS)}")
    print(f"binders   : {sum(1 for _, _, b in TARGETS if binder_seq.get(b))} with sequence")
    print(f"grid cells: {len(ORDER) ** 2}  measured: {n_meas}  "
          f"cognate: {sum(1 for (a, b) in KD if a == b)}  "
          f"cross-reactive: {sum(1 for (a, b) in KD if a != b)}")
    print(f"binder length range: {min(len(binder_seq[b]) for _, _, b in TARGETS if binder_seq.get(b))}"
          f"-{max(len(binder_seq[b]) for _, _, b in TARGETS if binder_seq.get(b))} aa")
    print(f"peptide length range: {min(len(s) for _, s, _ in TARGETS)}"
          f"-{max(len(s) for _, s, _ in TARGETS)} aa")


if __name__ == "__main__":
    main()
