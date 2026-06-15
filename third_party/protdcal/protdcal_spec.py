"""Decoded ProtDCal spec for faithful PPI-Affinity reproduction.

Source: ProtDCal BMC Bioinformatics 2015 supplementary (third_party/protdcal/protdcal_SM.pdf),
tables SM-3 (topographic/contact weightings), SM-6..SM-9 (invariants), SM-11 (residue groups).

PPI-Affinity peptide model (.idl SI-File-2) selects 37 descriptors of the form
    w[Windex]([Property])_NO_[Group]_[Invariant]
where Windex in {Nc, FLC, NLC} are 3D-CONTACT weighting operators (need the bound structure),
Property in {ECI, IP, ISA, Z1, Z2, Z3}, Group is an SM-11 residue group, Invariant is SM-6..9.
=> PPI-Affinity is STRUCTURE-based (weighted intra-chain contact networks), NOT sequence-only.
"""
from __future__ import annotations

# SM-11 residue groups (one-letter), verbatim from the supplement.
GROUPS = {
    "AHR": set("ACQEHLKM"),       # ALA CYS GLN GLU HIS LEU LYS MET  (alpha-helix)
    "BSR": set("IFTWYV"),         # ILE PHE THR TRP TYR VAL          (beta-sheet)
    "RTR": set("NDGPS"),          # ASN ASP GLY PRO SER              (reverse turn)
    "PCR": set("RHK"),            # ARG HIS LYS                      (positive)
    "NCR": set("DE"),             # ASP GLU                          (negative)
    "UCR": set("NCQSTY"),         # ASN CYS GLN SER THR TYR          (uncharged)
    "ARM": set("HFWY"),           # HIS PHE TRP TYR                  (aromatic)
    "ALR": set("AGILMPV"),        # ALA GLY ILE LEU MET PRO VAL      (aliphatic)
    "UFR": set("GP"),             # GLY PRO                          (unfolding)
    "NPR": set("AGILMFPWV"),      # ALA GLY ILE LEU MET PHE PRO TRP VAL (non-polar)
    "PLR": set("RNDCQEHKSTY"),    # polar
    "PRT": set("ACDEFGHIKLMNPQRSTVWY"),  # entire protein (all residues)
}

# Properties (column names in protdcal_aa_table.csv are e.g. "ECI_NO")
PROPS = ["ECI", "IP", "ISA", "Z1", "Z2", "Z3"]
