#!/usr/bin/env python3
"""Prepare a receptor/peptide pair for a HybriDock-Pep run.

Written for the "try me" workshop. Two ways to use it:

1. Pick a curated target from ``data/workshop_targets.csv``::

       python scripts/workshop_prep.py --target W01

2. Bring your own structure -- any RCSB entry or local PDB file::

       python scripts/workshop_prep.py --pdb 2ABC --receptor-chains A --peptide-chain B
       python scripts/workshop_prep.py --pdb-file mine.pdb --receptor-chains A --peptide-chain P

Either way it writes a receptor-only PDB, writes the reference peptide (when
there is one), derives the ``--site`` centre and a ``--box`` that actually
contains the peptide, and prints the ``hybridock-pep dock`` command to run.

The site centre is the centroid of the reference peptide's heavy atoms, in the
coordinate frame of the input file. That is the same convention the repo's
validated 1YCR fixture uses, and it reproduces the documented
``--site 25.20 -25.61 -7.97`` for that entry to within 0.5 A.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, NamedTuple

LOG = logging.getLogger("workshop_prep")

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS_CSV = REPO_ROOT / "data" / "workshop_targets.csv"
RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
    "HYP": "P",
}
#: Residues RAPiDock/Vina cannot represent as a standard amino acid. Docking a
#: peptide that carries one of these silently changes the chemistry.
MODIFIED_RESIDUES = {
    "SEP": "phosphoserine", "TPO": "phosphothreonine", "PTR": "phosphotyrosine",
    "TYS": "sulfotyrosine", "M3L": "trimethyl-lysine", "MLZ": "monomethyl-lysine",
    "MLY": "dimethyl-lysine", "ALY": "acetyl-lysine", "KCX": "carboxy-lysine",
    "HYP": "hydroxyproline", "CSO": "oxidised cysteine", "PCA": "pyroglutamate",
    "ACE": "N-terminal acetyl cap", "NH2": "C-terminal amide cap",
}

Vec = tuple[float, float, float]


class Residue(NamedTuple):
    """One residue of one chain, model 1 only."""

    chain: str
    resid: str
    resname: str
    atoms: list[tuple[str, Vec]]


class PrepError(RuntimeError):
    """Raised when the requested chains or target cannot be prepared."""


# --------------------------------------------------------------------------- IO


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Return a local path to ``pdb_id``, downloading from RCSB if needed.

    Args:
        pdb_id: Four-character PDB accession, case-insensitive.
        cache_dir: Directory to cache downloads in; created if absent.

    Returns:
        Path to the cached ``.pdb`` file.

    Raises:
        PrepError: If the entry cannot be downloaded.
    """
    pdb_id = pdb_id.upper()
    local = REPO_ROOT / "datasets" / "raw_pdbs" / f"{pdb_id}.pdb"
    if local.is_file():
        LOG.info("using bundled copy %s", local.relative_to(REPO_ROOT))
        return local

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_id}.pdb"
    if dest.is_file() and dest.stat().st_size > 0:
        LOG.info("using cached download %s", dest)
        return dest

    url = RCSB_URL.format(pdb=pdb_id)
    LOG.info("downloading %s", url)
    tmp = dest.with_suffix(".pdb.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
            fh.write(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise PrepError(f"could not download {pdb_id} from RCSB: {exc}") from exc
    tmp.replace(dest)  # atomic: a torn download leaves a .part, not a bad .pdb
    return dest


def parse_pdb(path: Path) -> tuple[dict[str, list[Residue]], list[str]]:
    """Parse model 1 of a PDB file into per-chain residue lists.

    Waters and non-amino-acid heteroatoms are dropped. Alternate locations
    other than blank/``A`` are dropped.

    Args:
        path: PDB file to read.

    Returns:
        A ``(chains, header_lines)`` pair, where ``chains`` maps chain id to its
        residues in file order and ``header_lines`` holds CRYST1/REMARK records
        worth carrying into the split outputs.

    Raises:
        PrepError: If the file contains no parsable amino acids.
    """
    chains: dict[str, list[Residue]] = {}
    seen: set[tuple[str, str]] = set()
    header: list[str] = []
    with path.open(errors="ignore") as fh:
        for line in fh:
            record = line[:6]
            if record == "CRYST1":
                header.append(line.rstrip("\n"))
                continue
            if record == "ENDMDL":
                break
            if record not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in ("HOH", "WAT"):
                continue
            if line[16] not in (" ", "A"):
                continue
            chain = line[21]
            resid = line[22:27]
            key = (chain, resid)
            if key not in seen:
                seen.add(key)
                chains.setdefault(chain, []).append(Residue(chain, resid, resname, []))
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                LOG.debug("skipping unparsable coordinate line: %s", line.rstrip())
                continue
            chains[chain][-1].atoms.append((line[12:16].strip(), xyz))
    for chain in list(chains):
        kept = [r for r in chains[chain]
                if {"N", "CA", "C"} <= {name for name, _xyz in r.atoms}]
        if kept:
            chains[chain] = kept
        else:
            del chains[chain]
    if not chains:
        raise PrepError(f"no amino-acid residues found in {path}")
    return chains, header


def write_chains(path: Path, source: Path, keep: Iterable[str]) -> int:
    """Copy the ATOM/HETATM records of ``keep`` chains from ``source``.

    A TER is emitted at every real chain break and a single END terminates the
    file, so downstream ADFRsuite/Meeko parsing does not trip.

    Args:
        path: Output PDB path.
        source: Input PDB path.
        keep: Chain identifiers to retain.

    Returns:
        Number of atom records written.
    """
    keep = set(keep)
    written = 0
    previous_chain: str | None = None
    with source.open(errors="ignore") as src, path.open("w") as out:
        for line in src:
            record = line[:6]
            if record == "ENDMDL":
                break
            if record == "CRYST1":
                out.write(line)
                continue
            if record not in ("ATOM  ", "HETATM"):
                continue
            if line[21] not in keep:
                continue
            resname = line[17:20].strip()
            if resname in ("HOH", "WAT"):
                continue
            chain = line[21]
            if previous_chain is not None and chain != previous_chain:
                out.write("TER\n")
            previous_chain = chain
            out.write(line)
            written += 1
        if written:
            out.write("TER\nEND\n")
    return written


# ------------------------------------------------------------------- geometry


def heavy_atoms(residues: Iterable[Residue]) -> list[Vec]:
    """Return heavy-atom coordinates (hydrogens dropped) for ``residues``."""
    return [
        xyz
        for res in residues
        for name, xyz in res.atoms
        if not name.startswith("H") and not (name[:1].isdigit() and "H" in name[:2])
    ]


def centroid(points: list[Vec]) -> Vec:
    """Return the arithmetic centroid of ``points``."""
    n = len(points)
    return (
        round(sum(p[0] for p in points) / n, 2),
        round(sum(p[1] for p in points) / n, 2),
        round(sum(p[2] for p in points) / n, 2),
    )


def span(points: list[Vec]) -> Vec:
    """Return the per-axis extent (max minus min) of ``points``."""
    return tuple(  # type: ignore[return-value]
        round(max(p[i] for p in points) - min(p[i] for p in points), 1) for i in range(3)
    )


def box_for(extent: Vec, padding: float = 8.0, minimum: int = 24) -> int:
    """Return an even box edge in A that contains ``extent`` plus ``padding``.

    The default padding reproduces the 30 A box the repo settled on for the
    1YCR 12-mer after poses were found leaking outside a 20 A box.

    Args:
        extent: Per-axis peptide extent in A.
        padding: Slack added to the longest axis, in A.
        minimum: Floor on the returned edge, in A.

    Returns:
        Box edge length in A, rounded up to an even integer.
    """
    return max(minimum, int(math.ceil((max(extent) + padding) / 2.0) * 2))


def sequence(residues: Iterable[Residue]) -> str:
    """Return the one-letter sequence of ``residues``."""
    return "".join(AA3_TO_1.get(r.resname, "X") for r in residues)


def modified_residues(residues: Iterable[Residue]) -> dict[str, str]:
    """Return the chemically modified residues present, as ``{code: name}``."""
    return {r.resname: MODIFIED_RESIDUES[r.resname] for r in residues if r.resname in MODIFIED_RESIDUES}


def contacts(a: list[Vec], b: list[Vec], cutoff: float = 5.0) -> int:
    """Count ``a``-``b`` heavy-atom pairs closer than ``cutoff`` A."""
    grid: dict[tuple[int, int, int], list[Vec]] = {}
    for p in b:
        grid.setdefault((int(p[0] // cutoff), int(p[1] // cutoff), int(p[2] // cutoff)), []).append(p)
    total = 0
    for p in a:
        cx, cy, cz = int(p[0] // cutoff), int(p[1] // cutoff), int(p[2] // cutoff)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if math.dist(p, q) < cutoff:
                            total += 1
    return total


# --------------------------------------------------------------------- targets


def load_targets() -> dict[str, dict[str, str]]:
    """Load ``data/workshop_targets.csv`` keyed by target id.

    Raises:
        PrepError: If the curated CSV is missing.
    """
    if not TARGETS_CSV.is_file():
        raise PrepError(f"missing curated target list: {TARGETS_CSV}")
    with TARGETS_CSV.open() as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def print_targets(category: str | None, difficulty: str | None, group: str | None) -> None:
    """Print the curated target list, optionally filtered."""
    rows = load_targets().values()
    for row in rows:
        if category and category.lower() not in row["category"].lower():
            continue
        if difficulty and difficulty.lower() != row["difficulty"].lower():
            continue
        if group and group.lower() != row["selectivity_group"].lower():
            continue
        print(
            f"{row['id']}  {row['pdb']}  {row['peptide_seq']:<31} "
            f"{row['pep_len']:>2}aa  {row['difficulty']:<9} {row['category']}"
        )


# ------------------------------------------------------------------------ main


def prepare(
    source: Path,
    receptor_chains: list[str],
    peptide_chain: str | None,
    out_dir: Path,
    label: str,
    peptide_override: str | None,
    site_override: tuple[float, float, float] | None,
    box_override: int | None,
) -> None:
    """Split a structure and report the docking command for it.

    Args:
        source: Input PDB file.
        receptor_chains: Chain ids forming the receptor.
        peptide_chain: Chain id of the reference peptide, if the file has one.
        out_dir: Directory for the split outputs.
        label: Basename stem for the outputs and the suggested run directory.
        peptide_override: Sequence to dock when the file has no peptide chain.
        site_override: Explicit site centre, bypassing the peptide centroid.
        box_override: Explicit box edge in A.

    Raises:
        PrepError: If a requested chain is absent or empty.
    """
    chains, _header = parse_pdb(source)
    missing = [c for c in receptor_chains if c not in chains]
    if missing:
        raise PrepError(
            f"receptor chain(s) {missing} not in {source.name}; "
            f"available: {sorted(chains)}"
        )
    if peptide_chain and peptide_chain not in chains:
        raise PrepError(
            f"peptide chain {peptide_chain!r} not in {source.name}; "
            f"available: {sorted(chains)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdb = out_dir / f"{label}_receptor.pdb"
    n_rec = write_chains(receptor_pdb, source, receptor_chains)
    if not n_rec:
        raise PrepError(f"receptor chains {receptor_chains} contained no atoms")
    LOG.info("wrote %s (%d atoms, chains %s)", receptor_pdb, n_rec, "+".join(receptor_chains))

    receptor_atoms = heavy_atoms(r for c in receptor_chains for r in chains[c])
    peptide_seq = peptide_override
    site = site_override
    box = box_override
    extent: Vec | None = None

    if peptide_chain:
        residues = chains[peptide_chain]
        peptide_atoms = heavy_atoms(residues)
        peptide_pdb = out_dir / f"{label}_peptide.pdb"
        write_chains(peptide_pdb, source, [peptide_chain])
        LOG.info("wrote %s (reference pose, %d residues)", peptide_pdb, len(residues))

        peptide_seq = peptide_seq or sequence(residues)
        extent = span(peptide_atoms)
        site = site or centroid(peptide_atoms)
        box = box or box_for(extent)

        mods = modified_residues(residues)
        if mods:
            LOG.warning(
                "peptide chain %s carries modified residues: %s. HybriDock-Pep will "
                "dock the unmodified sequence, which is a different molecule -- for "
                "phospho/methyl-dependent binders the result is not meaningful.",
                peptide_chain,
                ", ".join(f"{k} ({v})" for k, v in sorted(mods.items())),
            )
        if len(set(sequence(residues))) == 1:
            LOG.warning("peptide sequence is a homopolymer; check the chain id")
        n_contacts = contacts(peptide_atoms, receptor_atoms)
        if n_contacts < 20:
            LOG.warning(
                "peptide chain %s makes only %d heavy-atom contacts with the chosen "
                "receptor chains -- it may bind a different copy in this file",
                peptide_chain,
                n_contacts,
            )

    if site is None:
        raise PrepError("no peptide chain and no --site given: cannot place the box")
    if peptide_seq is None:
        raise PrepError("no peptide chain and no --peptide given: nothing to dock")
    box = box or 30

    nearest = min(math.dist(site, q) for q in receptor_atoms)
    if nearest > 8.0:
        LOG.warning(
            "site centre is %.1f A from the nearest receptor atom -- that is open "
            "solvent, not a pocket. Check the chain selection.",
            nearest,
        )

    if extent is not None:
        half = box / 2.0
        outside = sum(
            1 for q in heavy_atoms(chains[peptide_chain])  # type: ignore[index]
            if any(abs(q[i] - site[i]) > half for i in range(3))
        )
        if outside:
            raise PrepError(
                f"box of {box} A leaves {outside} reference-peptide atoms outside; "
                f"pass --box {box_for(extent, padding=14.0)} or larger"
            )

    bad = sorted(set(peptide_seq) - set("ACDEFGHIKLMNPQRSTVWY"))
    if bad:
        codes = ""
        if peptide_chain:
            unknown = sorted({r.resname for r in chains[peptide_chain]
                              if r.resname not in AA3_TO_1})
            if unknown:
                codes = f" (residue codes {', '.join(unknown)})"
        raise PrepError(
            f"peptide contains residues HybriDock-Pep cannot represent: {bad}{codes}. "
            f"Docking the plain sequence would be a different molecule."
        )

    print()
    print(f"  receptor : {receptor_pdb}   ({n_rec} atoms, chains {'+'.join(receptor_chains)})")
    print(f"  peptide  : {peptide_seq}   ({len(peptide_seq)} aa)")
    print(f"  site     : {site[0]} {site[1]} {site[2]}   (nearest receptor atom {nearest:.1f} A)")
    print(f"  box      : {box} A" + (f"   (peptide extent {extent})" if extent else ""))
    print()
    print("Run it:")
    print(
        f"  hybridock-pep dock \\\n"
        f"      --peptide {peptide_seq} \\\n"
        f"      --receptor {receptor_pdb} \\\n"
        f"      --site {site[0]} {site[1]} {site[2]} \\\n"
        f"      --box {box} \\\n"
        f"      --n-samples 100 \\\n"
        f"      --scoring vina,ad4 \\\n"
        f"      --output-dir runs/{label}"
    )
    print()


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    p = argparse.ArgumentParser(
        description="Prepare a receptor/peptide pair for a HybriDock-Pep run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target", help="curated target id from data/workshop_targets.csv, e.g. W01")
    src.add_argument("--pdb", help="RCSB accession to fetch, e.g. 1YCR")
    src.add_argument("--pdb-file", type=Path, help="local PDB file to split")
    src.add_argument("--list", action="store_true", help="list the curated targets and exit")

    p.add_argument("--receptor-chains", help="comma- or plus-separated chain ids, e.g. A or A+B")
    p.add_argument("--peptide-chain", help="chain id of the reference peptide in the file")
    p.add_argument("--peptide", help="peptide sequence to dock (needed if the file has no peptide chain)")
    p.add_argument("--site", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="box centre in A, overriding the reference-peptide centroid")
    p.add_argument("--box", type=int, help="box edge in A, overriding the derived value")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "workshop_prep",
                   help="where to write the split PDBs (default: runs/workshop_prep)")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "runs" / "pdb_cache",
                   help="where to cache RCSB downloads (default: runs/pdb_cache)")
    p.add_argument("--category", help="with --list: filter by category substring")
    p.add_argument("--difficulty", help="with --list: filter by easy/moderate/hard/'very hard'")
    p.add_argument("--group", help="with --list: filter by selectivity group, e.g. nrbox")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        if args.list:
            print_targets(args.category, args.difficulty, args.group)
            return 0

        if args.target:
            targets = load_targets()
            row = targets.get(args.target.upper())
            if row is None:
                raise PrepError(f"unknown target {args.target!r}; try --list")
            LOG.info("%s: %s", row["id"], row["notes"])
            pep_chain = row["peptide_chain"]
            pep_chain = None if pep_chain.startswith("(") else pep_chain
            if row.get("local_file"):
                source = REPO_ROOT / row["local_file"]
                if not source.is_file():
                    raise PrepError(f"curated local file missing: {source}")
                LOG.info("using curated local file %s", row["local_file"])
            else:
                source = fetch_pdb(row["pdb"], args.cache_dir)
            receptor_chains = row["receptor_chains"].replace("+", ",").split(",")
            site = None if pep_chain else (
                float(row["site_x"]), float(row["site_y"]), float(row["site_z"]))
            return_label = f"{row['id']}_{row['pdb']}"
            prepare(source, receptor_chains, pep_chain, args.out_dir, return_label,
                    row["peptide_seq"] if not pep_chain else None,
                    args.site or site, args.box or int(row["box"]))
            return 0

        if not args.receptor_chains:
            raise PrepError("--receptor-chains is required with --pdb / --pdb-file")
        source = args.pdb_file if args.pdb_file else fetch_pdb(args.pdb, args.cache_dir)
        if not source.is_file():
            raise PrepError(f"no such file: {source}")
        label = source.stem
        prepare(source, args.receptor_chains.replace("+", ",").split(","),
                args.peptide_chain, args.out_dir, label, args.peptide,
                tuple(args.site) if args.site else None, args.box)
        return 0
    except PrepError as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
