"""Run Boltz-2 in its own conda environment and hand back a co-folded complex.

WHY A THIRD ENVIRONMENT.  CLAUDE.md section 2.4 keeps RAPiDock and the scoring stack apart
because their pins are incompatible. Boltz is a third such stack -- installing its
cuequivariance dependency moved that environment from torch 2.7.0+cu128 to 2.14.0+cu130 -- so
it gets its own env and is driven the same way RAPiDock is, by subprocess with absolute paths.
It stays OPTIONAL: `--cofold none` is the default and the two-environment install in the
README is unchanged.

LICENSING.  Boltz is MIT, which satisfies the iGEM OSI requirement (CLAUDE.md section 7).
AlphaFold 3 weights are not redistributable and the server output cannot be automated, so AF3
is supported only through `engine="import"`, where the user supplies a structure they
generated themselves. Chai-1's weights carry a non-commercial restriction; it is reachable
the same way, by import, and is not wired in as a dependency.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

BOLTZ_PYTHON = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
#: Boltz writes mmCIF; the rest of the pipeline is PDB-only.
_CIF_GLOB = "*_model_0.cif"


def available() -> bool:
    """Whether the Boltz environment is installed and executable."""
    return BOLTZ_PYTHON.exists() and os.access(BOLTZ_PYTHON, os.X_OK)


def _yaml(receptor_seq: str, peptide_seq: str) -> str:
    """Boltz input for a two-chain protein-peptide complex."""
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {receptor_seq}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide_seq}\n"
    )


def cofold(
    receptor_seq: str,
    peptide_seq: str,
    out_pdb: Path,
    *,
    name: str = "cofold",
    recycles: int = 3,
    samples: int = 1,
    use_msa_server: bool = True,
    timeout_s: int = 3600,
) -> dict:
    """Co-fold a receptor and peptide with Boltz-2 and write the complex as PDB.

    Args:
        receptor_seq: Receptor sequence, one-letter codes.
        peptide_seq: Peptide sequence, one-letter codes.
        out_pdb: Where to write the converted complex.
        name: Job name, used for the working directory and Boltz's own output naming.
        recycles: Boltz recycling steps.
        samples: Diffusion samples; the top-ranked one is converted.
        use_msa_server: Query the public ColabFold MSA server. Turn this OFF for anything
            confidential -- it sends the sequence to a third party.
        timeout_s: Hard limit on the subprocess.

    Returns:
        Dict with the output path and Boltz's confidence metrics (iptm, ptm, plddt) where
        the run reported them.

    Raises:
        FileNotFoundError: If the Boltz environment is not installed.
        RuntimeError: If Boltz exits non-zero or produces no structure.
    """
    if not available():
        raise FileNotFoundError(
            f"Boltz not found at {BOLTZ_PYTHON}. Install it in its own environment, or use "
            f"--cofold import with a structure you generated elsewhere."
        )
    work = Path(tempfile.mkdtemp(prefix=f"cofold_{name}_"))
    try:
        spec = work / f"{name}.yaml"
        spec.write_text(_yaml(receptor_seq, peptide_seq))
        cmd = [
            str(BOLTZ_PYTHON), "predict", str(spec),
            "--out_dir", str(work),
            "--recycling_steps", str(recycles),
            "--diffusion_samples", str(samples),
            "--output_format", "mmcif",
            "--override",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        logger.info("co-folding %s (%d + %d aa) with Boltz-2", name, len(receptor_seq),
                    len(peptide_seq))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                              check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"boltz predict failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
            )
        cifs = sorted(work.rglob(_CIF_GLOB))
        if not cifs:
            raise RuntimeError(f"boltz produced no structure under {work}")
        info = _confidence(cifs[0])
        _cif_to_pdb(cifs[0], out_pdb)
        info["path"] = str(out_pdb)
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _confidence(cif: Path) -> dict:
    """Read Boltz's confidence JSON that sits beside the structure, if present."""
    import json

    for cand in cif.parent.glob("confidence_*_model_0.json"):
        try:
            c = json.loads(cand.read_text())
            return {k: c[k] for k in ("iptm", "ptm", "complex_plddt") if k in c}
        except (OSError, ValueError):
            break
    return {}


def _cif_to_pdb(cif: Path, out: Path) -> Path:
    """Convert a Boltz mmCIF to PDB, keeping chain ids and residue numbering.

    Boltz output is a plain `loop_` with no multi-line values, so a minimal reader is enough
    and avoids a gemmi/biopython round trip that renumbers chains.

    Args:
        cif: Boltz mmCIF.
        out: Destination PDB path.

    Returns:
        The path written.

    Raises:
        RuntimeError: If no ATOM records were found.
    """
    aa3 = {"MSE"}  # non-standard names we still emit; everything else passes through
    cols: list[str] = []
    rows: list[dict] = []
    in_loop = False
    for line in cif.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if not s or s.startswith(("#", "loop_", "data_")):
                if rows:
                    break
                continue
            parts = s.split()
            if len(parts) >= len(cols):
                rows.append(dict(zip(cols, parts)))
    if not rows:
        raise RuntimeError(f"no atom_site records in {cif}")

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for r in rows:
            if r.get("group_PDB") not in ("ATOM", "HETATM"):
                continue
            if r.get("group_PDB") == "HETATM" and r.get("label_comp_id") not in aa3:
                continue
            n += 1
            atom = r["label_atom_id"].strip('"')
            ch = (r.get("auth_asym_id") or r.get("label_asym_id") or "A")[0]
            seqid = int(r.get("auth_seq_id") or r.get("label_seq_id") or 0)
            el = (r.get("type_symbol") or atom[0]).rjust(2)
            an = atom if len(atom) >= 4 else f" {atom:<3}"
            fh.write(
                f"ATOM  {n:5d} {an:<4}{r['label_comp_id']:>4} {ch}{seqid:4d}    "
                f"{float(r['Cartn_x']):8.3f}{float(r['Cartn_y']):8.3f}"
                f"{float(r['Cartn_z']):8.3f}  1.00"
                f"{float(r.get('B_iso_or_equiv') or 0.0):6.2f}          {el}\n"
            )
        fh.write("END\n")
    logger.debug("converted %s -> %s (%d atoms)", cif.name, out, n)
    return out
