"""E108 — ingest PDBbind v2020 peptide subset → curated CSV with our 16 features (the data lever, LIVE).

Source: Ram's Drive upload (PDBbind v2020 general PL). 2150 peptide entries with Kd/Ki (14x our 156).
For each: parse ΔG from index, convert ligand.mol2 → peptide PDB (residue names preserved), compute our
16 structural features vs the receptor (_protein.pdb). Filters reproduce PPI-Affinity: standard residues,
peptide len 3-40, ΔG ∈ [−14.4,−3.6], Kd/Ki only. Resumable JSONL; parallel across complexes.

Output: data/pdbbind_peptides.jsonl (one row/complex: pdb, seq, length, affinity_type, y, 16 features).
Run e109 after to pool with our 156, train, and grade vs PPI-Affinity (0.554/0.629) — learn the charged floor.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PL_INDEX = ROOT / "data/drive_pull/index/index/INDEX_general_PL.2020R1.lst"
PL_ROOT = ROOT / "data/drive_pull/pl/P-L"
OUT = ROOT / "data/pdbbind_peptides.jsonl"
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
RT = 0.5922
UNIT = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}
DIRMAP: dict[str, Path] = {}


def parse_aff(line):
    m = re.search(r"(Kd|Ki)[=~]([0-9.]+)([fpnumM]+)", line)
    if not m:
        return None
    typ, val, unit = m.group(1), float(m.group(2)), m.group(3)
    if unit not in UNIT or val <= 0:
        return None
    return typ, RT * log(val * UNIT[unit])


def mol2_to_pdb_seq(mol2: Path, out: Path):
    """Convert a PDBbind peptide ligand mol2 → PDB, deriving residue numbers from backbone.

    PDBbind mol2 stores subst_id=1 for every atom and subst_name=resname only (no residue number),
    so residue boundaries must be detected structurally. Each residue has exactly one backbone amide
    N, so we increment the residue counter on each new backbone 'N' atom.
    """
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []  # (atom_name, resname, x, y, z)
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            continue
        name, x, y, z, _typ, _sid, sname = f[1], f[2], f[3], f[4], f[5], f[6], f[7]
        rn = "".join(c for c in sname if c.isalpha()).upper()[:3]
        if rn not in AA3:
            return None  # non-standard residue → reject (peptidomimetic)
        try:
            atoms.append((name, rn, float(x), float(y), float(z)))
        except ValueError:
            return None
    # assign residue numbers: increment on each backbone 'N'; leading atoms (before first N) → residue 1
    rec, seq = [], []
    resnum = 0
    for aid, (name, rn, x, y, z) in enumerate(atoms, start=1):
        if name == "N":
            resnum += 1
            seq.append(rn)
        assigned = max(resnum, 1)
        rec.append(f"ATOM  {aid:>5} {name:<4} {rn:>3} A{assigned:>4}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
    if len(seq) < 3:
        return None
    out.write_text("\n".join(rec) + "\nEND\n")
    return "".join(AA3[r] for r in seq)


def process(args):
    pid, typ, dg = args
    d = DIRMAP.get(pid)
    if d is None:
        return None
    mol2 = d / f"{pid}_ligand.mol2"
    prot = d / f"{pid}_protein.pdb"
    if not (mol2.exists() and prot.exists()):
        return None
    pep = Path(f"/tmp/pdbbind_pep_{pid}.pdb")
    try:
        seq = mol2_to_pdb_seq(mol2, pep)
        if not seq or not (3 <= len(seq) <= 40):
            return None
        from hybridock_pep.scoring.geometry_features import compute_geometry_features, GEOMETRY_FEATURE_KEYS
        f = compute_geometry_features(pep, prot.resolve())
        if not f:
            return None
        row = {"pdb": pid, "dataset": "pdbbind", "affinity_type": typ, "seq": seq,
               "length": len(seq), "y": round(dg, 3)}
        row.update({k: f[k] for k in GEOMETRY_FEATURE_KEYS})
        return row
    except Exception:  # noqa: BLE001
        return None
    finally:
        if pep.exists():
            pep.unlink()


def main():
    # build dir map once
    for yd in PL_ROOT.iterdir():
        if yd.is_dir():
            for cd in yd.iterdir():
                if cd.is_dir():
                    DIRMAP[cd.name] = cd
    # parse peptide+Kd/Ki entries
    jobs = []
    for ln in PL_INDEX.read_text().splitlines():
        if ln.startswith("#") or "-mer)" not in ln:
            continue
        a = parse_aff(ln)
        if not a:
            continue
        typ, dg = a
        if not (-14.4 <= dg <= -3.6):
            continue
        pid = ln.split()[0]
        jobs.append((pid, typ, dg))
    print(f"=== E108 ingest: {len(jobs)} peptide+Kd/Ki entries in range; dir map {len(DIRMAP)} ===", flush=True)

    done = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try:
                done.add(json.loads(ln)["pdb"])
            except Exception:  # noqa: BLE001
                pass
    jobs = [j for j in jobs if j[0] not in done]
    print(f"  resuming: {len(done)} already done, {len(jobs)} to do", flush=True)

    n_ok = 0
    with open(OUT, "a") as fh, ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n_ok += 1
            if (i + 1) % 100 == 0:
                print(f"  processed {i+1}/{len(jobs)}, kept {n_ok}", flush=True)
    total = len(done) + n_ok
    print(f"=== done: {n_ok} new kept, {total} total in {OUT.name} ===", flush=True)


if __name__ == "__main__":
    main()
