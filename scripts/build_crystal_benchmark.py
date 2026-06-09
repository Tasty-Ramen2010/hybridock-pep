"""Build the crystal-pose affinity benchmark (overhaul step-1 keystone).

Intersects the clean Kd+Ki affinity set (data/eval_kd_ki_clean.json) with the
PepPC crystal structures we already have on disk
(datasets/training_formatted_peppc/peppc_<PDB>_<chain>/). Each surviving entry
has BOTH a measured Kd/Ki AND a crystal pocket + crystal peptide pose — so
scoring the native pose isolates the scoring function from docking error, which
is the whole point of a scoring benchmark.

Guards against the chain-mismatch trap (wrong peptide paired to an affinity):
records the PepPC peptide sequence and its length so a downstream sanity check
can flag entries whose peptide looks wrong for the measured complex.

Writes data/benchmark_crystal.json — a manifest consumed by
scripts/score_crystal_benchmark.py and scripts/benchmark_scoring.py.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PEPPC = ROOT / "datasets" / "training_formatted_peppc"
CLEAN = ROOT / "data" / "eval_kd_ki_clean.json"
OUT = ROOT / "data" / "benchmark_crystal.json"


def peppc_index() -> dict[str, Path]:
    """Map PDB-ID -> PepPC entry dir (first chain seen for that PDB)."""
    idx: dict[str, Path] = {}
    for d in glob.glob(str(PEPPC / "peppc*")):
        m = re.match(r"peppc[f]?_([0-9A-Za-z]{4})_", os.path.basename(d))
        if m:
            idx.setdefault(m.group(1).upper(), Path(d))
    return idx


def read_peptide_seq(entry_dir: Path) -> str | None:
    for f in entry_dir.glob("*_peptide_sequence"):
        return f.read_text().strip()
    return None


def main() -> None:
    idx = peppc_index()
    clean = json.loads(CLEAN.read_text())
    rows = []
    for r in clean:
        pid = (r.get("pdb") or "").upper()
        entry = idx.get(pid)
        if entry is None:
            continue
        pocket = next(iter(entry.glob("*_protein_pocket.pdb")), None)
        peptide = next(iter(entry.glob("*_peptide.pdb")), None)
        if pocket is None or peptide is None:
            continue
        seq = read_peptide_seq(entry) or ""
        rows.append({
            "pdb": pid,
            "affinity_type": r.get("affinity_type"),
            "pkd": r.get("pkd"),
            "dg_exp": r.get("dg_exp"),
            "peptide_seq": seq,
            "peptide_len": len(seq),
            # precomputed docked-pose features (for cross-check vs crystal rescoring)
            "vina_docked": r.get("vina"),
            "pocket_pdb": str(pocket.relative_to(ROOT)),
            "peptide_pdb": str(peptide.relative_to(ROOT)),
        })

    rows.sort(key=lambda x: x["pdb"])
    OUT.write_text(json.dumps(rows, indent=2))

    lens = [r["peptide_len"] for r in rows if r["peptide_len"]]
    pkds = [r["pkd"] for r in rows if r["pkd"] is not None]
    print(f"Crystal benchmark: {len(rows)} complexes → {OUT.relative_to(ROOT)}")
    print(f"  peptide length: min={min(lens)} max={max(lens)} median={sorted(lens)[len(lens)//2]}")
    print(f"  pKd range: {min(pkds):.2f}–{max(pkds):.2f} (ΔG {-1.3633*max(pkds):.1f}..{-1.3633*min(pkds):.1f})")
    types: dict[str, int] = {}
    for r in rows:
        types[r["affinity_type"]] = types.get(r["affinity_type"], 0) + 1
    print(f"  affinity types: {types}")
    # Flag suspicious entries (empty/very short peptide = possible chain mismatch)
    sus = [r["pdb"] for r in rows if r["peptide_len"] < 4]
    if sus:
        print(f"  ⚠ {len(sus)} entries with peptide <4 res (verify chain pairing): {sus}")


if __name__ == "__main__":
    main()
