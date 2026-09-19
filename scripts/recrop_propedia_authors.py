#!/usr/bin/env python
"""Re-crop Propedia receptors with the RAPiDock AUTHORS' pocket rule.

Why: our Propedia pockets (datasets/propedia_formatted/*_protein_pocket.pdb) keep only the
residues that have an atom within 20 A of the peptide -- a residue-by-residue cut that leaves
the receptor full of holes. The authors' third_party/RAPiDock/pocket_trunction.py (level
"Residue", threshold 20, threshold_keep 5) instead takes, per chain, the MIN..MAX residue
number of the residues in range and keeps EVERY residue in between: contiguous segments,
gaps filled. RefPepDB -- the distribution rapidock_local.pt was trained and tested on --
was cropped that way, which is why its pockets look like ~33 A crops against our strict 20 A
and are ~2x larger at matched peptide length. At <=25 aa, Propedia scores ~1.3 A worse than
RefPepDB; this removes the preparation difference so that gap can be measured cleanly.

Held fixed: the crystal PEPTIDE file is copied unchanged from propedia_formatted, so RMSD
targets are identical and old-vs-new results pair complex by complex. The zip's peptide
coordinates are checked against that file; a complex whose frames disagree is refused.

Deliberate deviation from the authors' code: only standard residues (hetero flag ' ') are
searched and saved, so crystal waters/ligands whose residue numbers fall inside a kept range
are not swept into the receptor.

Output goes to a NEW directory (datasets/propedia_formatted_authors/). RAPiDock caches graphs
next to the protein file keyed only by complex name, so writing in place could hand back a
graph built from the old pocket.

pocket_trunction.py runs argparse at import time, so its Residue-level logic is reproduced in
_authors_residue_pocket rather than imported.

Usage (rapidock env, Biopython):
    python scripts/recrop_propedia_authors.py --set bench   # in-scope bench propedia complexes
    python scripts/recrop_propedia_authors.py --set train   # in-scope training propedia complexes
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import statistics as st
import warnings
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from Bio.PDB import PDBIO, NeighborSearch, PDBParser, Select  # noqa: E402
from Bio.PDB.Model import Model  # noqa: E402
from Bio.PDB.Structure import Structure  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "datasets" / "propedia" / "complex2_3.zip"
OLD = ROOT / "datasets" / "propedia_formatted"
NEW = ROOT / "datasets" / "propedia_formatted_authors"
THRESHOLD = 20.0       # pocket_trunction.py default `threshold`
THRESHOLD_KEEP = 5.0   # pocket_trunction.py default `threshold_keep`
FRAME_TOL = 0.05       # A: zip peptide must coincide with the benchmark peptide


class _Keep(Select):
    """Select residues by (chain id, residue id) -- NOT object identity, which changes when
    a chain is moved between Biopython structures."""

    def __init__(self, keys: set) -> None:
        self.keys = keys

    def accept_residue(self, residue) -> bool:  # noqa: D102
        return (residue.get_parent().id, residue.id) in self.keys


def _authors_residue_pocket(model, peptide_coords: np.ndarray) -> set:
    """pocket_trunction(level="Residue", exclude_chain=None) on standard residues.

    For each chain: the residues with any atom within THRESHOLD (far) / THRESHOLD_KEEP (near)
    of any peptide atom define a residue-number range [min, max]; every standard residue of
    that chain numbered inside the range is kept. Near and far ranges are unioned exactly as
    in the original (with no exclude_chain, near is always inside far; kept for fidelity).
    """
    atoms = [a for a in model.get_atoms() if a.get_parent().id[0] == " "]
    ns = NeighborSearch(atoms)
    keys: set = set()
    for cutoff in (THRESHOLD_KEEP, THRESHOLD):
        hit: dict[str, list[int]] = {}
        for c in peptide_coords:
            for a in ns.search(c, cutoff):
                res = a.get_parent()
                hit.setdefault(res.get_parent().id, []).append(res.id[1])
        for chain_id, nums in hit.items():
            lo, hi = min(nums), max(nums)
            for res in model[chain_id]:
                if res.id[0] == " " and lo <= res.id[1] <= hi:
                    keys.add((chain_id, res.id))
    return keys


def _coords(path: Path) -> np.ndarray:
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                out.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                pass
    return np.array(out, dtype=float)


def _ca_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA"
    )


def recrop(name: str) -> dict:
    """Rebuild one complex's receptor pocket. Returns a status record; never raises."""
    rec: dict = {"name": name}
    try:
        _, pdb, pep = name.split("_", 2)
        old_dir = OLD / name
        old_pep = old_dir / f"{name}_peptide.pdb"
        old_rec = old_dir / f"{name}_protein_pocket.pdb"
        if not old_pep.exists():
            return rec | {"status": "no-old-peptide"}
        bench_pep = _coords(old_pep)
        if len(bench_pep) == 0:
            return rec | {"status": "empty-old-peptide"}

        zf = zipfile.ZipFile(ZIP)
        # zip entries are complex/{pdb}_{pepChain}_{recChain}.pdb (verified 6xwv_E_A/B/C for
        # peptide chain E). format_propedia.py's docstring states the order reversed.
        entries = sorted(n for n in zf.namelist() if n.startswith(f"complex/{pdb.lower()}_{pep}_"))
        if not entries:
            return rec | {"status": "not-in-zip"}

        parser = PDBParser(QUIET=True)
        receptor = Structure("receptor")
        rmodel = Model(0)
        receptor.add(rmodel)
        zip_pep = None
        for entry in entries:
            s = parser.get_structure(entry, io.StringIO(zf.read(entry).decode(errors="ignore")))
            m = next(iter(s))
            for chain in list(m):
                if chain.id == pep:
                    if zip_pep is None:
                        zip_pep = np.array([a.get_coord() for a in chain.get_atoms()], dtype=float)
                    continue
                if chain.id in rmodel:
                    continue  # the same receptor chain appears in several pair files
                chain.detach_parent()
                rmodel.add(chain)
        if zip_pep is None or len(rmodel) == 0:
            return rec | {"status": "no-receptor-or-peptide-in-zip"}

        # Frame check: every benchmark peptide atom must sit on a zip peptide atom.
        d = np.sqrt(((bench_pep[:, None, :] - zip_pep[None, :, :]) ** 2).sum(-1)).min(1)
        dev = float(d.max())
        if dev > FRAME_TOL:
            return rec | {"status": "frame-mismatch", "frame_dev": round(dev, 3)}

        keys = _authors_residue_pocket(rmodel, bench_pep)
        if not keys:
            return rec | {"status": "empty-pocket"}

        new_dir = NEW / name
        new_dir.mkdir(parents=True, exist_ok=True)
        new_rec = new_dir / f"{name}_protein_pocket.pdb"
        io_obj = PDBIO()
        io_obj.set_structure(receptor)
        io_obj.save(str(new_rec), _Keep(keys))
        shutil.copy2(old_pep, new_dir / f"{name}_peptide.pdb")
        seq = old_dir / f"{name}_peptide_sequence"
        if seq.exists():
            shutil.copy2(seq, new_dir / seq.name)
        return rec | {
            "status": "ok",
            "old_ca": _ca_count(old_rec),
            "new_ca": _ca_count(new_rec),
            "frame_dev": round(dev, 4),
            "rec_chains": len(rmodel),
        }
    except Exception as exc:  # noqa: BLE001 -- per-complex failure must not kill the batch
        return rec | {"status": f"error: {type(exc).__name__}: {exc}"}


def _swap(row: dict, rec_col: str, pep_col: str, name_col: str, ok: set) -> dict:
    n = row[name_col]
    if n in ok:
        row = dict(row)
        row[rec_col] = str(NEW / n / f"{n}_protein_pocket.pdb")
        row[pep_col] = str(NEW / n / f"{n}_peptide.pdb")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=["bench", "train"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if a.set == "bench":
        src = ROOT / "data" / "bench_long_scope25.csv"
        rows = list(csv.DictReader(open(src)))
        name_col, rec_col, pep_col = "name", "receptor", "peptide_pdb"
    else:
        src = ROOT / "data" / "longft_train_scope25.csv"
        rows = list(csv.DictReader(open(src)))
        name_col, rec_col, pep_col = "complex_name", "protein_description", "peptide_description"
    names = sorted({r[name_col] for r in rows if r[name_col].startswith("propedia")})
    print(f"{a.set}: {len(names)} propedia complexes to re-crop ({a.workers} workers)")

    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(recrop, names))

    status = Counter(r["status"] for r in results)
    print(f"status: {dict(status)}")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        oc = [r["old_ca"] for r in ok]
        nc = [r["new_ca"] for r in ok]
        print(f"receptor Ca: old median {st.median(oc):.0f} -> new median {st.median(nc):.0f} "
              f"(x{st.median(n / max(o, 1) for o, n in zip(oc, nc)):.2f} per complex)")
        print(f"frame deviation: max {max(r['frame_dev'] for r in ok):.4f} A")
    for r in results:
        if r["status"] != "ok":
            print(f"   {r['name']}: {r['status']}")

    okset = {r["name"] for r in ok}
    flds = list(rows[0].keys())
    if a.set == "bench":
        outs = {
            ROOT / "data" / "bench_long_scope25_authorsP.csv": rows,
            ROOT / "data" / "bench_propedia25_authors.csv":
                [r for r in rows if r[name_col] in okset],
            ROOT / "data" / "bench_propedia25_old.csv":
                [r for r in rows if r[name_col] in okset],
        }
        for path, sel in outs.items():
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=flds)
                w.writeheader()
                for r in sel:
                    w.writerow(r if path.name.endswith("_old.csv")
                               else _swap(r, rec_col, pep_col, name_col, okset))
            print(f"wrote {path.relative_to(ROOT)} ({len(sel)} rows)")
    else:
        out = ROOT / "data" / "longft_train_scope25_authorsP.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds)
            w.writeheader()
            for r in rows:
                w.writerow(_swap(r, rec_col, pep_col, name_col, okset))
        print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows, {len(okset)} complexes re-pointed)")
        print("NOTE: receptor sequences changed for the re-pointed complexes -> their ESM "
              "embeddings must be recomputed before training on this CSV.")


if __name__ == "__main__":
    main()
