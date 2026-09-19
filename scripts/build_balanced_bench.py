#!/usr/bin/env python
"""Build a length-balanced, leak-free docking benchmark from post-2020 PDB structures.

Why: RAPiDock's own RecentSet is almost all short/medium peptides (18 of our 345 held-out
complexes are >12 aa), so it cannot show whether a long-peptide finetune helps.

Rules (every complex must satisfy all):
  * deposited on/after 2020-09-21 (after RAPiDock's training cutoff) -> unseen by the original
  * PDB id not in any of our finetune CSVs (data/longft_*.csv)       -> unseen by ours
  * not already in the RecentSet held-out bench                        -> independent set
  * X-ray, resolution <= 3.0 A, peptide 5-25 aa, standard residues only
  * peptide fully resolved (CA count == manifest length); one entry per unique peptide sequence
  * receptor cropped with the RAPiDock authors' contiguous-segment rule (pocket_trunction.py,
    Residue level, 20 A / 5 A) -- the same preparation the pretrained model was trained on
Target: PER_BAND complexes in each of 5-8 / 9-12 / 13-16 / 17-25 aa, best resolution first,
at most 2 per receptor.

Usage (rapidock env): python scripts/build_balanced_bench.py [per_band]
Output: datasets/balanced_post2020/<name>/{peptide,protein_pocket}.pdb
        data/bench_balanced_post2020.csv
"""
from __future__ import annotations

import csv
import glob
import io
import math
import re
import sys
import time
import urllib.request
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser, PDBIO, Select  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recrop_propedia_authors import _authors_residue_pocket  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "balanced_post2020"
RAW = OUT / "_raw"
CSV_OUT = ROOT / "data" / "bench_balanced_post2020.csv"
PER_BAND = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BANDS = ("05-08", "09-12", "13-16", "17-25")
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def band(n: int) -> str:
    return "05-08" if n <= 8 else "09-12" if n <= 12 else "13-16" if n <= 16 else "17-25"


def excluded_ids() -> set[str]:
    ids: set[str] = set()
    for f in glob.glob(str(ROOT / "data" / "longft_*.csv")):
        for r in csv.DictReader(open(f)):
            for s in (r["complex_name"], Path(r["protein_description"]).parent.name):
                m = re.search(r"([0-9][a-z0-9]{3})", s.lower())
                if m:
                    ids.add(m.group(1))
    for r in csv.DictReader(open(ROOT / "data" / "bench_recentset_heldout.csv")):
        ids.add(r["name"].lower()[:4])
    return ids


def candidates() -> dict[str, list[dict]]:
    skip = excluded_ids()
    rows = []
    for m in ("datasets/pdb_2019_2023/manifest.csv", "datasets/pdb_2024_2026/manifest.csv"):
        for r in csv.DictReader(open(ROOT / m)):
            if r.get("excluded_reason") or r.get("peptide_nonstd") or r["method"] != "X-ray":
                continue
            try:
                L = int(r["peptide_len"])
                res = float(r["resolution_A"].strip("[]").split(",")[0])
            except ValueError:
                continue
            pid = r["pdb_id"].lower()
            # Manifest length is the deposited sequence; the resolved stretch is often shorter
            # (disordered termini), so admit up to 40 and band on what is actually modelled.
            if not 5 <= L <= 40 or res > 3.0 or r["deposition_date"][:10] < "2020-09-21" or pid in skip:
                continue
            rows.append(dict(pdb=pid, pep_chain=r["peptide_chain"], rec_chain=r["receptor_chain"],
                             seq=r["peptide_seq"], L=L, res=res, date=r["deposition_date"][:10],
                             rec_key=r.get("receptor_seq_md5") or f"{pid}_{r['receptor_chain']}"))
    by_seq: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r["res"]):
        by_seq.setdefault(r["seq"], r)
    pools: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(by_seq.values(), key=lambda r: r["res"]):
        pools[band(r["L"])].append(r)
    return pools


def fetch(pid: str) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"{pid}.cif"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://files.rcsb.org/download/{pid}.cif", timeout=60) as u:
                dest.write_bytes(u.read())
            return dest
        except Exception:  # noqa: BLE001 -- network; retry then give up on this entry
            time.sleep(2 + 3 * attempt)
    return None


class _Chain(Select):
    def __init__(self, chain_id: str, keys: set | None = None) -> None:
        self.chain_id, self.keys = chain_id, keys

    def accept_chain(self, chain) -> bool:  # noqa: D102
        return self.keys is not None or chain.id == self.chain_id

    def accept_residue(self, residue) -> bool:  # noqa: D102
        if residue.id[0] != " " or residue.get_resname() not in AA3:
            return False
        if self.keys is not None:
            return (residue.get_parent().id, residue.id) in self.keys
        return True

    def accept_atom(self, atom) -> bool:  # noqa: D102
        return atom.element != "H"


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def ss_label(residues: list) -> str:
    """HELIX / SHEET / UNUSUAL from backbone phi/psi (majority of interior residues)."""
    h = e = p = n = 0
    for i in range(1, len(residues) - 1):
        try:
            a, b, c = residues[i - 1], residues[i], residues[i + 1]
            phi = _dihedral(a["C"].coord, b["N"].coord, b["CA"].coord, b["C"].coord)
            psi = _dihedral(b["N"].coord, b["CA"].coord, b["C"].coord, c["N"].coord)
        except KeyError:
            continue
        n += 1
        if -160 < phi < -20 and -120 < psi < 50:
            h += 1
        elif -180 <= phi < -100 and (psi > 90 or psi < -150):
            e += 1
        elif -100 <= phi < -45 and psi > 90:
            p += 1
    if n == 0:
        return "UNUSUAL"
    # The first build lumped extended strands (phi < -100) and polyproline-II (phi -100..-45)
    # together as SHEET, which labelled 49 of 60 short peptides SHEET. Split them.
    if h / n >= 0.5:
        return "HELIX"
    if e / n >= 0.5:
        return "SHEET"
    if (e + p) / n >= 0.5:
        return "PPII" if p >= e else "SHEET"
    return "UNUSUAL"


def build_one(c: dict) -> dict | None:
    name = f"{c['pdb']}_{c['pep_chain']}"
    path = fetch(c["pdb"])
    if path is None:
        return {"name": name, "fail": "download"}
    try:
        model = next(iter(MMCIFParser(QUIET=True).get_structure(c["pdb"], str(path))))
        if c["pep_chain"] not in model or c["rec_chain"] not in model:
            return {"name": name, "fail": "chain missing"}
        # Target = the longest UNBROKEN resolved stretch: standard residues with full backbone,
        # each peptide-bonded to the next (C-N < 2.0 A). Disordered termini are dropped (the
        # usual practice), but an internal gap splits the chain -- docking a gapped peptide as
        # one chain would give the model a different molecule from the crystal.
        chain = [r for r in model[c["pep_chain"]] if r.id[0] == " "]
        runs, cur = [], []
        for r in chain:
            ok = r.get_resname() in AA3 and all(a in r for a in ("N", "CA", "C", "O"))
            if ok and cur and np.linalg.norm(cur[-1]["C"].coord - r["N"].coord) < 2.0:
                cur.append(r)
            else:
                if cur:
                    runs.append(cur)
                cur = [r] if ok else []
        if cur:
            runs.append(cur)
        pep_res = max(runs, key=len) if runs else []
        if not 5 <= len(pep_res) <= 25:
            return {"name": name, "fail": f"resolved stretch {len(pep_res)} aa outside 5-25"}
        c = dict(c, L=len(pep_res))
        keep_ids = {r.id for r in pep_res}
        for r in [r for r in model[c["pep_chain"]] if r.id not in keep_ids]:
            model[c["pep_chain"]].detach_child(r.id)
        pep_xyz = np.array([a.coord for r in pep_res for a in r if a.element != "H"], dtype=float)
        # receptor = the manifest receptor chain only (no other copies / partners)
        for ch in [ch for ch in model if ch.id not in (c["rec_chain"], c["pep_chain"])]:
            model.detach_child(ch.id)
        keys = {k for k in _authors_residue_pocket(model, pep_xyz) if k[0] == c["rec_chain"]}
        if len(keys) < 20:
            return {"name": name, "fail": f"pocket too small ({len(keys)})"}
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        io_ = PDBIO()
        io_.set_structure(model)
        io_.save(str(d / f"{name}_peptide.pdb"), _Chain(c["pep_chain"]))
        io_.save(str(d / f"{name}_protein_pocket.pdb"), _Chain(c["rec_chain"], keys))
        return {"name": name, "receptor": str(d / f"{name}_protein_pocket.pdb"),
                "peptide_pdb": str(d / f"{name}_peptide.pdb"),
                "seq": "".join(AA3[r.get_resname()] for r in pep_res), "pep_len": c["L"],
                "ss_class": ss_label(pep_res), "length_bucket": band(c["L"]),
                "source": "pdb_post2020", "resolution": c["res"], "deposited": c["date"],
                "rec_key": c["rec_key"], "pocket_residues": len(keys)}
    except Exception as exc:  # noqa: BLE001 -- one bad mmCIF must not kill the build
        return {"name": name, "fail": f"{type(exc).__name__}: {str(exc)[:60]}"}


def main() -> None:
    pools = candidates()
    allc = sorted((c for b in BANDS for c in pools[b]), key=lambda c: c["res"])
    print(f"candidates (manifest 5-40 aa): {len(allc)}", flush=True)
    # The band is only known AFTER building (resolved length), so build every candidate,
    # then fill each band best-resolution-first, at most 2 per receptor, unique sequence.
    built, fails = [], Counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(build_one, allc):
            if res is None:
                continue
            if "fail" in res:
                fails[re.sub(r"\s*\d.*$", "", res["fail"])] += 1
            else:
                built.append(res)
    print(f"built {len(built)}", flush=True)
    chosen, seen_seq = [], set()
    for b in BANDS:
        got, per_rec = [], Counter()
        for r in sorted((r for r in built if r["length_bucket"] == b), key=lambda r: r["resolution"]):
            if r["seq"] in seen_seq or per_rec[r["rec_key"]] >= 2:
                continue
            got.append(r)
            seen_seq.add(r["seq"])
            per_rec[r["rec_key"]] += 1
            if len(got) == PER_BAND:
                break
        print(f"  band {b}: kept {len(got)}", flush=True)
        chosen += got
    flds = ["name", "receptor", "peptide_pdb", "seq", "pep_len", "ss_class", "length_bucket",
            "source", "resolution", "deposited", "rec_key", "pocket_residues"]
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        for r in chosen:
            w.writerow(r)
    print(f"\nwrote {CSV_OUT.relative_to(ROOT)}: {len(chosen)} complexes")
    print("by band:", dict(Counter(r["length_bucket"] for r in chosen)))
    print("by SS  :", dict(Counter(r["ss_class"] for r in chosen)))
    print("band x SS:", dict(Counter((r["length_bucket"], r["ss_class"]) for r in chosen)))
    print("rejections:", dict(fails))


if __name__ == "__main__":
    main()
