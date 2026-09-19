#!/usr/bin/env python
"""Assemble a de novo designed protein + peptide training set.

WHY.  Our finetune set spans 3,020 unique PDB entries and exactly 2 of them (7dng, 8gjg)
are de novo designed protein/peptide complexes -- 0.07%.  That is why we fail on 9CCE
(DYNA_1b7, a designed IDR binder): the model has essentially never seen this geometry,
a long pseudo-symmetric designed groove holding an extended peptide.

LEAKAGE CONTROL, READ BEFORE USING THIS SET.  Brian Coventry's challenge came with
"hopefully you didn't find these sequences when you were looking for your training data".
The designed-complex corner of the PDB is small and heavily Baker-lab, so training on it
naively would pull in the very structures we are being tested on.  Three guards:

  1. HARD EXCLUDE the challenge paper's own depositions (9CCE, 9CCF).
  2. HARD EXCLUDE any entry whose primary citation is Wu et al. Science 2025 (adr8063),
     caught by DOI and by title, so sibling depositions from the same paper go too.
  3. Record every entry's citation so a reviewer can audit what went in.

Additionally the caller should hold out a random fraction as a designed-complex TEST set;
improvement has to be demonstrated on designed complexes the model never saw, not on 9CCE
alone (n=1 proves nothing).

Stage 1 (this script, network only): resolve the candidate list and metadata.
Stage 2: download, pair chains by contact, crop with the authors' pocket rule.

Usage: build_denovo_set.py [--stage list|build] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
OUT = ROOT / "datasets/denovo_pep"
META = ROOT / "data/denovo_pep_candidates.json"
EXCLUDE_IDS = {"9cce", "9ccf"}
EXCLUDE_DOI = {"10.1126/science.adr8063"}
EXCLUDE_TITLE_BITS = ("intrinsically disordered region binding protein",)
MIN_PEP, MAX_PEP = 4, 25
MIN_REC = 50


def _post(url: str, payload: dict, timeout: int = 90):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def search_ids() -> list[str]:
    denovo = {"type": "terminal", "service": "text", "parameters": {
        "attribute": "struct_keywords.pdbx_keywords",
        "operator": "contains_phrase", "value": "DE NOVO PROTEIN"}}
    short = {"type": "terminal", "service": "text", "parameters": {
        "attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
        "value": {"from": MIN_PEP, "to": MAX_PEP,
                  "include_lower": True, "include_upper": True}}}
    d = _post("https://search.rcsb.org/rcsbsearch/v2/query",
              {"query": {"type": "group", "logical_operator": "and",
                         "nodes": [denovo, short]},
               "return_type": "entry",
               "request_options": {"paginate": {"start": 0, "rows": 2000}}})
    return [h["identifier"].lower() for h in d.get("result_set", [])]


GQL = """query($ids:[String!]!){ entries(entry_ids:$ids){
  rcsb_id
  struct{title}
  rcsb_accession_info{initial_release_date}
  rcsb_entry_info{ resolution_combined polymer_entity_count_protein }
  citation{ id title pdbx_database_id_DOI year }
  polymer_entities{
    entity_poly{ rcsb_sample_sequence_length pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
    rcsb_polymer_entity_container_identifiers{ auth_asym_ids }
  }
} }"""


def fetch_meta(ids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        try:
            d = _post("https://data.rcsb.org/graphql",
                      {"query": GQL, "variables": {"ids": [c.upper() for c in chunk]}})
            out += [e for e in (d.get("data", {}).get("entries") or []) if e]
        except Exception as exc:  # noqa: BLE001 -- keep going, report at the end
            print(f"  chunk {i}: {type(exc).__name__}", flush=True)
        time.sleep(0.2)
        if (i // 40) % 3 == 0:
            print(f"  metadata {min(i + 40, len(ids))}/{len(ids)}", flush=True)
    return out


def classify(e: dict) -> dict | None:
    """Keep entries that pair a >=50 aa protein with a 4-25 aa peptide."""
    pid = e["rcsb_id"].lower()
    ents = e.get("polymer_entities") or []
    peps, recs = [], []
    for pe in ents:
        ep = pe.get("entity_poly") or {}
        if (ep.get("rcsb_entity_polymer_type") or "") != "Protein":
            continue
        n = ep.get("rcsb_sample_sequence_length") or 0
        ch = (pe.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
        rec = {"len": n, "chains": ch,
               "seq": (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")}
        if MIN_PEP <= n <= MAX_PEP:
            peps.append(rec)
        elif n >= MIN_REC:
            recs.append(rec)
    if not peps or not recs:
        return None
    cit = (e.get("citation") or [{}])[0] or {}
    doi = (cit.get("pdbx_database_id_DOI") or "").lower()
    title = (cit.get("title") or "").lower()
    excl = None
    if pid in EXCLUDE_IDS:
        excl = "challenge paper deposition"
    elif doi in EXCLUDE_DOI:
        excl = f"challenge paper DOI {doi}"
    elif any(b in title for b in EXCLUDE_TITLE_BITS):
        excl = "challenge paper title match"
    res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    return {"pdb": pid, "title": (e.get("struct") or {}).get("title"),
            "date": (e.get("rcsb_accession_info") or {}).get("initial_release_date", "")[:10],
            "resolution": res[0] if res else None, "doi": doi,
            "citation_title": cit.get("title"), "year": cit.get("year"),
            "peptides": peps, "receptors": recs, "excluded": excl}


AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
POCKET_CUT = 20.0


def _atoms(path, chain=None):
    out = []
    for l in open(path, errors="replace"):
        if l.startswith("ATOM") and l[76:78].strip() != "H" and l[16] in (" ", "A"):
            if chain is None or l[21] == chain:
                out.append(l)
    return out


def _xyz(lines):
    import numpy as np
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lines])


def _seq(lines):
    return "".join(AA3.get(l[17:20].strip(), "X") for l in lines if l[12:16].strip() == "CA")


def build_one(pdb: str) -> dict:
    """Download one entry, pair chains by contact, crop with the authors' pocket rule."""
    import numpy as np
    from scipy.spatial import cKDTree
    d = OUT / pdb
    d.mkdir(parents=True, exist_ok=True)
    raw = d / f"{pdb}.pdb"
    if not raw.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", raw)

    chains: dict[str, list] = {}
    for l in _atoms(raw):
        chains.setdefault(l[21], []).append(l)
    nres = {c: sum(1 for l in ls if l[12:16].strip() == "CA") for c, ls in chains.items()}
    peps = [c for c in chains if MIN_PEP <= nres[c] <= MAX_PEP]
    recs = [c for c in chains if nres[c] >= MIN_REC]
    if not peps or not recs:
        return {"pdb": pdb, "fail": "no peptide/receptor chain pair"}

    best = None
    for p in peps:
        for r in recs:
            n = int((cKDTree(_xyz(chains[r])).query(_xyz(chains[p]))[0] < 4.5).sum())
            if best is None or n > best[2]:
                best = (r, p, n)
    r, p, ncon = best
    if ncon < 15:
        return {"pdb": pdb, "fail": f"weak interface ({ncon} contacts)"}

    pxyz = _xyz(chains[p])
    tree = cKDTree(pxyz)
    near = {int(l[22:26]) for l in chains[r]
            if tree.query(_xyz([l]))[0][0] <= POCKET_CUT}
    if len(near) < 20:
        return {"pdb": pdb, "fail": f"pocket too small ({len(near)})"}
    lo, hi = min(near), max(near)
    pocket = [l for l in chains[r] if lo <= int(l[22:26]) <= hi]

    pep_f = d / f"{pdb}_peptide.pdb"
    rec_f = d / f"{pdb}_protein_pocket.pdb"
    for lines, f in ((chains[p], pep_f), (pocket, rec_f)):
        with f.open("w") as fh:
            fh.writelines(lines)
            fh.write("END\n")
    seq = _seq(chains[p])
    return {"pdb": pdb, "complex_name": f"denovo_{pdb}",
            "protein_description": str(rec_f), "peptide_description": str(pep_f),
            "source": "denovo_pdb", "pep_len": len(seq), "seq": seq,
            "ss_class": "UNLABELLED", "contacts": ncon,
            "pocket_residues": sum(1 for l in pocket if l[12:16].strip() == "CA"),
            "length_bucket": ("05-08" if len(seq) <= 8 else "09-12" if len(seq) <= 12
                              else "13-16" if len(seq) <= 16 else "17-25")}


def build() -> None:
    import csv as _csv
    from concurrent.futures import ThreadPoolExecutor
    data = json.loads(META.read_text())
    keep = data["kept"]
    print(f"building {len(keep)} designed complexes -> {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fails = [], []
    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(lambda r: _safe(r["pdb"]), keep):
            (fails if "fail" in res else rows).append(res)
    print(f"built {len(rows)}, failed {len(fails)}")
    for f in fails[:15]:
        print(f"   {f['pdb']}: {f['fail']}")
    if rows:
        out_csv = ROOT / "data/denovo_pep_all.csv"
        cols = ["complex_name", "protein_description", "peptide_description", "source",
                "pep_len", "ss_class", "length_bucket", "seq", "pdb", "contacts",
                "pocket_residues"]
        with out_csv.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        import statistics as st
        print(f"wrote {out_csv}")
        print(f"  peptide length: median {st.median(r['pep_len'] for r in rows)}, "
              f"range {min(r['pep_len'] for r in rows)}-{max(r['pep_len'] for r in rows)}")
        print(f"  pocket residues: median {st.median(r['pocket_residues'] for r in rows)}")
    print("DENOVO_BUILD_DONE")


def _safe(pdb: str) -> dict:
    try:
        return build_one(pdb)
    except Exception as exc:  # noqa: BLE001 -- one bad entry must not stop the build
        return {"pdb": pdb, "fail": f"{type(exc).__name__}: {str(exc)[:70]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="list", choices=["list", "build"])
    a = ap.parse_args()
    if a.stage == "build":
        build()
        return

    if a.stage == "list":
        ids = search_ids()
        print(f"candidate de novo entries with a {MIN_PEP}-{MAX_PEP} aa chain: {len(ids)}",
              flush=True)
        meta = fetch_meta(ids)
        print(f"metadata fetched for {len(meta)}", flush=True)
        rows = [r for r in (classify(e) for e in meta) if r]
        keep = [r for r in rows if not r["excluded"]]
        drop = [r for r in rows if r["excluded"]]
        META.write_text(json.dumps({"kept": keep, "excluded": drop}, indent=2))
        print(f"\nprotein+peptide complexes: {len(rows)}")
        print(f"  EXCLUDED by leakage guard: {len(drop)}")
        for r in drop:
            print(f"    {r['pdb']}  {r['excluded']}  {(r['title'] or '')[:50]}")
        print(f"  KEPT: {len(keep)}")
        yrs = {}
        for r in keep:
            yrs[r["year"]] = yrs.get(r["year"], 0) + 1
        print("  by year:", dict(sorted((k, v) for k, v in yrs.items() if k)))
        print(f"\nwrote {META}")
        print("DENOVO_LIST_DONE")


if __name__ == "__main__":
    main()
