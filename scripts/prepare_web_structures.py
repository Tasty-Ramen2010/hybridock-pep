#!/usr/bin/env python3
"""Trim RCSB structures for HybriDock-Pep and work out a sensible default binding site.

Run once (needs internet, standard library only):

    python3 scripts/prepare_web_structures.py

What it does for each protein listed in PROTEINS below:
  1. downloads  https://files.rcsb.org/download/<ID>.pdb
  2. keeps heavy ATOM records of the chosen chains (first model, first alt-location)
  3. works out a *suggested* binding site = centre of whatever is bound in the crystal
     (a ligand, a bound peptide) or of a named motif; this is the "I know where it binds" default
  4. writes data/pdb/<ID>.pdb and data/proteins.json

The suggested site is a starting point, not a prediction.  The UI says so.
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "src" / "hybridock_pep" / "web" / "static"  # data/pdb and data/proteins.json are written here

# site = ("chain", first_residue, last_residue)  -> centre of that stretch (ATOM records)
#      = ("het", "LIG")                           -> centre of a bound ligand (HETATM records)
#      = ("peptide-chain", ["C", "D"])            -> centre of the bound peptide chain nearest the kept chains
#      = ("fixed", x, y, z)                       -> a site validated elsewhere (the backend's /api/examples)
PROTEINS = [
    dict(id="5O3L", key="tau", name="Tau K18", keep="ABCDE",
         about="Tangle-forming protein in Alzheimer's disease",
         siteNote="the VQIVYK stretch (residues 306–311)", site=("chain", "C", 306, 311),
         example="LIYKWVNK"),
    dict(id="1YCR", key="mdm2", name="MDM2", keep="A",
         about="A protein that switches off the tumour-suppressor p53",
         siteNote="the groove where p53's peptide sits in the crystal", site=("fixed", 25.20, -25.61, -7.97),
         example="ETFSDLWKLLPE", backend="mdm2"),
    # One layer of the fibril (chain A). The five-chain stack leaves no room for a peptide: every pose clashed and a real
    # run reported "no ΔG" (found by docking every built-in protein through the browser UI).
    dict(id="2N0A", key="asyn", name="α-synuclein", keep="A",
         about="Clumps up in Parkinson's disease (one layer of the fibril)",
         siteNote="the NAC core stretch (residues 61–72) on one fibril layer", site=("chain", "A", 61, 72),
         example="KTKEGVL"),
    dict(id="6OIM", key="kras", name="KRAS G12C", keep="A",
         about="A cancer-driving switch protein",
         siteNote="the pocket where the drug sotorasib sits in the crystal", site=("het", "MOV"),
         example="LVVVGACGV"),
    dict(id="1M17", key="egfr", name="EGFR kinase", keep="A",
         about="A growth-signal enzyme targeted in lung cancer",
         siteNote="the ATP pocket where erlotinib sits in the crystal", site=("het", "AQ4"),
         example="EEEEYFELV"),
    dict(id="2XA0", key="bcl2", name="BCL-2", keep="A",
         about="A protein that helps cancer cells survive",
         siteNote="the groove where a BAX peptide sits in the crystal", site=("peptide-chain", ["C", "D"]),
         example="LSECLKRIGDELDS"),
]

#: Structures that already live in the repo (data/pdbs). The backend docks against exactly these files, so the
#: browser shows the identical coordinates. Only ATOM records are kept (no waters, no ligands).
LOCAL = [
    dict(id="1T2D", key="pfldh", name="PfLDH", file="data/pdbs/1T2D_receptor.pdb",
         about="The malaria parasite's lactate dehydrogenase, a drug target",
         siteNote="the enzyme's active site, where the team docks", site=("fixed", 24.84, 22.73, 41.69),
         example="LISDAELEAIFEADC", backend="pfldh"),
    dict(id="1I0Z", key="hldh", name="Human LDH", file="data/pdbs/1I0Z.pdb",
         about="Our human off-target: the version we want the peptide to ignore",
         siteNote="the enzyme's active site, where the team docks", site=("fixed", 24.84, 22.73, 41.69),
         example="LISDAELEAIFEADC", backend="hldh"),
]


def fetch(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read().decode("latin-1").splitlines()


def parse(lines):
    """Return (atom_records, het_records) from the first model; each is a list of dicts."""
    atoms, hets = [], []
    for ln in lines:
        if ln.startswith("ENDMDL"):
            break
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        alt = ln[16]
        if alt not in (" ", "A"):
            continue
        element = ln[76:78].strip() or ln[12:16].strip()[:1]
        if element in ("H", "D"):
            continue
        d = dict(line=ln, chain=ln[21], res=int(ln[22:26]), resname=ln[17:20].strip(),
                 x=float(ln[30:38]), y=float(ln[38:46]), z=float(ln[46:54]))
        (atoms if rec == "ATOM" else hets).append(d)
    return atoms, hets


def centre(rows):
    n = len(rows)
    return [round(sum(r[k] for r in rows) / n, 2) for k in "xyz"]


def min_dist(a_rows, b_rows):
    best = 1e9
    for a in a_rows:
        for b in b_rows:
            d = (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
            best = min(best, d)
    return math.sqrt(best)


def suggested_site(spec, atoms, hets, kept):
    kind = spec["site"][0]
    if kind == "fixed":
        return [spec["site"][1], spec["site"][2], spec["site"][3]]
    if kind == "chain":
        _, chain, lo, hi = spec["site"]
        rows = [a for a in atoms if a["chain"] == chain and lo <= a["res"] <= hi]
    elif kind == "het":
        rows = [h for h in hets if h["resname"] == spec["site"][1]]
    else:
        candidates = spec["site"][1]
        best = min(candidates, key=lambda c: min_dist(
            [a for a in atoms if a["chain"] == c], kept))
        rows = [a for a in atoms if a["chain"] == best]
    if not rows:
        sys.exit(f"{spec['id']}: could not find the site atoms")
    return centre(rows)


def entry(spec, site):
    out = dict(key=spec["key"], pdb=spec["id"], name=spec["name"], about=spec["about"],
               file=f"data/pdb/{spec['id']}.pdb", site=dict(zip("xyz", site)), box=30,
               siteNote=spec["siteNote"], examplePeptide=spec["example"])
    if spec.get("backend"):
        out["backendExample"] = spec["backend"]  # matches an id in the server's /api/examples
    return out


def main():
    out_pdb = ROOT / "data" / "pdb"
    out_pdb.mkdir(parents=True, exist_ok=True)
    entries = []
    for spec in PROTEINS:
        print(f"{spec['id']} …", end=" ", flush=True)
        atoms, hets = parse(fetch(spec["id"]))
        kept = [a for a in atoms if a["chain"] in spec["keep"]]
        site = suggested_site(spec, atoms, hets, kept)
        text = [f"REMARK   Trimmed from RCSB {spec['id']} by tools/prepare_structures.py "
                f"(chains {spec['keep']}, heavy atoms only)."]
        prev = None
        for a in kept:
            if prev is not None and a["chain"] != prev:
                text.append("TER")
            text.append(a["line"].rstrip())
            prev = a["chain"]
        text += ["TER", "END"]
        (out_pdb / f"{spec['id']}.pdb").write_text("\n".join(text) + "\n")
        print(f"{len(kept)} atoms, site {site}")
        entries.append(entry(spec, site))
    for spec in LOCAL:
        print(f"{spec['id']} (local) …", end=" ", flush=True)
        atoms, _ = parse((REPO / spec["file"]).read_text().splitlines())
        text = [f"REMARK   Copied from {spec['file']} by scripts/prepare_web_structures.py (ATOM records, heavy atoms only)."]
        prev = None
        for a in atoms:
            if prev is not None and a["chain"] != prev:
                text.append("TER")
            text.append(a["line"].rstrip())
            prev = a["chain"]
        text += ["TER", "END"]
        (out_pdb / f"{spec['id']}.pdb").write_text("\n".join(text) + "\n")
        print(f"{len(atoms)} atoms")
        entries.append(entry(spec, suggested_site(spec, atoms, [], atoms)))
    (ROOT / "data" / "proteins.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print("wrote data/proteins.json")


if __name__ == "__main__":
    main()
