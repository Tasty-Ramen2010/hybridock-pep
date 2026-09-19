#!/usr/bin/env python
"""Crop protein-protein interfaces into peptide-shaped training examples with measured kcal/mol.

THE MOVE. Our scoring function has always been starved: ~1,500 usable protein-peptide complexes,
and every attempt to fit anything delicate on that has hit a flat learning curve. The survey in
ppi_interface_survey.py showed PPI interfaces are the same KIND of object as peptide interfaces --
median 18 vs 13 residues on the smaller side, 508 vs 568 cross-interface contacts -- so a PPI with
its interface cropped out is, for scoring purposes, a peptide complex that happens to arrive
attached to more protein. SKEMPI carries 7,086 measured affinities over 323 structures.

WHAT IS WRITTEN, AND WHY IN THIS SHAPE. Each complex becomes exactly the file pair our existing
extractor already consumes -- a receptor PDB and a "peptide" PDB -- so the SAME 31 features are
computed on PPI and peptide data with no second code path. If the features had to be
reimplemented for PPI, any transfer result would be confounded by the reimplementation.

  peptide side    the interface residues of the SMALLER partner, contiguous runs kept whole
  receptor side   the larger partner cropped to residues within POCKET_R of that interface,
                  which matches the pocket convention our peptide corpus already uses

NOTHING IS REPACKED OR MINIMISED. Coventry, 9:05: "take the interface section. Again, don't touch
it after that point, because you'll change it in a way that's probably bad." The crop is pure
atom selection.

THE LABEL. dG = RT ln(Kd) at the temperature SKEMPI records, in kcal/mol, from the WILD-TYPE
affinity. Mutants are emitted too when their affinity is measured, because a mutant of a known
complex is a different interface with a different number -- and those pairs are what a contrastive
loss will want later.

LEAKAGE. Holdout is by PDB id, recorded in the manifest, so a later split cannot put two chains of
the same complex on both sides.

Usage: ppi_interface_build.py [--max-small 30] [--limit N]
Output: datasets/ppi_interfaces/<id>/{receptor.pdb,peptide.pdb} + data/ppi_interface_bench.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
SKEMPI = ROOT / "datasets/skempi"
OUT = ROOT / "datasets/ppi_interfaces"
MANIFEST = ROOT / "data/ppi_interface_bench.csv"
CUT = 5.0            # interface definition, matches the survey
POCKET_R = 10.0      # receptor crop radius around the interface, matches our peptide pockets
R_GAS = 0.0019872    # kcal/(mol*K)
AA3 = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
       'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def load(p: Path):
    """[(chain, resseq, resname, [lines], coords)] — standard residues, altloc A only."""
    res, cur = [], None
    for l in p.read_text(errors="ignore").splitlines():
        if not l.startswith("ATOM"):
            continue
        if l[16] not in (" ", "A") or l[17:20].strip() not in AA3:
            continue
        if l[76:78].strip() == "H":
            continue
        k = (l[21], l[22:27])
        if k != cur:
            res.append([l[21], l[22:27], l[17:20].strip(), [], []])
            cur = k
        res[-1][3].append(l)
        res[-1][4].append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return [(c, s, n, ls, np.array(xs)) for c, s, n, ls, xs in res if len(xs)]


def skempi_affinity():
    """{PDB: (chains1, chains2, dG_wt, T)} plus per-mutant rows."""
    wt, muts = {}, defaultdict(list)
    with open(ROOT / "data/skempi_v2.csv") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            pid = r["#Pdb"]
            bits = pid.split("_")
            if len(bits) < 3:
                continue
            pdb, c1, c2 = bits[0].upper(), bits[1], bits[2]
            try:
                T = float((r.get("Temperature") or "298").split("(")[0].strip() or 298)
            except ValueError:
                T = 298.0
            T = T if 250 < T < 350 else 298.0
            for key, store in (("Affinity_wt_parsed", "wt"), ("Affinity_mut_parsed", "mut")):
                v = (r.get(key) or "").strip()
                if not v:
                    continue
                try:
                    kd = float(v)
                except ValueError:
                    continue
                if not (0 < kd < 1):
                    continue
                dg = R_GAS * T * math.log(kd)
                if store == "wt" and pdb not in wt:
                    wt[pdb] = (c1, c2, dg, T)
                elif store == "mut":
                    muts[pdb].append((r.get("Mutation(s)_PDB", ""), dg))
    return wt, muts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-small", type=int, default=30,
                    help="skip interfaces whose smaller side exceeds this many residues")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    wt, muts = skempi_affinity()
    print(f"SKEMPI: {len(wt)} complexes with a parsed wild-type affinity, "
          f"{sum(len(v) for v in muts.values())} mutant measurements\n")
    OUT.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], defaultdict(int)
    files = sorted(SKEMPI.glob("*.pdb"))
    if a.limit:
        files = files[:a.limit]
    for f in files:
        pdb = f.stem.upper()
        if pdb not in wt:
            skipped["no affinity"] += 1
            continue
        c1, c2, dg, T = wt[pdb]
        res = load(f)
        if not res:
            skipped["unparseable"] += 1
            continue
        A = [r for r in res if r[0] in c1]
        B = [r for r in res if r[0] in c2]
        if not A or not B:
            skipped["chains missing"] += 1
            continue

        # which residues touch the other side
        bflat = np.vstack([r[4] for r in B])
        ia, ib_mask = [], np.zeros(len(bflat), dtype=bool)
        off, spans = 0, []
        for r in B:
            spans.append((off, off + len(r[4]))); off += len(r[4])
        for r in A:
            d = np.linalg.norm(r[4][:, None, :] - bflat[None, :, :], axis=2)
            m = d < CUT
            if m.any():
                ia.append(r)
                ib_mask |= m.any(0)
        ib = [r for r, (s, e) in zip(B, spans) if ib_mask[s:e].any()]
        if not ia or not ib:
            skipped["no interface"] += 1
            continue

        # smaller side plays the peptide; the other side is the receptor
        if len(ib) <= len(ia):
            pep_if, rec_all = ib, A
        else:
            pep_if, rec_all = ia, B
        if len(pep_if) > a.max_small:
            skipped[f"smaller side > {a.max_small}"] += 1
            continue

        pep_xyz = np.vstack([r[4] for r in pep_if])
        pocket = [r for r in rec_all
                  if (np.linalg.norm(r[4][:, None, :] - pep_xyz[None, :, :], axis=2).min()
                      < POCKET_R)]
        if len(pocket) < 15:
            skipped["pocket too small"] += 1
            continue

        d = OUT / pdb
        d.mkdir(exist_ok=True)
        (d / "peptide.pdb").write_text(
            "\n".join(l for r in pep_if for l in r[3]) + "\nTER\nEND\n")
        (d / "receptor.pdb").write_text(
            "\n".join(l for r in pocket for l in r[3]) + "\nTER\nEND\n")
        seq = "".join(ONE.get(r[2], "X") for r in pep_if)
        rows.append({"name": pdb, "receptor": str(d / "receptor.pdb"),
                     "peptide_pdb": str(d / "peptide.pdb"), "seq": seq,
                     "pep_len": len(pep_if), "pocket_res": len(pocket),
                     "dG_kcal_mol": round(dg, 3), "temp_K": T,
                     "n_mutants": len(muts.get(pdb, [])), "holdout_id": pdb,
                     "source": "skempi_wt"})

    MANIFEST.parent.mkdir(exist_ok=True)
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)

    print(f"  built {len(rows)} interface examples -> {MANIFEST.relative_to(ROOT)}")
    if rows:
        L = [r["pep_len"] for r in rows]
        G = [r["dG_kcal_mol"] for r in rows]
        print(f"  peptide side: median {np.median(L):.0f} residues, range {min(L)}-{max(L)}")
        print(f"  pocket:       median {np.median([r['pocket_res'] for r in rows]):.0f} residues")
        print(f"  dG:           median {np.median(G):.2f}, range {min(G):.2f} to {max(G):.2f} "
              f"kcal/mol")
        print(f"  mutant measurements available on these complexes: "
              f"{sum(r['n_mutants'] for r in rows)}")
    print("\n  skipped:")
    for k, v in sorted(skipped.items(), key=lambda t: -t[1]):
        print(f"    {v:>5}  {k}")


if __name__ == "__main__":
    main()
