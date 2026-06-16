"""E244 — diverse per-complex manifest for POSE-AWARE GIST. Unlike the receptor-baseline set (one apo
value per receptor, proven dead), this targets per-complex affinity: each entry is ONE peptide-protein
complex with its CRYSTAL bound pose, so GIST can score the water that THIS peptide displaces.

Stratify the 925 PDBbind peptide complexes across Kd (8 bins) x length (short/med/long) to maximize
diversity. Resolve the peptide chain id (e228.receptor_seq). Prefer complexes that already have a
tleap-validated apo from the RISM runs (runs/e230_rism/{pdb}/apo_amber.pdb) so GIST MD won't fail.

Run: python3 scripts/e244_diverse_manifest.py [--n 80]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e228_pilot_assemble as e228  # receptor_seq -> (rec, pep_ch)

SRC = ROOT / "data" / "pdbbind_peptides.jsonl"
OUT = ROOT / "data" / "e244_diverse_manifest.json"
RISM_DIR = ROOT / "runs" / "e230_rism"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    for r in rows:
        r["has_apo"] = (RISM_DIR / r["pdb"] / "apo_amber.pdb").exists()

    # stratify: 8 Kd bins x 3 length bins; within each cell prefer has_apo, then spread
    y = np.array([r["y"] for r in rows])
    kb = np.digitize(y, np.quantile(y, np.linspace(0, 1, 9)[1:-1]))
    lb = np.digitize([r["length"] for r in rows], [8, 13])  # short<=8 / med / long>=13
    cells = {}
    for r, k, l in zip(rows, kb, lb):
        cells.setdefault((int(k), int(l)), []).append(r)
    for c in cells.values():
        c.sort(key=lambda r: (not r["has_apo"], r["pdb"]))  # has_apo first, deterministic

    # round-robin across cells until we hit n
    picked, order = [], sorted(cells)
    idx = {c: 0 for c in order}
    while len(picked) < a.n and any(idx[c] < len(cells[c]) for c in order):
        for c in order:
            if idx[c] < len(cells[c]) and len(picked) < a.n:
                picked.append(cells[c][idx[c]]); idx[c] += 1

    # resolve pep_ch (needs structure parse); drop unparseable
    out = []
    for r in picked:
        rec, pep_ch = e228.receptor_seq(r["pdb"], r["seq"])
        if rec is None or pep_ch is None:
            continue
        out.append({"pdb": r["pdb"], "seq": r["seq"], "y": float(r["y"]), "L": r["length"],
                    "pep_ch": pep_ch, "rec_seq": rec, "receptor_len": len(rec.replace("/", "")),
                    "has_apo": r["has_apo"]})
    out.sort(key=lambda r: r["receptor_len"])  # cheapest MD first
    json.dump({"task": "per-complex affinity (pose-aware GIST)", "n": len(out),
               "n_with_apo": sum(r["has_apo"] for r in out), "complexes": out}, open(a.out, "w"))
    yy = [r["y"] for r in out]
    print(f"=== E244 diverse manifest: {len(out)} complexes ({sum(r['has_apo'] for r in out)} with cached apo) ===")
    print(f"  Kd range {min(yy):.1f}..{max(yy):.1f} (std {np.std(yy):.2f}), "
          f"len {min(r['L'] for r in out)}-{max(r['L'] for r in out)}, "
          f"reclen {out[0]['receptor_len']}-{out[-1]['receptor_len']}")
    print(f"  wrote -> {a.out}")


if __name__ == "__main__":
    main()
