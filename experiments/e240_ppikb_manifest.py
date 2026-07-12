"""E240 — assemble a PPIKB MULTI-BINDER receptor manifest for 3D-RISM (the new multi-peptide source the
ML actually needs; PDBbind-925 only had 53 multi-binders, mostly done). PPIKB (data/ppikb_clean.jsonl)
has 274 receptors with >=2 distinct peptide binders.

Group by receptor protein_seq. Keep clean affinity types only (Kd/Ki; drop IC50/EC50 assay noise that
would corrupt the per-receptor baseline y_mean). Cap receptor size so RISM can grid the pocket. For each
receptor pick a REPRESENTATIVE pdb whose structure parses and where the peptide chain is locatable
(via e228.receptor_seq) — that goes first so e239 uses it as rep_pdb/pep_ch.

Emits data/e240_ppikb_manifest.json in the e228 schema, ready for:
  python3 experiments/e239_rism_overnight.py --manifest data/e240_ppikb_manifest.json \
      --out data/e240_ppikb_rism.jsonl --no-t100-guard --workers 18

Run: python3 experiments/e240_ppikb_manifest.py [--max-reclen 600] [--aff Kd Ki]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e228_pilot_assemble as e228  # noqa: E402  (receptor_seq finds rec + pep_ch from structure)

SRC = ROOT / "data" / "ppikb_clean.jsonl"
OUT = ROOT / "data" / "e240_ppikb_manifest.json"


def find_rep(args):
    """Try each distinct peptide's pdb until receptor_seq returns a parseable (rec, pep_ch)."""
    rec_seq, peps = args
    for p in peps:
        rec, pep_ch = e228.receptor_seq(p["pdb"], p["seq"])
        if rec is not None and pep_ch is not None:
            return rec_seq, {"pdb": p["pdb"], "seq": p["seq"], "y": p["y"], "L": p["L"],
                             "pep_ch": pep_ch}, rec
    return rec_seq, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-reclen", type=int, default=600)
    ap.add_argument("--aff", nargs="+", default=["Kd", "KD", "Ki"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    aff = set(a.aff)

    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("aff_type") in aff]
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["protein_seq"]].append(r)

    # collapse to distinct peptides per receptor (avg y over repeat measurements), keep multi-binders
    cand = []
    for rec_seq, rs in by_rec.items():
        if len(rec_seq) > a.max_reclen:
            continue
        by_pep = defaultdict(list)
        for r in rs:
            by_pep[r["seq"]].append(r)
        peps = [{"pdb": v[0]["pdb"], "seq": s, "y": float(np.mean([x["y"] for x in v])),
                 "L": v[0]["length"]} for s, v in by_pep.items()]
        if len(peps) >= 2:
            cand.append((rec_seq, peps))
    cand_map = dict(cand)
    print(f"=== E240 PPIKB manifest: {len(by_rec)} receptors ({''.join(aff)}), "
          f"{len(cand)} multi-binder & reclen<={a.max_reclen} ===", flush=True)

    # find a representative parseable pdb+pep_ch per receptor (I/O bound -> parallel)
    out = []
    with Pool(a.workers) as pool:
        for i, (rec_seq, rep, rec) in enumerate(pool.imap_unordered(find_rep, cand), 1):
            if rep is None:
                continue
            # rep peptide first (carries the valid pep_ch e239 needs); the rest follow
            others = [{"pdb": q["pdb"], "seq": q["seq"], "y": q["y"], "L": q["L"], "pep_ch": None}
                      for q in cand_map[rec_seq] if q["seq"] != rep["seq"]]
            peps = [rep] + others
            ys = [p["y"] for p in peps]
            out.append({"receptor_len": len(rec.replace("/", "")), "n_pep": len(peps),
                        "y_mean": float(np.mean(ys)), "y_std": float(np.std(ys)),
                        "y_min": float(min(ys)), "y_max": float(max(ys)),
                        "peptides": peps, "rec_seq": rec})
            if i % 25 == 0:
                print(f"  resolved {i}/{len(cand)}  kept={len(out)}", flush=True)

    out.sort(key=lambda d: (d["n_pep"], d["y_std"]), reverse=True)
    json.dump({"source": "ppikb_clean", "aff_types": sorted(aff), "max_reclen": a.max_reclen,
               "n_receptors": len(out), "n_peptides": sum(d["n_pep"] for d in out),
               "receptors": out}, open(a.out, "w"))
    sp = np.std([d["y_mean"] for d in out]) if out else 0
    print(f"\n  kept {len(out)} receptors, {sum(d['n_pep'] for d in out)} peptides, "
          f"baseline std={sp:.2f}\n  wrote → {a.out}")


if __name__ == "__main__":
    main()
