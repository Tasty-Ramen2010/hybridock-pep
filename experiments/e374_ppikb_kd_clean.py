"""E374 — build a CLEAN PPIKB subset: Kd-only, label-agreeing (Ram's request).

PPIKB carries mixed label types (Kd/Ki/IC50/EC50) and cross-source disagreement up to 10.5 kcal/mol for the same
peptide. This emits a defensible clean set:
  1. keep only aff_type in {Kd, KD}                         (drop Ki / IC50 / EC50 assay-specific labels)
  2. drop rows with missing seq/y
  3. for each unique sequence: keep it iff its Kd measurements AGREE — max−min spread <= AGREE_KCAL (singletons
     trivially agree). Sequences whose repeat measurements contradict each other are removed entirely.
  4. within a kept multi-measurement sequence, set y := mean of the agreeing measurements (rows keep their own
     structural features, so multiple structures of the same peptide are retained for structure-based training).

Writes data/ppikb_kd_clean.jsonl. Reports before/after counts and confirms the same-seq label spread collapses.

Run: OMP_NUM_THREADS=1 python experiments/e374_ppikb_kd_clean.py
"""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/ppikb_features.jsonl"
OUT = ROOT / "data/ppikb_kd_clean.jsonl"
AGREE_KCAL = 1.0  # a sequence's repeat Kd measurements must fall within this window to be trusted


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    kd = [r for r in rows if r.get("aff_type") in ("Kd", "KD") and r.get("seq") and r.get("y") is not None]
    print(f"source rows={len(rows)}  ->  Kd-only with seq/y={len(kd)}")

    byseq = defaultdict(list)
    for r in kd:
        byseq[r["seq"]].append(float(r["y"]))

    agree_seqs, dropped_disagree = {}, 0
    for s, ys in byseq.items():
        if max(ys) - min(ys) <= AGREE_KCAL:
            agree_seqs[s] = float(np.mean(ys))      # collapse to the mean of agreeing measurements
        else:
            dropped_disagree += 1

    # merge in the 16 STRUCT geometry features (from e371) keyed by row id, if available
    struct = {}
    sf = ROOT / "data/ppikb_struct_features.jsonl"
    if sf.exists():
        for l in sf.read_text().splitlines():
            if l.strip():
                s = json.loads(l)
                if "poc_n" in s:
                    struct[s.get("id")] = s

    STRUCT_KEYS = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis", "bsa_hyd", "sasa_hb", "sasa_sb",
                   "arom_cc", "hb_count", "strength_bur", "mean_burial", "mj_contact", "rg_per_L",
                   "org_density", "cys_frac"]
    kept = []
    for r in kd:
        if r["seq"] in agree_seqs:
            r = dict(r)
            r["y"] = agree_seqs[r["seq"]]           # use the agreed mean label
            r["clean"] = "kd_agree<=1.0kcal"
            sm = struct.get(r.get("id"))
            if sm:
                for k in STRUCT_KEYS:
                    r[k] = float(sm.get(k, 0.0))
            kept.append(r)

    OUT.write_text("\n".join(json.dumps(r) for r in kept) + "\n")

    # verify the clean set is actually cleaner
    chk = defaultdict(list)
    for r in kept:
        chk[r["seq"]].append(float(r["y"]))
    spreads = [max(v) - min(v) for v in chk.values() if len(v) > 1]
    print(f"\ndropped sequences whose repeat Kd disagreed by >{AGREE_KCAL} kcal: {dropped_disagree}")
    print(f"CLEAN set: rows={len(kept)}  unique seqs={len(chk)}")
    print(f"  max same-seq label spread now = {max(spreads) if spreads else 0.0:.2f} kcal "
          f"(was up to 10.55; collapses because we averaged agreeing repeats)")
    ys = np.array([r["y"] for r in kept])
    print(f"  ΔG range {ys.min():.1f}..{ys.max():.1f}  mean {ys.mean():.2f}  std {ys.std():.2f}")
    print(f"  has 16 STRUCT feats: {sum('poc_n' in r for r in kept)}/{len(kept)}   "
          f"has desc3d: {sum(bool(r.get('desc3d')) for r in kept)}/{len(kept)}")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
