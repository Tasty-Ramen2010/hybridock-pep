#!/usr/bin/env python
"""Build the distillation training set: co-folded designed-groove complexes as denoising targets.

WHAT IS BEING TAUGHT. Our generator places peptides in designed grooves at ~11 A while co-folding
places them at ~3.4 A, and on the 150-complex gap check the two disagree by 5 A or more on 96% of
the corpus. So there is a real signal to transfer, and unlike the four previous retrains -- which
all tried to buy pose accuracy by reweighting the CORPUS COMPOSITION and all failed -- this one
changes the TARGET the model denoises toward.

FOUR THINGS THAT KILLED EARLIER RUNS, AND WHAT IS DONE ABOUT EACH.

1. RECEPTOR-PROVENANCE CONTAMINATION. longgeo's validation set was 97.4% inside its own training
   set, so its val loss was a memorisation score. Worse here: 82 of the 130 highest-ipTM co-folds
   use COVENTRY binders as receptors, and the Coventry grid is our evaluation. Training on them
   would contaminate the only benchmark that measures the thing we are trying to fix. So every
   Coventry receptor is excluded outright, and train/val are split so that no receptor appears in
   both.

2. THE TARGET. Every run before Sep 12 denoised toward a sequence-built idealized strand ~50 A
   from the real pose. `--crystal-target` is mandatory and is emitted in the launcher; here the
   "crystal" is the co-folded complex, which is the entire point.

3. GATE COLLAPSE. MSE on the score rewards shrinking the translation gate whenever the predicted
   direction is imperfect, and it decays monotonically from epoch 1, so no early checkpoint is
   salvageable. `--freeze-gate-layers` is mandatory and is also emitted.

4. BROKEN TOPOLOGY. Bonds come from the sequence template and coordinates from the target file, so
   a chain break leaves a bond spanning the gap and epoch-1 losses explode. Co-folded structures
   should be continuous by construction, but "should be" is how the earlier ones got through, so
   consecutive CA distances are checked and violators dropped.

THE ipTM FILTER IS EARNED, NOT ASSUMED. On the fourteen designed complexes where the paper gives
us truth, ipTM ranks Boltz's correct co-folds above its incorrect ones (Spearman -0.613, p=0.020,
AUC 0.800), so a threshold really does select correct targets. It is not perfect -- at 0.82 about
a quarter of what we keep is still wrong -- and that residual is the honest cost of this corpus.

REPLAY. Real crystal complexes are mixed in so the model does not trade general docking for
designed grooves. Provenance is a column, never blended silently.

Usage: distill_train_build.py [--min-iptm 0.82] [--replay 1500] [--repeat 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("HDP_ROOT", Path(__file__).resolve().parent.parent))
CORPUS = ROOT / "logs/distill_corpus.jsonl"
WORK = ROOT / "datasets/distill_train"
REPLAY_CSV = ROOT / "data/longft_train.csv"
FIELDS = ["complex_name", "protein_description", "peptide_description", "source",
          "pep_len", "ss_class", "length_bucket"]


def split_chains(p: Path) -> tuple[list[str], list[str]] | tuple[None, None]:
    ch: dict[str, list[str]] = {}
    for l in p.read_text().splitlines():
        if l.startswith("ATOM"):
            ch.setdefault(l[21], []).append(l)
    if len(ch) < 2:
        return None, None

    def nres(ls):
        return len({x[22:27] for x in ls})
    ks = sorted(ch, key=lambda c: -nres(ch[c]))
    return ch[ks[0]], ch[ks[1]]


def ca_of(lines: list[str]) -> np.ndarray:
    return np.array([(float(x[30:38]), float(x[38:46]), float(x[46:54]))
                     for x in lines if x[12:16].strip() == "CA"])


def bucket(n: int) -> str:
    return "short" if n <= 8 else "med" if n <= 12 else "long" if n <= 19 else "vlong"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-iptm", type=float, default=0.82)
    ap.add_argument("--replay", type=int, default=1500, help="real crystal rows mixed in")
    ap.add_argument("--repeat", type=int, default=8, help="times the distil set is repeated")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", default="distill")
    a = ap.parse_args()

    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    usable = [r for r in rows if r.get("usable") and r.get("pdb") and Path(r["pdb"]).exists()]
    clean = [r for r in usable if not r["name"].startswith("coventry_")]
    sel = [r for r in clean if r["iptm"] >= a.min_iptm]
    print(f"{len(rows)} corpus rows -> {len(usable)} usable -> {len(clean)} non-Coventry "
          f"-> {len(sel)} at ipTM >= {a.min_iptm}")
    print(f"  ({len(usable) - len(clean)} Coventry-receptor rows excluded to keep the grid "
          f"evaluation clean)")

    WORK.mkdir(parents=True, exist_ok=True)
    built, drop = [], Counter()
    for r in sel:
        rec, pep = split_chains(Path(r["pdb"]))
        if rec is None:
            drop["<2 chains"] += 1
            continue
        ca = ca_of(pep)
        if len(ca) < 4:
            drop["peptide too short"] += 1
            continue
        d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
        if d.max() > 4.3 or d.min() < 3.2:
            # a chain break would become a bond spanning the gap once the sequence template
            # supplies the topology -- the failure that produced epoch-1 losses of 10,086
            drop[f"broken backbone ({d.max():.1f} A)"] += 1
            continue
        out = WORK / r["name"]
        out.mkdir(parents=True, exist_ok=True)
        rp, pp = out / "receptor.pdb", out / "peptide.pdb"
        rp.write_text("\n".join(rec) + "\nTER\nEND\n")
        pp.write_text("\n".join(pep) + "\nTER\nEND\n")
        built.append({"complex_name": r["name"], "protein_description": str(rp),
                      "peptide_description": str(pp), "source": "cofold_distill",
                      "pep_len": r["pep_len"], "ss_class": "UNK",
                      "length_bucket": bucket(r["pep_len"]),
                      "_rec": r["name"].split("__")[0]})
    print(f"\n  {len(built)} complexes staged")
    for k, v in sorted(drop.items(), key=lambda t: -t[1]):
        print(f"    dropped {v:>3}  {k}")
    if not built:
        print("nothing to train on"); return

    # SPLIT BY RECEPTOR, not by row. Two peptides against the same binder share the groove we are
    # trying to learn, so splitting rows at random would put the same receptor on both sides.
    recs = sorted({b["_rec"] for b in built})
    random.Random(0).shuffle(recs)
    nval = max(2, int(round(a.val_frac * len(recs))))
    val_recs = set(recs[:nval])
    tr = [b for b in built if b["_rec"] not in val_recs]
    va = [b for b in built if b["_rec"] in val_recs]
    print(f"\n  receptor-disjoint split: {len(recs) - nval} receptors train / {nval} val")
    print(f"    {len(tr)} train complexes, {len(va)} val complexes")
    assert not ({b['_rec'] for b in tr} & {b['_rec'] for b in va}), "receptor leak"

    replay = []
    if REPLAY_CSV.exists() and a.replay:
        pool = list(csv.DictReader(REPLAY_CSV.open()))
        random.Random(1).shuffle(pool)
        replay = pool[:a.replay]
        print(f"  + {len(replay)} real-crystal replay rows from {REPLAY_CSV.name}")

    train_rows = []
    for _ in range(a.repeat):
        train_rows += [{k: v for k, v in b.items() if k != "_rec"} for b in tr]
    train_rows += [{f: r.get(f, "") for f in FIELDS} for r in replay]
    random.Random(2).shuffle(train_rows)

    tp = ROOT / f"data/{a.tag}_train.csv"
    vp = ROOT / f"data/{a.tag}_val.csv"
    for path, rws in ((tp, train_rows), (vp, [{k: v for k, v in b.items() if k != "_rec"}
                                              for b in va])):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rws)

    frac = 100.0 * (len(tr) * a.repeat) / max(len(train_rows), 1)
    print(f"\n  -> {tp.relative_to(ROOT)}  {len(train_rows)} rows/epoch "
          f"({frac:.0f}% co-folded, {100 - frac:.0f}% real crystal)")
    print(f"  -> {vp.relative_to(ROOT)}  {len(va)} rows")
    print("\n  NOTE: val loss here is a sanity signal ONLY. RAPiDock selects on "
          "valinf_rmsds_backbone_lt2,\n  and every diffusion checkpoint this project has picked "
          "on val loss has been worse at docking.\n  Select on direct-RMSD docking against a "
          "held-out bench.")


if __name__ == "__main__":
    main()
