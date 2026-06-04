#!/usr/bin/env python3
"""
extract_encoder_gen_n100.py — 96-dim encoder features for the n=100 gen subset.

Thin wrapper around confidence_diagnosis.{load_encoder, load_or_extract} so the
new gen poses get the SAME pretrained-encoder features (frozen BN) as bench300.
Runs in the rapidock env (GPU). Caches/resumes to feats_gen_n100.pkl.

  PYTHONPATH=$(pwd) ~/miniconda3/envs/rapidock/bin/python \
      scripts/extract_encoder_gen_n100.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.confidence_diagnosis as cd   # sets RAPiDock sys.path at import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=str(cd.REPO / "logs" / "gen_n100" / "benchmark_results.json"))
    ap.add_argument("--csv",  default=str(cd.REPO / "data" / "gen_subset_n100.csv"))
    ap.add_argument("--out",  default=str(cd.REPO / "logs" / "diagnosis" / "feats_gen_n100.pkl"))
    ap.add_argument("--tmp",  default=str(cd.REPO / "logs" / "gen_n100" / "_encoder_tmp"))
    a = ap.parse_args()

    jd = json.load(open(a.json))
    print(f"gen_n100 JSON: {len(jd)} complexes")
    enc = cd.load_encoder(a.device)
    feats = cd.load_or_extract(enc, jd, Path(a.csv), a.tmp, a.device, Path(a.out), "gen_n100")
    n_cx = len({k[0] for k in feats})
    print(f"encoder features: {len(feats)} poses / {n_cx} complexes → {a.out}")


if __name__ == "__main__":
    main()
