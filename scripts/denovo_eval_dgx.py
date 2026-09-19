#!/usr/bin/env python
"""Did the designed-complex finetune actually help? Run on the DGX, where both live.

Compares two checkpoints on complexes NEITHER has been trained on:
  epoch10   longft_tanh/rapidock_finetuned_epoch010.pt   -- our current shipped model
  denovo    longft_denovo/<final>                        -- continuation of epoch10 on a mix
                                                            enriched to 12.1% designed

Test set: the 15 held-out designed PDB ids (20 complexes) from data/denovo_pep_test.csv,
which were excluded from the denovo run's training data by PDB id, not by complex, so no
structure leaks across the split. epoch10 never saw any of them either, so the comparison
is like-for-like.

Metric: best-of-N direct CA RMSD against the crystal peptide, no superposition -- the same
metric as the RecentSet benchmark, so numbers are comparable to the 1.83 A we report there.

Usage (on DGX): denovo_eval_dgx.py [N] [ckpt_denovo]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

R = Path("/home/ox_dgx_atl_26/Ram_Work/hybridock-pep")
PY = str(Path.home() / "miniforge3/envs/rapidock/bin/python")
INFER = R / "third_party/RAPiDock_finetuned"   # fork tree: has the cross_type_embedding
                                             # layer our checkpoints were trained with;
                                             # the upstream tree on DGX lacks the port
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
ARMS = {
    "epoch10": (R / "third_party/RAPiDock_finetuned/longft_tanh",
                "rapidock_finetuned_epoch010.pt"),
    "denovo": (R / "third_party/RAPiDock_finetuned/longft_denovo",
               sys.argv[2] if len(sys.argv) > 2 else None),
}


def ca(p):
    out = []
    for l in open(p, errors="replace"):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            out.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return out


def rmsd(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return (sum(sum((x - y) ** 2 for x, y in zip(a[i], b[i])) for i in range(n)) / n) ** 0.5


def main() -> None:
    d = ARMS["denovo"][0]
    if ARMS["denovo"][1] is None:
        cks = sorted(glob.glob(str(d / "*.pt")))
        if not cks:
            raise SystemExit(f"no checkpoint in {d}")
        ARMS["denovo"] = (d, Path(sorted(cks, key=os.path.getmtime)[-1]).name)
    print(f"denovo checkpoint: {ARMS['denovo'][1]}", flush=True)

    rows = list(csv.DictReader(open(R / "data/denovo_pep_test.csv")))
    print(f"held-out designed complexes: {len(rows)} "
          f"({len({r['pdb'] for r in rows})} pdb ids), N={N}", flush=True)

    out_root = R / "runs/denovo_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    res = {}
    for arm, (mdir, ckpt) in ARMS.items():
        vals = []
        for r in rows:
            name = f"{arm}__{r['complex_name']}"
            rec = r["protein_description"].replace("/home/igem/unknown_software", str(R))
            pep = r["peptide_description"].replace("/home/igem/unknown_software", str(R))
            if not (Path(rec).exists() and Path(pep).exists()):
                print(f"  missing files for {r['complex_name']}", flush=True)
                continue
            seq = "".join(c for c in (r.get("seq") or "") if c.isalpha())
            if not seq:
                continue
            if not glob.glob(str(out_root / name / "rank*.pdb")):
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
                    w = csv.writer(tf)
                    w.writerow(["complex_name", "protein_description", "peptide_description"])
                    w.writerow([name, rec, seq])
                    cp = tf.name
                proc = subprocess.run([PY, str(INFER / "inference.py"),
                                "--protein_peptide_csv", cp, "--out", str(out_root),
                                "--output_dir", str(out_root), "--model_dir", str(mdir),
                                "--ckpt", ckpt, "--N", str(N), "--batch_size", "8",
                                "--inference_steps", "16", "--actual_steps", "16",
                                "--no_final_step_noise", "--conformation_partial", "1:1:1",
                                "--scoring_function", "none", "--confidence_ckpt", "null",
                                "--cpu", "4"], cwd=str(INFER),
                               capture_output=True, text=True,
                               # GB10 is compute capability 12.1; torch 2.7+cu128's nvrtc
                               # rejects -arch=sm_121, so every JIT fusion group fails and
                               # inference dies. Same guard the training script uses.
                               env=dict(os.environ, RAPIDOCK_DISABLE_JIT_FUSER="1",
                                        OMP_NUM_THREADS="1"))
                os.unlink(cp)
                if proc.returncode != 0 and not glob.glob(str(out_root / name / "rank*.pdb")):
                    print(f"  {name}: inference failed rc={proc.returncode}\n"
                          f"    {proc.stderr[-400:]}", flush=True)
            ref = ca(pep)
            best = min((rmsd(ca(p), ref) for p in
                        glob.glob(str(out_root / name / "rank*.pdb"))), default=float("nan"))
            if best == best:
                vals.append((r["complex_name"], best))
                print(f"  {arm:>8} {r['complex_name']:<28} {best:6.2f} A", flush=True)
        res[arm] = vals

    print(f"\n{'arm':<10}{'n':>4}{'median':>9}{'mean':>8}{'<=2A':>7}{'<=5A':>7}")
    import statistics as st
    for arm, vals in res.items():
        v = [x for _, x in vals]
        if v:
            print(f"{arm:<10}{len(v):>4}{st.median(v):>9.2f}{st.mean(v):>8.2f}"
                  f"{sum(1 for x in v if x <= 2):>4}/{len(v):<2}"
                  f"{sum(1 for x in v if x <= 5):>4}/{len(v):<2}")
    a = dict(res.get("epoch10", []))
    b = dict(res.get("denovo", []))
    both = sorted(set(a) & set(b))
    if both:
        better = sum(1 for k in both if b[k] < a[k] - 0.1)
        worse = sum(1 for k in both if b[k] > a[k] + 0.1)
        print(f"\npaired on {len(both)} complexes: denovo better {better}, "
              f"worse {worse}, tied {len(both) - better - worse}")
        print(f"median delta (denovo - epoch10): "
              f"{st.median([b[k] - a[k] for k in both]):+.2f} A")
    (R / "logs/denovo_eval.json").write_text(json.dumps(res, indent=2))
    print("DENOVO_EVAL_DONE")


if __name__ == "__main__":
    main()
