#!/usr/bin/env python
"""Generate co-folded poses as training targets — designed-groove data we cannot otherwise get.

WHY THIS IS NOT RETRAIN NUMBER FIVE. Four finetunes have failed: solenoid (34/60, p=0.257),
designed-data (p=0.167), longgeo (an active regression, Coventry AUC 0.718 -> 0.531), and the
gate-collapse run before them. Every one of them tried to buy pose accuracy by RE-WEIGHTING the
corpus we already had -- more solenoids, more grooves, more long peptides -- and the corpus is
the thing that is wrong, so re-weighting it could not have worked.

This changes the corpus instead of its weights. The measured failure is specific: our poses reach
median 1.83 A on natural complexes and 10.88 A on designed repeat grooves, 0/14 under 5 A. The
cause Coventry gave (24:48) is that native peptide-protein pairs co-evolved and leave a strong
geometric signal, while de novo backbones are "purely geometrically helical" and carry none, so
placement has to come from fine chemical detail instead. RAPiDock saw ~4,000 complexes; Boltz saw
~10 million. That is a pretraining gap, not a weighting problem.

We could only ever find ~50 real designed peptide-binder complexes, which is why the designed-data
finetune was underpowered. Co-folding lifts that ceiling: it produces designed-groove complexes on
demand, at 38 s each, and on the one set where we can check it against truth it lands at median
3.42 A with 9/14 under 5 A -- far from perfect, but three times closer than we get, and available
in thousands rather than dozens.

WHAT IS AND IS NOT CLAIMED. These are not crystal structures and must never be mixed with them as
if they were: on natural complexes the crystal is better and co-folding would be a downgrade. They
are for the designed-groove class ONLY, where the alternative is no data at all. Provenance is
recorded per row so a training run cannot silently blend the two.

One diffusion sample per pair -- we want one target per complex, not an ensemble, and n=1 at 38 s
is the cheap end of the curve.

Usage: distill_corpus_build.py [--n 600] [--out datasets/distill]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
BOLTZ = Path.home() / "miniconda3/envs/boltz-env/bin/boltz"
LOG = ROOT / "logs/distill_corpus.jsonl"
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def seq_of(pdb: Path) -> str:
    out, seen = [], set()
    for l in pdb.read_text(errors="ignore").splitlines():
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            k = (l[21], l[22:27])
            if k in seen:
                continue
            seen.add(k)
            out.append(AA3.get(l[17:20].strip(), "X"))
    return "".join(out)


def designed_receptors() -> list[tuple[str, str]]:
    """(name, sequence) for every de novo binder we have -- the geometry class we fail on."""
    out = []
    for d, tag in ((ROOT / "datasets/coventry/binders_paper", "coventry"),
                   (ROOT / "datasets/cao2022/folded", "cao")):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdb")):
            if f.name.endswith("_peptide.pdb"):
                continue
            s = seq_of(f)
            if 40 <= len(s) <= 400 and "X" not in s[:40]:
                out.append((f"{tag}_{f.stem}", s))
    return out


def peptides(limit: int) -> list[tuple[str, str]]:
    """In-scope peptides from our own corpus: real sequences, 8-25 aa."""
    out, seen = [], set()
    for csvf in ("data/longft_train_scope25_authorsP.csv", "data/bench_recentset_heldout.csv",
                 "data/longft_train.csv"):
        p = ROOT / csvf
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            s = (r.get("seq") or "").strip().upper()
            if not s:
                pp = r.get("peptide_description") or ""
                if pp and Path(pp).exists():
                    s = seq_of(Path(pp))
            if 8 <= len(s) <= 25 and s.isalpha() and s not in seen:
                seen.add(s)
                out.append((r.get("name") or r.get("complex_name") or f"pep{len(out)}", s))
            if len(out) >= limit:
                return out
    return out


def cofold(name: str, rec_seq: str, pep_seq: str, dest: Path, timeout_s: int = 900) -> dict:
    from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb
    work = Path(tempfile.mkdtemp(prefix="dist_", dir="/tmp/claude-1000"))
    try:
        spec = work / "s.yaml"
        spec.write_text("version: 1\nsequences:\n"
                        f"  - protein:\n      id: A\n      sequence: {rec_seq}\n      msa: empty\n"
                        f"  - protein:\n      id: B\n      sequence: {pep_seq}\n      msa: empty\n")
        t0 = time.time()
        pr = subprocess.run(
            [str(BOLTZ), "predict", str(spec), "--out_dir", str(work),
             "--recycling_steps", "3", "--diffusion_samples", "1",
             "--output_format", "mmcif", "--override"],
            capture_output=True, text=True, timeout=timeout_s, check=False)
        if pr.returncode != 0:
            return {"error": f"boltz exit {pr.returncode}"}
        cifs = sorted(work.rglob("*_model_0.cif"))
        if not cifs:
            return {"error": "no structure"}
        rec: dict = {"seconds": round(time.time() - t0, 1)}
        for j in cifs[0].parent.glob("confidence_*_model_0.json"):
            try:
                c = json.loads(j.read_text())
                rec["iptm"] = c.get("iptm"); rec["plddt"] = c.get("complex_plddt")
            except (OSError, ValueError):
                pass
            break
        dest.parent.mkdir(parents=True, exist_ok=True)
        _cif_to_pdb(cifs[0], dest)
        rec["pdb"] = str(dest)
        return rec
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="datasets/distill")
    ap.add_argument("--min-iptm", type=float, default=0.55,
                    help="rows below this are kept on disk but flagged unusable")
    ap.add_argument("--exclude-receptor-prefix", default="",
                    help="comma-separated receptor name prefixes to skip. Use this to keep an "
                         "evaluation set's receptors OUT of the training corpus: 82 of the first "
                         "130 high-ipTM complexes used Coventry binders, and training on those "
                         "while evaluating on the Coventry grid is the same receptor-provenance "
                         "contamination that made longgeo's validation meaningless.")
    a = ap.parse_args()

    OUT = ROOT / a.out
    recs, peps = designed_receptors(), peptides(4000)
    if a.exclude_receptor_prefix:
        bad = tuple(x.strip() for x in a.exclude_receptor_prefix.split(",") if x.strip())
        before = len(recs)
        recs = [r for r in recs if not r[0].startswith(bad)]
        print(f"excluded {before - len(recs)} receptors matching {bad}", flush=True)
    print(f"{len(recs)} designed receptors x {len(peps)} in-scope peptides available", flush=True)
    if not recs or not peps:
        print("nothing to pair"); return

    done = set()
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["name"])

    rng = random.Random(0)
    pairs, seen = [], set()
    while len(pairs) < a.n * 2 and len(seen) < len(recs) * len(peps):
        r = rng.choice(recs); p = rng.choice(peps)
        k = (r[0], p[1])
        if k in seen:
            continue
        seen.add(k)
        nm = f"{r[0]}__{p[0]}"[:120]
        if nm in done:
            continue
        pairs.append((nm, r[1], p[1]))
        if len(pairs) >= a.n:
            break

    print(f"{len(pairs)} pairs to co-fold (~{38 * len(pairs) / 3600:.1f} h at 38 s each)",
          flush=True)
    t0 = time.time()
    good = 0
    with LOG.open("a") as fh:
        for k, (nm, rs, ps) in enumerate(pairs, 1):
            row = {"name": nm, "rec_len": len(rs), "pep_seq": ps, "pep_len": len(ps),
                   "provenance": "cofolded_synthetic"}
            row.update(cofold(nm, rs, ps, OUT / nm / "complex.pdb"))
            row["usable"] = bool(row.get("iptm") and row["iptm"] >= a.min_iptm)
            good += row["usable"]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 10 == 0 or k == 1:
                el = time.time() - t0
                print(f"  [{k}/{len(pairs)}] {good} usable  ipTM "
                      f"{row.get('iptm', float('nan')):.2f}  "
                      f"eta {(len(pairs) - k) * el / k / 3600:.1f} h", flush=True)
    print(f"DISTILL_CORPUS_DONE  {good}/{len(pairs)} usable at ipTM >= {a.min_iptm}")


if __name__ == "__main__":
    main()
