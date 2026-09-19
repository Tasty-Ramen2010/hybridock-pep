#!/usr/bin/env python
"""Re-co-fold the cells that came out bad, with 5 diffusion samples instead of 1.

WHICH CELLS ARE "BAD", AND WHY EACH IS WORTH A RETRY.

  fold-gate reject  the co-folded binder disagrees with our model by more than 3 A, so the
                    transplant was refused. With 1 sample we cannot tell an unlucky diffusion
                    trajectory from a binder Boltz genuinely folds differently -- 5 samples
                    can, because if all 5 agree with each other and disagree with our model,
                    the disagreement is real and OUR model is the outlier (as already proven
                    for n3/n7/pc21/pc26, where AF3 sided with Boltz).
  positive iface    the transplanted peptide clashes into the receptor after repacking. Either
                    the pose is wrong or the transplant landed it badly; more samples give the
                    scorer something better to pick from.
  error             no structure produced at all.

Takes the BEST of the 5 by interface energy after refinement, not by Boltz's own confidence --
ipTM is measurably not tracking the truth on this grid (cognate n1 scores 0.787 while three
non-binders score higher), so letting it choose would be letting a bad ranker pick.

Waits for the main grid to finish before starting, because both want the same GPU and two
concurrent jobs on the 12 GB card is how the Sep-12 hypervisor crash happened.

Usage: coventry_boltz_retry5.py [--samples 5] [--no-wait]
Output: logs/coventry_boltz_retry5.jsonl (same schema as the main grid)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MAIN = ROOT / "logs/coventry_boltz.jsonl"
REFINE = ROOT / "logs/coventry_boltz_refine.jsonl"
GRID_LOG = ROOT / "logs/coventry_boltz_grid.log"
OUT = ROOT / "logs/coventry_boltz_retry5.jsonl"


def classify() -> list[tuple[str, str]]:
    """Cells worth retrying, as (peptide, binder), with the reason printed."""
    scored = {}
    if REFINE.exists():
        for line in REFINE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                scored[r["name"]] = r
    bad, reasons = [], {}
    for line in MAIN.read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        name = m["name"]
        s = scored.get(name, {})
        why = None
        if "error" in m:
            why = "cofold error"
        elif m.get("transplant_error"):
            why = "transplant error"
        elif m.get("accepted") is False:
            why = f"fold gate {m.get('fold_rmsd', float('nan')):.1f} A"
        elif s.get("best_iface") is None:
            why = "not scored"
        elif s["best_iface"] > 0:
            why = f"positive iface {s['best_iface']:.0f}"
        if why:
            bad.append((m["peptide"], m["binder"]))
            reasons[name] = why
    for n, w in sorted(reasons.items()):
        print(f"  retry {n:18s} {w}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    if not args.no_wait:
        print("waiting for the main 324-cell grid to finish...", flush=True)
        while True:
            if GRID_LOG.exists() and "COVENTRY_BOLTZ_DONE" in GRID_LOG.read_text()[-4000:]:
                break
            time.sleep(60)
        print("main grid done; starting retries", flush=True)

    import coventry_boltz_grid as G

    peps, binders = G.load_grid()
    bad = classify()
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["name"])
    jobs = [(p, b) for p, b in bad if f"{p}__{b}" not in done]
    print(f"\n{len(jobs)} cells to retry at {args.samples} diffusion samples "
          f"({len(done)} already retried)", flush=True)

    # route retries to their own pose directory so the n=1 result stays on disk for comparison
    G.POSES = ROOT / "runs/coventry/boltz_n5"
    G.POSES.mkdir(parents=True, exist_ok=True)

    with OUT.open("a") as fh:
        for k, (p, b) in enumerate(jobs, 1):
            name = f"{p}__{b}"
            rec = {"name": name, "peptide": p, "binder": b, "cognate": int(p == b),
                   "peptide_seq": peps[p], "n_samples": args.samples}
            rec.update(_cofold_n(G, p, b, peps[p], binders[b], args.samples))
            if "pdb" in rec:
                rec.update(G.transplant_cell(Path(rec["pdb"]), b, name))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{k}/{len(jobs)}] {name:16s} "
                  f"fold {rec.get('fold_rmsd', float('nan')):.2f} A "
                  f"iptm {rec.get('iptm', float('nan')):.3f}"
                  f"{'  ERROR ' + rec['error'][:60] if 'error' in rec else ''}", flush=True)
    print("COVENTRY_BOLTZ_RETRY5_DONE")


def _cofold_n(G, pep: str, binder: str, pep_seq: str, bind_seq: str, n: int) -> dict:
    """Same as the grid's cofold_cell but asking Boltz for n diffusion samples."""
    import shutil
    import tempfile

    from hybridock_pep.sampling.cofold_boltz import _cif_to_pdb

    name = f"{pep}__{binder}"
    dest = G.POSES / name
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"bz5_{name}_", dir="/tmp/claude-1000"))
    try:
        spec = work / f"{name}.yaml"
        spec.write_text(
            "version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {bind_seq}\n      msa: empty\n"
            f"  - protein:\n      id: B\n      sequence: {pep_seq}\n      msa: empty\n"
        )
        t0 = time.time()
        proc = subprocess.run(
            [str(G.BOLTZ), "predict", str(spec), "--out_dir", str(work),
             "--recycling_steps", "3", "--diffusion_samples", str(n),
             "--output_format", "mmcif", "--override"],
            capture_output=True, text=True, timeout=5400, check=False,
        )
        if proc.returncode != 0:
            return {"error": f"boltz exit {proc.returncode}: {proc.stderr[-400:]}"}
        cifs = sorted(work.rglob("*_model_*.cif"))
        if not cifs:
            return {"error": "no structure produced"}
        # keep every sample; the refinement pass picks the winner on interface energy
        out: dict = {"seconds": round(time.time() - t0, 1), "n_models": len(cifs)}
        for i, c in enumerate(cifs):
            _cif_to_pdb(c, dest / f"cofolded_{i}.pdb")
        out["pdb"] = str(dest / "cofolded_0.pdb")
        out["all_models"] = [str(dest / f"cofolded_{i}.pdb") for i in range(len(cifs))]
        for j in cifs[0].parent.glob("confidence_*_model_0.json"):
            try:
                c = json.loads(j.read_text())
                out.update({k: c[k] for k in ("iptm", "ptm", "complex_plddt") if k in c})
            except (OSError, ValueError):
                pass
            break
        return out
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
