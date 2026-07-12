"""E49 — the decisive test: does ENSEMBLE-averaged MM-GBSA break the charged floor that
SINGLE-POSE MM-GBSA could not? (Ram's 'ensemble is all we need', done with the OpenMM we have.)

Every prior charged-floor test used SINGLE-POSE MM-GBSA (r≈0.07 on the charged half — even with real
GB). We have never tested the ENSEMBLE average. sample_interaction_energies() already IS the
wiggle-relax engine: Langevin MD under ff14SB+GBn2 (physics-constrained wiggling, 300 K Boltzmann),
recording per-frame interaction energy E_int = E_complex − E_rec − E_pep (GB solvent included). The
trajectory mean ⟨E_int⟩ is the LIE/ensemble-MM-GBSA quantity; interaction_entropy gives −TΔS_IE.

For a charge-balanced subset of crystal-65 (pose_0 poses), compute and compare vs experimental ΔG,
SPLIT BY CHARGE:
  dg_single   : compute_mmgbsa_single (1-traj)             — the old single-pose number
  e_int_mean  : ⟨E_int⟩ over the MD ensemble               — the ensemble average (the new thing)
  e_int_ent   : ⟨E_int⟩ − TΔS_IE                            — ensemble + entropy correction
CPU-only (force_cpu=True) — never contends with the GPU. Cached per complex; resumable.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.interaction_entropy import (interaction_entropy,  # noqa: E402
                                                       sample_interaction_energies)
from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single  # noqa: E402

N_FRAMES = 50          # 50 frames x 300 steps = 30 ps — short; IE/⟨E_int⟩ converge fast
STEPS = 300
USE_GPU = True         # GPU idle (PfLDH dock done); ~40s/complex vs ~5min CPU
N_PER_BIN = 0          # 0 = use ALL complexes (decisive charge split); >0 = pilot subset/bin


def charged_frac(seq):
    return sum(c in "DEKR" for c in seq) / max(1, len(seq))


def select():
    bench = [r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())]
    items = []
    for m in bench:
        pdb = m["pdb"].upper()
        pose = ROOT / f"logs/crystal65_n100/cr_{pdb}/poses/pose_0.pdb"
        rec = ROOT / m["pocket_pdb"]
        if pose.exists() and rec.exists() and m.get("peptide_seq"):
            items.append((pdb, pose, rec, m["dg_exp"], charged_frac(m["peptide_seq"])))
    items.sort(key=lambda t: t[4], reverse=True)   # HIGH-charge first (the decisive floor test)
    if N_PER_BIN <= 0:
        return items                        # all complexes — decisive charge split
    return items[:N_PER_BIN] + items[-N_PER_BIN:]


def main():
    cache = Path("/tmp/e49_ens_mmgbsa.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    todo = select()
    print(f"=== E49 ensemble MM-GBSA on {len(todo)} complexes (CPU, {N_FRAMES}f x {STEPS}st) ===",
          flush=True)
    for pdb, pose, rec, y, cf in todo:
        if pdb in out:
            continue
        try:
            dg_single = compute_mmgbsa_single(pose.resolve(), rec.resolve(), force_cpu=not USE_GPU)
            eint = sample_interaction_energies(pose.resolve(), rec.resolve(), n_frames=N_FRAMES,
                                               steps_between_frames=STEPS, force_cpu=not USE_GPU)
            mts = interaction_entropy(eint)
            out[pdb] = dict(y=y, cf=cf, dg_single=float(dg_single),
                            e_int_mean=float(eint.mean()), e_int_std=float(eint.std()),
                            minus_tds=float(mts))
            cache.write_text(json.dumps(out))
            print(f"  {pdb} cf={cf:.2f} y={y:+.1f} | single={dg_single:+.1f} "
                  f"⟨Eint⟩={eint.mean():+.1f} -TΔS={mts:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
    evaluate(out)


def evaluate(out):
    from scipy.stats import pearsonr
    ks = list(out)
    if len(ks) < 6:
        print(f"\n(only {len(ks)} done — rerun to finish before evaluating)")
        return
    y = np.array([out[k]["y"] for k in ks]); cf = np.array([out[k]["cf"] for k in ks])
    hi = cf >= np.median(cf)
    print(f"\n=== single-pose vs ENSEMBLE MM-GBSA vs experimental ΔG (n={len(ks)}) ===")
    print(f"  {'predictor':<14}{'all':>9}{'low-charge':>12}{'high-charge':>13}")
    feats = {"dg_single": "single pose", "e_int_mean": "⟨E_int⟩ ensemble",
             "e_int_ent": "⟨E_int⟩−TΔS"}
    for f, lbl in feats.items():
        v = np.array([(out[k]["e_int_mean"] + out[k]["minus_tds"]) if f == "e_int_ent"
                      else out[k][f] for k in ks])
        ra = pearsonr(v, y).statistic if v.std() > 0 else 0
        rl = pearsonr(v[~hi], y[~hi]).statistic if v[~hi].std() > 0 else 0
        rh = pearsonr(v[hi], y[hi]).statistic if v[hi].std() > 0 else 0
        print(f"  {lbl:<14}{ra:>+9.3f}{rl:>+12.3f}{rh:>+13.3f}")
    print("  >> does ⟨E_int⟩ (ensemble) beat single-pose on the HIGH-CHARGE column (the floor)?")
    print("  >> if YES: ensemble IS the lever, wire ensemble-MM-GBSA into --refine-topk.")
    print("  >> if NO: the floor is force-field-deep (polarization), MLFF becomes justified.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(json.loads(Path("/tmp/e49_ens_mmgbsa.json").read_text()))
    else:
        main()
