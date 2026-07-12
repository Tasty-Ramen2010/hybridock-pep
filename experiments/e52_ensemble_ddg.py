"""E52 — ENSEMBLE (true-LIE) ΔΔG: does MD-averaging the interaction energy reach LIE-level?

E51 single-point MM-GBSA ΔΔG is too noisy (pooled ~0.24, complex-to-complex −0.10..+0.30) — the same
reason flex-ddG averages ~35 backrub models. LIE itself = MD-AVERAGED interaction energy. So here:
    ΔΔG_pred = <E_int>_mut − <E_int>_wt           (<E_int> from sample_interaction_energies, e49)
This denoises the single-point estimate the way LIE/flex-ddG do. Tested on the complexes E51 scored,
same mutant models (PyRosetta mutate+pack), comparing ensemble ΔΔG vs single-point vs experimental.

Usage: e52_ensemble_ddg.py <PDB_chainsA_chainsB> [n_max]   resumable.
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
sys.path.insert(0, str(ROOT / "scripts"))
import e51_skempi_ddg as E51  # noqa: E402
from hybridock_pep.scoring.interaction_entropy import sample_interaction_energies  # noqa: E402

N_FRAMES, STEPS = 40, 250


def ens_score(pose, gA, gB, tag):
    """MD-averaged <E_int> for the split complex (smaller group = peptide)."""
    pf = Path(f"/tmp/skempi_work/{tag}.pdb"); pose.dump_pdb(str(pf))
    a, b = [], []
    for ln in pf.read_text().splitlines():
        if ln.startswith("ATOM"):
            (a if ln[21] in set(gA) else b if ln[21] in set(gB) else []).append(ln)
    small, big = (a, b) if len(a) < len(b) else (b, a)
    sp = Path(f"/tmp/skempi_work/{tag}_epep.pdb"); rp = Path(f"/tmp/skempi_work/{tag}_erec.pdb")
    sp.write_text("\n".join(small) + "\nEND\n"); rp.write_text("\n".join(big) + "\nEND\n")
    e = sample_interaction_energies(sp.resolve(), rp.resolve(), n_frames=N_FRAMES,
                                    steps_between_frames=STEPS, force_cpu=False)
    return float(e.mean())


def main():
    pdb_key = sys.argv[1] if len(sys.argv) > 1 else "1PPF_E_I"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    pdb, gA, gB = pdb_key.split("_")
    sp_cache = json.loads(Path(f"/tmp/e51_{pdb_key}.json").read_text())  # reuse exp ΔΔG + single-pt
    muts = E51.parse_skempi(pdb_key)[:n_max]
    cache = Path(f"/tmp/e52_{pdb_key}.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    pr = E51.init_pr()
    wt_pose = E51.clean_complex(pr, E51.fetch(pdb), gA, gB)
    print(f"=== E52 ensemble ΔΔG {pdb_key}: {len(muts)} muts ===", flush=True)
    if "WT" not in out:
        out["WT"] = dict(eint=ens_score(wt_pose, gA, gB, f"{pdb_key}_eWT")); cache.write_text(json.dumps(out))
    eint_wt = out["WT"]["eint"]
    for m in muts:
        key = f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}"
        if key in out or key not in sp_cache:
            continue
        try:
            mp = E51.mutate(pr, wt_pose, m["chain"], m["resnum"], m["mut"])
            if mp is None:
                continue
            ei = ens_score(mp, gA, gB, f"{pdb_key}_e{key}")
            out[key] = dict(ddg_ens=float(ei - eint_wt), ddg_single=sp_cache[key]["ddg_pred"],
                            ddg_exp=m["ddg_exp"]); cache.write_text(json.dumps(out))
            print(f"  {key} ens={ei-eint_wt:+.2f} single={sp_cache[key]['ddg_pred']:+.2f} exp={m['ddg_exp']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {str(e)[:40]}", flush=True)
    evaluate(out, pdb_key)


def evaluate(out, pdb_key):
    from scipy.stats import pearsonr, spearmanr
    r = [(v["ddg_ens"], v["ddg_single"], v["ddg_exp"]) for k, v in out.items()
         if k != "WT" and abs(v.get("ddg_ens", 99)) < 100]
    if len(r) < 5:
        print(f"  ({len(r)} done)"); return
    en = np.array([x[0] for x in r]); sg = np.array([x[1] for x in r]); ex = np.array([x[2] for x in r])
    print(f"\n=== {pdb_key} (n={len(r)}): ENSEMBLE vs SINGLE-point ΔΔG ===")
    print(f"  ensemble <E_int>  Pearson {pearsonr(en,ex).statistic:+.3f}  Spearman {spearmanr(en,ex).statistic:+.3f}")
    print(f"  single-point      Pearson {pearsonr(sg,ex).statistic:+.3f}  Spearman {spearmanr(sg,ex).statistic:+.3f}")


if __name__ == "__main__":
    main()
