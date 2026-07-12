"""E91 — ref2015 (FlexPepDock-family) interface-dG affinity scoring on crystal-65, for the scorecard.

FlexPepDock correlates a reweighted Rosetta interface energy with affinity (lit r=0.59 within-target).
We compute the same physics: ref2015 InterfaceAnalyzer dG_separated on the crystal pose. Uses the PDB
paths in data/benchmark_crystal.json directly (no /tmp dependency). CPU, crash-safe append.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e91_ref2015_cr65.json"


def init_rosetta():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false", silent=True)
    return pyrosetta


def score_complex(pr, pep_pdb, poc_pdb):
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    # merge peptide (chain P) + pocket (chain R)
    pep_lines, poc_lines = [], []
    for src, tag, store in [(pep_pdb, "P", pep_lines), (poc_pdb, "R", poc_lines)]:
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")):
                store.append(ln[:21] + tag + ln[22:])
    merged = Path("/tmp/e91_merged.pdb")
    merged.write_text("\n".join(pep_lines) + "\nTER\n" + "\n".join(poc_lines) + "\nTER\nEND\n")
    pose = pr.pose_from_pdb(str(merged))
    sfxn = pr.get_fa_scorefxn()
    total = float(sfxn(pose))
    iam = InterfaceAnalyzerMover("P_R")
    iam.set_pack_separated(True)
    iam.apply(pose)
    return dict(ros_total=total, ros_ifdG=float(iam.get_interface_dG()))


def main():
    pr = init_rosetta()
    b = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["pdb"] for r in out}
    print(f"=== E91 ref2015 interface-dG on crystal-65 ({len(b)} complexes) ===", flush=True)
    for meta in b:
        pdb = meta["pdb"]
        if pdb in done:
            continue
        try:
            s = score_complex(pr, meta["peptide_pdb"], meta["pocket_pdb"])
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        out.append(dict(pdb=pdb, y=meta["dg_exp"], **s))
        OUT.write_text(json.dumps(out))
        print(f"  {pdb} ifdG={s['ros_ifdG']:+.2f} total={s['ros_total']:+.1f}", flush=True)
    if len(out) >= 10:
        y = np.array([r["y"] for r in out]); x = np.array([r["ros_ifdG"] for r in out])
        print(f"\n=== ref2015 interface-dG vs experimental ΔG (n={len(out)}) ===")
        print(f"  Pearson={pearsonr(x, y)[0]:+.3f}  |r|={abs(pearsonr(x, y)[0]):.3f}  "
              f"Spearman={spearmanr(x, y).statistic:+.3f}")


if __name__ == "__main__":
    main()
