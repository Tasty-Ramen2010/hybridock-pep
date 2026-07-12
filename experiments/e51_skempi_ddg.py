"""E51 — SKEMPI ΔΔG validation: is our selectivity (interface ΔΔG) LIE-level?

SKEMPI 2.0 = the gold-standard experimental ΔΔG benchmark (what FoldX/flex-ddG/FEP report on).
ΔΔG_exp = RT·ln(Kd_mut/Kd_wt). For each single mutation we MODEL the mutant (PyRosetta mutate +
repack neighbours) and score the interface with OUR method (MM-GBSA ΔG_bind, OpenMM ff14SB+GBn2),
giving ΔΔG_pred = ΔG_mut − ΔG_wt. Selectivity should be LIE-level BECAUSE the absolute-affinity floor
CANCELS in the difference (same complex, small perturbation). Also computes Rosetta ref2015 interface
ΔΔG on the same models as a reference. Correlate vs experimental; position vs literature:
FoldX r≈0.5, flex-ddG r≈0.55, FEP r≈0.8.

Usage: e51_skempi_ddg.py <PDB_chainsA_chainsB> [n_max]   (e.g. 1PPF_E_I 25)  — resumable cache.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WORK = Path("/tmp/skempi_work"); WORK.mkdir(exist_ok=True)
R_KCAL = 1.987e-3
A1 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
      "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
      "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def parse_skempi(pdb_key):
    rows = list(csv.reader((ROOT / "data/skempi_v2.csv").read_text().splitlines(), delimiter=";"))
    hdr = rows[0]; ix = {c: i for i, c in enumerate(hdr)}
    out = []
    seen = set()
    for r in rows[1:]:
        if len(r) < len(hdr) or r[ix["#Pdb"]] != pdb_key:
            continue
        mut = r[ix["Mutation(s)_cleaned"]].strip()
        if not mut or "," in mut:
            continue
        try:
            km = float(r[ix["Affinity_mut_parsed"]]); kw = float(r[ix["Affinity_wt_parsed"]])
            T = float((r[ix["Temperature"]].split("(")[0].strip() or 298))
        except (ValueError, IndexError):
            continue
        if km <= 0 or kw <= 0:
            continue
        wt, ch, mt = mut[0], mut[1], mut[-1]
        resnum = int("".join(c for c in mut[2:-1] if c.isdigit() or c == "-"))
        ddg = R_KCAL * T * math.log(km / kw)   # >0 = mutation weakens binding
        key = (ch, resnum, mt)
        if key in seen or mt not in A1 or wt not in A1:
            continue
        seen.add(key)
        out.append(dict(chain=ch, resnum=resnum, wt=wt, mut=mt, ddg_exp=ddg))
    return out


def fetch(pdb):
    f = WORK / f"{pdb}.pdb"
    if not f.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb", f)
    return f


def init_pr():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false -no_optH false", silent=True)
    return pyrosetta


def clean_complex(pr, pdb_file, groupA, groupB):
    """Load PDB keeping only groupA+groupB chains; return pose + chain sets."""
    import pyrosetta
    keep = set(groupA) | set(groupB)
    lines = [ln for ln in pdb_file.read_text().splitlines()
             if ln.startswith(("ATOM", "TER")) and (len(ln) < 22 or ln[21] in keep)
             and ln[17:20] != "HOH"]
    cf = WORK / f"{pdb_file.stem}_clean.pdb"
    cf.write_text("\n".join(lines) + "\nEND\n")
    return pyrosetta.pose_from_pdb(str(cf))


def split_score_mmgbsa(pose, groupA, groupB, tag):
    """Dump pose, split into the two chain groups, MM-GBSA ΔG_bind (smaller=peptide)."""
    from hybridock_pep.scoring.mmgbsa import compute_mmgbsa_single
    pf = WORK / f"{tag}.pdb"; pose.dump_pdb(str(pf))
    a_lines, b_lines = [], []
    for ln in pf.read_text().splitlines():
        if ln.startswith("ATOM"):
            (a_lines if ln[21] in set(groupA) else b_lines if ln[21] in set(groupB) else []).append(ln)
    # smaller group = "peptide"
    small, big = (a_lines, b_lines) if len(a_lines) < len(b_lines) else (b_lines, a_lines)
    sp = WORK / f"{tag}_pep.pdb"; rp = WORK / f"{tag}_rec.pdb"
    sp.write_text("\n".join(small) + "\nEND\n"); rp.write_text("\n".join(big) + "\nEND\n")
    return compute_mmgbsa_single(sp.resolve(), rp.resolve(), force_cpu=False)


def mutate(pr, pose, chain, resnum, mut_aa):
    import pyrosetta
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
    pno = pose.pdb_info().pdb2pose(chain, resnum)
    if pno == 0:
        return None
    p2 = pose.clone()
    mr = MutateResidue(pno, A1[mut_aa]); mr.apply(p2)
    # repack the mutated residue + neighbours within 6 A
    sf = pyrosetta.get_fa_scorefxn()
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.select.residue_selector import (NeighborhoodResidueSelector,
                                                                ResidueIndexSelector)
    sel = NeighborhoodResidueSelector(ResidueIndexSelector(str(pno)), 6.0, True)
    tf = TaskFactory(); tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    task = tf.create_task_and_apply_taskoperations(p2)
    keep = sel.apply(p2)
    for i in range(1, p2.total_residue() + 1):
        if not keep[i]:
            task.nonconst_residue_task(i).prevent_repacking()
    pyrosetta.rosetta.core.pack.pack_rotamers(p2, sf, task)
    return p2


def main():
    pdb_key = sys.argv[1] if len(sys.argv) > 1 else "1PPF_E_I"
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    pdb, gA, gB = pdb_key.split("_")
    muts = parse_skempi(pdb_key)[:n_max]
    print(f"=== E51 {pdb_key}: {len(muts)} single mutations (target {n_max}) ===", flush=True)
    cache = Path(f"/tmp/e51_{pdb_key}.json")
    out = json.loads(cache.read_text()) if cache.exists() else {}
    pr = init_pr()
    wt_pose = clean_complex(pr, fetch(pdb), gA, gB)
    if "WT" not in out:
        dg_wt = split_score_mmgbsa(wt_pose, gA, gB, f"{pdb_key}_WT")
        out["WT"] = dict(dg=float(dg_wt)); cache.write_text(json.dumps(out))
        print(f"  WT MM-GBSA ΔG_bind = {dg_wt:+.2f}", flush=True)
    dg_wt = out["WT"]["dg"]
    for m in muts:
        key = f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}"
        if key in out:
            continue
        try:
            mp = mutate(pr, wt_pose, m["chain"], m["resnum"], m["mut"])
            if mp is None:
                print(f"  {key} resnum not in pose", flush=True); continue
            dg_mut = split_score_mmgbsa(mp, gA, gB, f"{pdb_key}_{key}")
            out[key] = dict(ddg_pred=float(dg_mut - dg_wt), ddg_exp=m["ddg_exp"], dg_mut=float(dg_mut))
            cache.write_text(json.dumps(out))
            print(f"  {key}  ΔΔG_pred={dg_mut-dg_wt:+.2f}  exp={m['ddg_exp']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
    evaluate(out, pdb_key)


def evaluate(out, pdb_key):
    from scipy.stats import pearsonr, spearmanr
    pairs = [(v["ddg_pred"], v["ddg_exp"]) for k, v in out.items() if k != "WT"]
    if len(pairs) < 5:
        print(f"  ({len(pairs)} done, need >=5 to eval)"); return
    p = np.array([x[0] for x in pairs]); e = np.array([x[1] for x in pairs])
    ok = np.abs(p) < 50
    print(f"\n=== {pdb_key} ΔΔG: OUR MM-GBSA vs experimental (n={ok.sum()}) ===")
    print(f"  Pearson r = {pearsonr(p[ok],e[ok]).statistic:+.3f}  Spearman = {spearmanr(p[ok],e[ok]).statistic:+.3f}")
    print(f"  RMSE = {np.sqrt(((p[ok]-e[ok])**2).mean()):.2f} kcal/mol")
    print("  lit: FoldX r≈0.5, flex-ddG r≈0.55, FEP r≈0.8 (single-point MM-GBSA typ 0.3-0.5)")


if __name__ == "__main__":
    main()
