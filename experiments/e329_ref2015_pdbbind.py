"""E329 — ref2015 (FlexPepDock-family) interface-dG on the PDBbind ~900 peptide-Kd set.

Head-to-head vs our scorer. For each complex in data/pdbbind_peptides.jsonl:
  - rebuild the peptide PDB from ligand.mol2 (same residue-from-backbone-N logic as e108)
  - run ref2015 InterfaceAnalyzer dG_separated (peptide chain P vs protein chain R)
Saves data/e329_ref2015_pdbbind.json incrementally (resumable). Correlation reported at the end.

Usage: python experiments/e329_ref2015_pdbbind.py [--limit N]
Run in score-env (has pyrosetta).
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data/pdbbind_peptides.jsonl"
PL_ROOT = ROOT / "data/drive_pull/pl/P-L"
OUT = ROOT / "data/e329_ref2015_pdbbind.json"

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def build_dirmap():
    m = {}
    for yd in PL_ROOT.iterdir():
        if yd.is_dir():
            for cd in yd.iterdir():
                if cd.is_dir():
                    m[cd.name] = cd
    return m


def mol2_to_pep_pdb(mol2: Path, out: Path):
    lines = mol2.read_text().splitlines()
    if "@<TRIPOS>ATOM" not in lines:
        return None
    a = lines.index("@<TRIPOS>ATOM")
    atoms = []
    for ln in lines[a + 1:]:
        if ln.startswith("@"):
            break
        f = ln.split()
        if len(f) < 9:
            continue
        name, x, y, z, sname = f[1], f[2], f[3], f[4], f[7]
        rn = "".join(c for c in sname if c.isalpha()).upper()[:3]
        if rn not in AA3:
            return None
        try:
            atoms.append((name, rn, float(x), float(y), float(z)))
        except ValueError:
            return None
    rec, seq, resnum = [], [], 0
    for aid, (name, rn, x, y, z) in enumerate(atoms, start=1):
        if name == "N":
            resnum += 1
            seq.append(rn)
        assigned = max(resnum, 1)
        elem = name[0]
        rec.append(f"ATOM  {aid:>5} {name:<4} {rn:>3} P{assigned:>4}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}")
    if len(seq) < 3:
        return None
    out.write_text("\n".join(rec) + "\nTER\nEND\n")
    return "".join(AA3[r] for r in seq)


def protein_as_chain_R(prot: Path, out: Path):
    keep = []
    for ln in prot.read_text().splitlines():
        if ln.startswith("ATOM"):
            keep.append(ln[:21] + "R" + ln[22:])
    out.write_text("\n".join(keep) + "\nTER\nEND\n")


def init_rosetta():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false", silent=True)
    return pyrosetta


def score_complex(pr, pep_pdb: Path, prot_pdb: Path, tmp: Path):
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    pep_lines = [ln for ln in pep_pdb.read_text().splitlines() if ln.startswith("ATOM")]
    prot_lines = [ln for ln in prot_pdb.read_text().splitlines() if ln.startswith("ATOM")]
    tmp.write_text("\n".join(pep_lines) + "\nTER\n" + "\n".join(prot_lines) + "\nTER\nEND\n")
    pose = pr.pose_from_pdb(str(tmp))
    sfxn = pr.get_fa_scorefxn()
    total = float(sfxn(pose))
    iam = InterfaceAnalyzerMover("P_R")
    iam.set_pack_separated(True)
    iam.apply(pose)
    return dict(ros_total=total, ros_ifdG=float(iam.get_interface_dG()))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    dirmap = build_dirmap()
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r["pdb"] for r in out}
    todo = [r for r in rows if r["pdb"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"=== E329 ref2015 on PDBbind: {len(rows)} total, {len(done)} done, {len(todo)} this run ===",
          flush=True)

    pr = init_rosetta()
    tmpdir = Path("/tmp/e329"); tmpdir.mkdir(exist_ok=True)
    n_ok = n_fail = 0
    for i, r in enumerate(todo):
        pid = r["pdb"]
        d = dirmap.get(pid)
        if d is None:
            n_fail += 1; continue
        mol2 = d / f"{pid}_ligand.mol2"
        prot = d / f"{pid}_protein.pdb"
        pep_pdb = tmpdir / f"{pid}_pep.pdb"
        prot_pdb = tmpdir / f"{pid}_prot.pdb"
        merged = tmpdir / f"{pid}_merged.pdb"
        try:
            seq = mol2_to_pep_pdb(mol2, pep_pdb)
            if not seq:
                n_fail += 1; continue
            protein_as_chain_R(prot, prot_pdb)
            s = score_complex(pr, pep_pdb, prot_pdb, merged)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 15:
                print(f"  {pid} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        finally:
            for p in (pep_pdb, prot_pdb, merged):
                if p.exists():
                    p.unlink()
        out.append(dict(pdb=pid, y=r["y"], length=r["length"], **s))
        n_ok += 1
        if n_ok % 25 == 0:
            OUT.write_text(json.dumps(out))
            print(f"  [{i+1}/{len(todo)}] ok={n_ok} fail={n_fail} last {pid} "
                  f"ifdG={s['ros_ifdG']:+.1f}", flush=True)
    OUT.write_text(json.dumps(out))
    print(f"=== done: ok={n_ok} fail={n_fail}, total cached={len(out)} ===", flush=True)

    if len(out) >= 10:
        y = np.array([r["y"] for r in out])
        x = np.array([r["ros_ifdG"] for r in out])
        pr_r = pearsonr(x, y)[0]; sp = spearmanr(x, y).statistic
        print(f"\n=== ref2015 interface-dG vs experimental ΔG (n={len(out)}) ===")
        print(f"  Pearson r = {pr_r:+.3f}   Spearman = {sp:+.3f}")


if __name__ == "__main__":
    main()
