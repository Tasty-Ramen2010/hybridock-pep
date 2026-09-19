#!/usr/bin/env python
"""The one real docking ground truth in the Coventry challenge: PDB 9CCE.

The 18 grid complexes have no experimental structures -- which is exactly why Coventry
told us to fold the binders ourselves.  But the same paper deposited 9CCE: the designed
IDR binder DYNA_1b7 (chain A) with its peptide bound (chain C, FLRRIRPKLKW).  That gives
a crystallographic answer to "are you actually good at docking" for a protein built by
the same method as the 18.

Three arms, so the grid's result can be discounted by a measured amount rather than a
guess:

  xtal_pocket  crystal receptor, cropped by the RAPiDock authors' pocket rule.
               This is exactly our RecentSet benchmark protocol, so it says whether this
               target behaves like the 345 we already reported.
  xtal_full    crystal receptor, whole chain, no crop.  Isolates the cost of handing the
               model an uncropped receptor, which is what the grid does.
  esmfold_full ESMFold model of chain A's sequence, whole chain.  Isolates the cost of a
               PREDICTED receptor.  This arm is the grid's exact protocol, so its gap to
               xtal_pocket is the discount that belongs on every grid number.

For esmfold_full the poses come out in the model's frame, so the model is superimposed
onto crystal chain A (CA Kabsch) and the same transform is applied to every pose before
RMSD -- the peptide is never used to fit anything.

Usage: coventry_9cce_test.py [stage]   stage = prep | dock | score | all
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
D = ROOT / "datasets/coventry/9cce"
XTAL = D / "9cce.pdb"
RAPIDOCK_PY = "/home/igem/miniconda3/envs/rapidock/bin/python"
INFER = ROOT / "third_party/RAPiDock"
MODEL_DIR = ROOT / "third_party/RAPiDock_finetuned/longft_tanh"
CKPT = "rapidock_finetuned_epoch010.pt"
POCKET_CUT = 20.0          # authors' pocket_trunction threshold
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def atoms(path, chain=None, ca_only=False, heavy=True):
    out = []
    for l in open(path):
        if not l.startswith("ATOM"):
            continue
        if chain and l[21] != chain:
            continue
        if heavy and l[76:78].strip() == "H":
            continue
        if ca_only and l[12:16].strip() != "CA":
            continue
        out.append(l)
    return out


def coords(lines):
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lines])


def write_pdb(lines, path, chain=None):
    with open(path, "w") as fh:
        for l in lines:
            fh.write((l[:21] + chain + l[22:]) if chain else l)
        fh.write("END\n")


def kabsch(P, Q):
    """Rotation+translation taking P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def pick_pair():
    """Choose the receptor/peptide chain pair by contact, not by chain letter.

    9CCE has two copies of each: chains A and B are the binder, C and D the peptide --
    and the pairing is A-D / B-C, NOT A-C.  Assuming alphabetical order silently pairs
    chains 16.7 A apart and every RMSD that follows is meaningless, so detect it.
    """
    from scipy.spatial import cKDTree
    chains = {}
    for l in atoms(XTAL):
        chains.setdefault(l[21], []).append(l)
    nres = {c: sum(1 for l in ls if l[12:16].strip() == "CA") for c, ls in chains.items()}
    recs = [c for c in chains if nres[c] >= 50]
    peps = [c for c in chains if nres[c] < 50]
    best = None
    for p in peps:
        for r in recs:
            d, _ = cKDTree(coords(chains[r])).query(coords(chains[p]))
            n = int((d < 4.5).sum())
            if best is None or (n, nres[p]) > (best[2], nres[best[1]]):
                best = (r, p, n)
    print(f"chain pairing by contact: receptor {best[0]} ({nres[best[0]]} aa) + "
          f"peptide {best[1]} ({nres[best[1]]} aa), {best[2]} contacts <4.5 A")
    return chains[best[0]], chains[best[1]]


def prep():
    D.mkdir(parents=True, exist_ok=True)
    rec, pep = pick_pair()
    seq = "".join(AA3.get(l[17:20].strip(), "X") for l in pep if l[12:16].strip() == "CA")
    write_pdb(pep, D / "peptide_xtal.pdb")
    write_pdb(rec, D / "receptor_xtal_full.pdb")

    # authors' pocket rule: residues within 20 A of the peptide, then keep the
    # contiguous MIN..MAX residue range (fills gaps), per chain.
    pxyz = coords(pep)
    near = set()
    for l in rec:
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        if np.linalg.norm(pxyz - xyz, axis=1).min() <= POCKET_CUT:
            near.add(int(l[22:26]))
    lo, hi = min(near), max(near)
    write_pdb([l for l in rec if lo <= int(l[22:26]) <= hi], D / "receptor_xtal_pocket.pdb")

    rseq = "".join(AA3.get(l[17:20].strip(), "X") for l in rec if l[12:16].strip() == "CA")
    rchain, pchain = rec[0][21], pep[0][21]
    (D / "binder.fasta").write_text(f">DYNA_1b7_chain{rchain}\n{rseq}\n")
    (D / "meta.csv").write_text("peptide_seq,receptor_len,pocket_range,rec_chain,pep_chain\n"
                                f"{seq},{len(rseq)},{lo}-{hi},{rchain},{pchain}\n")
    print(f"peptide  : {seq} ({len(seq)} aa)")
    print(f"receptor : {len(rseq)} aa, pocket residues {lo}-{hi} "
          f"({sum(1 for l in atoms(D / 'receptor_xtal_pocket.pdb') if l[12:16].strip() == 'CA')} res)")
    return seq, rseq


def fold_receptor(rseq: str):
    out = D / "receptor_esmfold.pdb"
    if out.exists():
        print("esmfold receptor already present")
        return
    code = (
        "import torch;from transformers import AutoTokenizer,EsmForProteinFolding;"
        f"s='{rseq}';"
        "t=AutoTokenizer.from_pretrained('facebook/esmfold_v1');"
        "m=EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',low_cpu_mem_usage=True)"
        ".cuda().eval();m.esm=m.esm.half();m.trunk.set_chunk_size(64);"
        "e=t([s],return_tensors='pt',add_special_tokens=False);"
        "o=m(e['input_ids'].cuda(),attention_mask=e['attention_mask'].cuda());"
        f"open('{out}','w').write(m.output_to_pdb(o)[0]);"
        "print('pLDDT %.1f'%(o['plddt'][0,:,1].mean()*100));"
        # hard exit: PyTorch does not release the CUDA context on normal teardown here,
        # and a lingering process pins ~5 GB of RAM plus its GPU allocation.
        "import sys,os;sys.stdout.flush();os._exit(0)"
    )
    env = dict(os.environ, LD_LIBRARY_PATH="/home/igem/miniconda3/envs/score-env/lib")
    r = subprocess.run(["/home/igem/miniconda3/envs/score-env/bin/python", "-c", code],
                       capture_output=True, text=True, env=env)
    print("esmfold:", [x for x in r.stdout.splitlines() if "pLDDT" in x] or r.stderr[-300:])


ARMS = {
    "xtal_pocket": "receptor_xtal_pocket.pdb",
    "xtal_full": "receptor_xtal_full.pdb",
    "esmfold_full": "receptor_esmfold.pdb",
}
# Both models run through the SAME upstream code path, so the only thing that differs
# between them is the checkpoint -- otherwise a code-path difference would be read as a
# model difference (which is exactly what the SiLU/Tanh fork bug did to us in September).
MODELS = {
    "rapidock_og": (ROOT / "third_party/RAPiDock/train_models/CGTensorProductEquivariantModel",
                    "rapidock_local.pt"),
    "hybridock_ft": (MODEL_DIR, CKPT),
}


def dock(seq: str, n: int = 24):
    out_root = ROOT / "runs/coventry/9cce"
    out_root.mkdir(parents=True, exist_ok=True)
    for model, (mdir, ckpt) in MODELS.items():
        for arm, rec in ARMS.items():
            recp = D / rec
            tag = f"{model}__{arm}"
            if not recp.exists():
                print(f"  {tag}: receptor missing, skipped")
                continue
            if glob.glob(str(out_root / tag / "rank*.pdb")):
                print(f"  {tag}: already docked")
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
                w = csv.writer(tf)
                w.writerow(["complex_name", "protein_description", "peptide_description"])
                w.writerow([tag, str(recp), seq])
                p = tf.name
            t0 = time.time()
            subprocess.run([RAPIDOCK_PY, str(INFER / "inference.py"),
                            "--protein_peptide_csv", p, "--out", str(out_root),
                            "--output_dir", str(out_root), "--model_dir", str(mdir),
                            "--ckpt", ckpt, "--N", str(n), "--batch_size", "8",
                            "--inference_steps", "16", "--actual_steps", "16",
                            "--no_final_step_noise", "--conformation_partial", "1:1:1",
                            "--scoring_function", "none", "--confidence_ckpt", "null",
                            "--cpu", "4"], cwd=str(INFER), capture_output=True, text=True)
            os.unlink(p)
            print(f"  {tag}: {len(glob.glob(str(out_root / tag / 'rank*.pdb')))} poses "
                  f"({time.time() - t0:.0f}s)", flush=True)


def score():
    sys.path.insert(0, str(ROOT / "scripts"))
    from dockq_rs import score_pose
    meta = list(csv.DictReader(open(D / "meta.csv")))[0]
    ref_pep = D / "peptide_xtal.pdb"
    ref_ca = coords(atoms(ref_pep, ca_only=True))
    xtal_rec_ca = {int(l[22:26]): coords([l])[0]
                   for l in atoms(XTAL, meta["rec_chain"], ca_only=True)}

    print(f"\n9CCE  DYNA_1b7 + FLRRIRPKLKW   best-of-24, direct CA RMSD (no superposition")
    print(f"of the peptide: receptor frames are shared, so this is placement + conformation)")
    print(f"\n{'model':<14}{'receptor arm':<14}{'best RMSD':>10}{'median':>8}"
          f"{'<=2A':>8}{'<=5A':>8}{'best DockQ':>11}  {'CAPRI':<10}")
    for model in MODELS:
        for arm, rec in ARMS.items():
            d = ROOT / "runs/coventry/9cce" / f"{model}__{arm}"
            poses = sorted(d.glob("rank*.pdb"),
                           key=lambda p: int(re.search(r"\d+", p.name).group()))
            if not poses:
                continue
            _score_arm(model, arm, rec, poses, ref_pep, ref_ca, xtal_rec_ca)


def _score_arm(model, arm, rec, poses, ref_pep, ref_ca, xtal_rec_ca):
        R, t = np.eye(3), np.zeros(3)
        recp = D / rec
        if arm == "esmfold_full":
            # superimpose the MODEL receptor onto crystal chain A using CA only,
            # matched by sequential index (ESMFold numbers 1..N over the same sequence)
            mod = atoms(recp, ca_only=True)
            ks = sorted(xtal_rec_ca)
            m = min(len(mod), len(ks))
            R, t = kabsch(coords(mod[:m]), np.array([xtal_rec_ca[k] for k in ks[:m]]))
            fit = np.sqrt((((coords(mod[:m]) @ R.T + t)
                            - np.array([xtal_rec_ca[k] for k in ks[:m]])) ** 2).sum(1).mean())
            if model == "rapidock_og":
                print(f"  (esmfold receptor vs crystal chain A: CA RMSD {fit:.2f} A)")
        rms, dqs = [], []
        tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-1000"))
        for p in poses:
            pl = atoms(p, ca_only=True)
            xyz = coords(pl) @ R.T + t
            m = min(len(xyz), len(ref_ca))
            rms.append(float(np.sqrt(((xyz[:m] - ref_ca[:m]) ** 2).sum(1).mean())))
            # DockQ needs the transformed pose on disk, in the crystal frame
            allp = atoms(p)
            tp = tmp / p.name
            with tp.open("w") as fh:
                for l in allp:
                    v = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) @ R.T + t
                    fh.write(f"{l[:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{l[54:]}")
                fh.write("END\n")
            try:
                dqs.append(score_pose(str(D / "receptor_xtal_full.pdb"), str(ref_pep), str(tp)))
            except Exception:  # noqa: BLE001
                pass
        dqs = [x for x in dqs if x == x]
        bd = max(dqs) if dqs else float("nan")
        capri = ("high" if bd >= 0.80 else "medium" if bd >= 0.49
                 else "acceptable" if bd >= 0.23 else "incorrect")
        print(f"{model:<14}{arm:<14}{min(rms):>10.2f}{float(np.median(rms)):>8.2f}"
              f"{sum(1 for r in rms if r <= 2):>5}/{len(rms):<2}"
              f"{sum(1 for r in rms if r <= 5):>5}/{len(rms):<2}"
              f"{bd:>11.3f}  {capri:<10}", flush=True)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    seq = rseq = None
    if stage in ("prep", "all"):
        seq, rseq = prep()
    if stage in ("fold", "all"):
        # separate stage: ESMFold loads ~5 GB on the GPU, so it is kept out of the
        # crystal-only arms and run when the docking grid is not competing for VRAM.
        rseq = rseq or (D / "binder.fasta").read_text().splitlines()[1]
        fold_receptor(rseq)
    if seq is None:
        r = list(csv.DictReader(open(D / "meta.csv")))[0]
        seq = r["peptide_seq"]
    if stage in ("dock", "all"):
        dock(seq)
    if stage in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
