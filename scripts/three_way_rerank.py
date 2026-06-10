"""3-way pose-selection comparison at N=100: OG RAPiDock vs ref2015 vs BSA+clash.

For each complex with 100 RAPiDock diffusion poses + RMSD labels:
  OG RAPiDock  = the diffusion's own pose order (pose_0 = top output)
  ref2015      = rerank 100 poses by ref2015 physics score (lower = better)
  BSA+clash    = rerank by buried surface area − clash penalty (the new ranker)

Reports, per method: RMSD of the SELECTED top-1 pose, best-of-top5, and CAPRI
success (<4 Å) — the production-relevant question "did reranking pick a better
pose than raw diffusion order." Receptors come from the PepPC dataset; a
frame-alignment sanity check is logged per complex.

Writes incrementally to logs/three_way_rerank.json (resumable).
Usage:  python scripts/three_way_rerank.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "logs" / "gen_n100"
PHYS = REPO / "logs" / "diagnosis" / "feats_gen_n100_physics.pkl"
PEPPC = REPO / "datasets" / "training_formatted_peppc"
OUT = REPO / "logs" / "three_way_rerank.json"
CLASH = 3.0
CROP = 10.0


def peppc_receptor(cn: str) -> Path | None:
    d = PEPPC / cn
    if d.is_dir():
        p = next(iter(d.glob("*_protein_pocket.pdb")), None)
        if p:
            return p
    return None


def heavy(pdb: Path):
    lines, xyz = [], []
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        an = ln[12:16].strip()
        if not an or an[0] in ("H", "D"):
            continue
        try:
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        except ValueError:
            continue
        lines.append(ln)
    return lines, (np.array(xyz) if xyz else np.empty((0, 3)))


def sasa(lines):
    import io
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    if not lines:
        return 0.0
    s = PDBParser(QUIET=True).get_structure("x", io.StringIO("\n".join(lines) + "\nEND\n"))
    ShrakeRupley().compute(s, level="A")
    return float(sum(a.sasa for a in s.get_atoms()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=18)
    args = ap.parse_args()
    phys = pickle.load(open(PHYS, "rb"))
    bj = json.loads((GEN / "benchmark_results.json").read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    done = 0
    for cn, d in bj.items():
        if done >= args.limit:
            break
        if cn in out:
            done += 1
            continue
        v = d["pretrained"]
        pdir = Path(v["poses_dir"])
        rms = v.get("ref_rmsds")
        rec_f = peppc_receptor(cn)
        if not pdir.exists() or not rms or rec_f is None:
            continue
        rlines, rxyz = heavy(rec_f)
        if len(rxyz) == 0:
            continue
        poses = []
        for i in range(len(rms)):
            pf = pdir / f"pose_{i}.pdb"
            key = (cn, "pretrained", i)
            if pf.exists() and key in phys:
                poses.append((i, pf, float(rms[i]), np.array(phys[key], float)))
        if len(poses) < 30:
            continue
        # frame check: the lowest-RMSD pose should make contacts (not all far)
        lowest = min(poses, key=lambda x: x[2])
        _, lxyz = heavy(lowest[1])
        mind = np.sqrt(((rxyz[:, None] - lxyz[None]) ** 2).sum(-1)).min() if len(lxyz) else 99
        if mind > 5.0:
            print(f"  {cn}: SKIP frame mismatch (min contact {mind:.1f} Å)", flush=True)
            continue
        # compute BSA + clash per pose
        bsa_list, clash_list, rmsd_list, ref_list, order = [], [], [], [], []
        for i, pf, rm, ph in poses:
            plines, pxyz = heavy(pf)
            if len(pxyz) == 0:
                continue
            d2 = ((rxyz[:, None] - pxyz[None]) ** 2).sum(-1)
            near = d2.min(0) <= CROP ** 2  # receptor atoms near peptide
            crop = [rlines[j] for j in np.where(d2.min(1) <= CROP ** 2)[0]]
            s_pep = sasa(plines); s_rec = sasa(crop); s_cx = sasa(crop + plines)
            bsa_list.append(s_pep + s_rec - s_cx)
            clash_list.append(float((d2.min(0) < CLASH ** 2).sum()))
            rmsd_list.append(rm); ref_list.append(float(ph.sum())); order.append(i)
        if len(rmsd_list) < 30:
            continue
        r = np.array(rmsd_list); bsa = np.array(bsa_list); clash = np.array(clash_list)
        ref = np.array(ref_list); idx = np.array(order)
        zb = (bsa - bsa.mean()) / (bsa.std() + 1e-9)
        zc = (clash - clash.mean()) / (clash.std() + 1e-9)
        bsa_fit = -zb + zc  # lower = better
        # selections (top-1) and best-of-top5
        def pick(score, asc=True):
            o = np.argsort(score if asc else -score)
            return float(r[o[0]]), float(r[o[:5]].min())
        og_order = np.argsort(idx)  # diffusion output order
        og1 = float(r[og_order[0]]); og5 = float(r[og_order[:5]].min())
        ref1, ref5 = pick(ref)
        bf1, bf5 = pick(bsa_fit)
        out[cn] = {"og": [og1, og5], "ref2015": [ref1, ref5], "bsa": [bf1, bf5],
                   "oracle": float(r.min()), "n": len(r)}
        OUT.write_text(json.dumps(out))
        done += 1
        print(f"  {cn}: OG={og1:.1f} ref2015={ref1:.1f} BSA={bf1:.1f} (oracle {r.min():.1f})", flush=True)

    # summary
    if out:
        def col(k, j): return np.array([out[c][k][j] for c in out])
        oc = np.array([out[c]["oracle"] for c in out])
        print(f"\n=== 3-WAY @ N=100, top-1 selected pose ({len(out)} complexes) ===")
        print(f"  {'method':16s} {'mean RMSD':>9s}  {'<4Å':>5s}   {'best-of-top5':>12s}")
        for k, name in [("og", "OG RAPiDock"), ("ref2015", "ref2015 rerank"), ("bsa", "BSA+clash rerank")]:
            t1 = col(k, 0); t5 = col(k, 1)
            print(f"  {name:16s} {t1.mean():7.2f}Å  {100*(t1<4).mean():4.0f}%   {t5.mean():10.2f}Å")
        print(f"  {'oracle':16s} {oc.mean():7.2f}Å")


if __name__ == "__main__":
    main()
