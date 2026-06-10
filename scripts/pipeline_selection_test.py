"""Production-realistic pose SELECTION test (no native ranking).

In production there is no native to rank against — the pipeline SELECTS the
best-scoring pose and reports it. So the meaningful metric is the RMSD of the
*selected* top-1 pose, not τ. This compares two selectors on identical RAPiDock
poses (isolating the scoring change, FastRelax orthogonal to both):

  Pipeline B (OG-ish):  ref2015 only        → select best → report its RMSD
  Pipeline A (proposed): ref2015 + BSA + interface → select best → report RMSD

Selection is leakage-free: a ridge predicting RMSD is fit leave-one-complex-out,
then used to pick the top-1 pose of the held-out complex. We report the actual
RMSD of that pick + CAPRI-style success rates, plus the diffusion-top, random,
and oracle (best-of-set) brackets.

  --build   parse poses, compute BSA (Biopython SASA) + cache to JSON (incremental)
  (no arg)  run the selection comparison on the cache + show 5 example structures
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.stats as ss

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "logs" / "analysis_bench300"
FOLDX = REPO / "logs" / "foldx_v2_features.json"
PHYS = REPO / "logs" / "diagnosis" / "feats_bench300_physics.pkl"
CACHE = REPO / "logs" / "pipeline_select_cache.json"

INTER = ("fa_rep", "n_clash", "hyd_pack", "desolv_polar", "n_hb",
         "buried_unsat", "n_sb", "shape_var", "band_frac", "n_contact")


def _sasa_sum(struct, sr):
    sr.compute(struct, level="A")
    return float(sum(a.sasa for a in struct.get_atoms()))


def build():
    import warnings
    warnings.simplefilter("ignore")
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.Chain import Chain

    P = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    pool = json.loads(FOLDX.read_text())
    phys = pickle.load(open(PHYS, "rb"))
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    cns = list(pool)
    for j, cn in enumerate(cns):
        if cn in out:
            continue
        rec_f = BENCH / cn / "scoring" / "receptor_cropped.pdb"
        if not rec_f.exists():
            continue
        try:
            rec0 = P.get_structure("r0", str(rec_f))
            s_rec = _sasa_sum(rec0, sr)
        except Exception:
            continue
        rows, rmsd = [], []
        ok = True
        for f, rm in zip(pool[cn]["feats"], pool[cn]["rmsd"]):
            pi = f["_pose_idx"]
            key = (cn, "pretrained", pi)
            pose_f = BENCH / cn / "pretrained" / "poses" / f"pose_{pi}.pdb"
            if key not in phys or not pose_f.exists():
                ok = False
                break
            try:
                pep = P.get_structure("p", str(pose_f))
                s_pep = _sasa_sum(pep, sr)
                rec2 = P.get_structure("r2", str(rec_f))
                c = Chain("9")
                for ch in pep[0]:
                    for res in ch:
                        c.add(res.copy())
                rec2[0].add(c)
                s_cx = _sasa_sum(rec2, sr)
            except Exception:
                ok = False
                break
            bsa = s_pep + s_rec - s_cx
            feat = [float(x) for x in phys[key]] + [bsa] + [float(f[k]) for k in INTER]
            rows.append(feat)
            rmsd.append(float(rm))
        if ok and len(rows) >= 3:
            out[cn] = {"X": rows, "rmsd": rmsd}
        if (j + 1) % 15 == 0:
            CACHE.write_text(json.dumps(out))
            print(f"  built {len(out)} complexes ({j+1}/{len(cns)})", flush=True)
    CACHE.write_text(json.dumps(out))
    print(f"Cached {len(out)} complexes → {CACHE.relative_to(REPO)}")


def _loo_select(data, cols):
    cns = list(data)
    sel = []
    for h in cns:
        tr = [c for c in cns if c != h]
        Xtr = np.vstack([np.array(data[c]["X"])[:, cols] for c in tr])
        ytr = np.concatenate([np.array(data[c]["rmsd"]) for c in tr])
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.hstack([(Xtr - m) / s, np.ones((len(Xtr), 1))])
        R = np.eye(A.shape[1]); R[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
        Xh = np.array(data[h]["X"])[:, cols]
        pred = (Xh - m) / s @ w[:-1] + w[-1]
        sel.append((h, int(np.argmin(pred)), float(data[h]["rmsd"][int(np.argmin(pred))])))
    return sel


def experiments():
    data = json.loads(CACHE.read_text())
    cns = list(data)
    nfeat = len(data[cns[0]]["X"][0])
    ref_cols = list(range(16))            # ref2015 physics-16
    all_cols = list(range(nfeat))         # + BSA + interface
    print(f"Selection test — {len(cns)} complexes, {nfeat} features/pose "
          f"(ref2015-16 + BSA + {len(INTER)} interface)\n")

    selB = _loo_select(data, ref_cols)
    selA = _loo_select(data, all_cols)
    rB = np.array([x[2] for x in selB]); rA = np.array([x[2] for x in selA])
    pose0 = np.array([data[c]["rmsd"][0] for c in cns])
    rand = np.array([np.mean(data[c]["rmsd"]) for c in cns])
    oracle = np.array([min(data[c]["rmsd"]) for c in cns])

    def line(name, a):
        print(f"  {name:34s} mean={a.mean():.2f}Å  <2Å={100*(a<2).mean():4.0f}%  "
              f"<4Å(CAPRI ok)={100*(a<4).mean():4.0f}%")
    print("SELECTED top-1 pose quality (lower mean / higher % = better):")
    line("[B] ref2015 only (OG selector)", rB)
    line("[A] ref2015 + BSA + interface", rA)
    line("    diffusion top (pose_0)", pose0)
    line("    random pose (avg)", rand)
    line("    ORACLE (best-of-set, ceiling)", oracle)

    win = (rA < rB - 0.1).sum(); loss = (rA > rB + 0.1).sum()
    print(f"\n  A vs B head-to-head: A better on {win}, worse on {loss}, "
          f"tie on {len(cns)-win-loss}  |  mean ΔRMSD = {(rA-rB).mean():+.2f}Å")

    print("\n  5 example structures (per-pose RMSD; * = picked):")
    for h, ib, _ in selB[:5]:
        ia = next(x[1] for x in selA if x[0] == h)
        rmsds = data[h]["rmsd"]
        marks = " ".join(
            f"{r:4.1f}{'B' if i==ib else ''}{'A' if i==ia else ''}".rstrip()
            if (i == ib or i == ia) else f"{r:4.1f}"
            for i, r in enumerate(rmsds))
        print(f"    {h:24s} [{marks}]   B→{rmsds[ib]:.1f}Å  A→{rmsds[ia]:.1f}Å")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    else:
        experiments()


if __name__ == "__main__":
    main()
