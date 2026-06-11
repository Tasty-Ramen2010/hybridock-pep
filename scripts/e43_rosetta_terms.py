"""E43 — dissect FlexPepDock: per-TERM ref2015 interface energies vs experimental ΔG.

What truly drives their gears? Score the relaxed complex/receptor/peptide and decompose the
INTERFACE energy per ScoreType (complex − rec − pep). Correlate EACH term with ΔG on crystal-65
and the-98. Reveals: which ref2015 terms carry universal signal, and — critically — whether
fa_elec + fa_sol (Rosetta's properly-implemented electrostatics + Lazaridis-Karplus desolvation,
on a RELAXED pose) succeed on the charged complexes where our crude geometric term failed.

Uses crystal poses (Ram: try crystal first). FastRelax on cropped pocket = the FlexPepDock move
(optimize salt-bridge geometry / rotamers before scoring).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

KEY_TERMS = ["fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec",
             "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc", "fa_dun", "ref"]


def init():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false "
                   "-load_PDB_components false", silent=True)
    return pyrosetta


def crop(pr, pep_pdb, rec_pdb, rad=10.0):
    from Bio.PDB import PDBParser, PDBIO, Select
    P = PDBParser(QUIET=True)
    pm = P.get_structure("p", str(pep_pdb))[0]
    rm = P.get_structure("r", str(rec_pdb))[0]
    pxyz = np.array([a.coord for r in pm.get_residues() if r.id[0] == " "
                     for a in r if a.element != "H"])
    keep = set()
    for ch in rm:
        for res in ch:
            if res.id[0] != " ":
                continue
            for a in res:
                if a.element != "H" and np.min(((pxyz - a.coord) ** 2).sum(1)) <= rad * rad:
                    keep.add((ch.id, res.id)); break

    class S(Select):
        def accept_residue(self, r):
            return (r.get_parent().id, r.id) in keep
    out = Path(f"/tmp/_e43crop_{Path(pep_pdb).stem}.pdb")
    io = PDBIO(); io.set_structure(P.get_structure("r2", str(rec_pdb))); io.save(str(out), S())
    return out


def per_term_interface(pr, pep_pdb, rec_pdb, relax=True):
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    sf = pyrosetta.get_fa_scorefxn()
    # merged complex P + R
    rc = crop(pr, pep_pdb, rec_pdb)
    merged = Path(f"/tmp/_e43m_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "A"), (rc, "B")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    merged.write_text("\n".join(lines) + "\nEND\n")
    pose = pyrosetta.pose_from_pdb(str(merged))
    if relax:
        from pyrosetta.rosetta.protocols.relax import FastRelax
        fr = FastRelax(sf, 1); fr.apply(pose)

    def emap(p):
        sf(p)
        e = p.energies().total_energies()
        return {t: float(e[getattr(ScoreType, t)]) for t in KEY_TERMS}
    # complex
    ec = emap(pose)
    # split chains: receptor-only (chain B) and peptide-only (chain A)
    from pyrosetta.rosetta.protocols.grafting import return_region  # noqa
    # simpler: use pose chain split
    chains = pose.split_by_chain()
    # identify peptide (smaller) vs receptor (larger)
    parts = sorted(chains, key=lambda c: c.total_residue())
    pep_pose, rec_pose = parts[0], parts[-1]
    ep = emap(pep_pose); er = emap(rec_pose)
    return {t: ec[t] - er[t] - ep[t] for t in KEY_TERMS}


def build(which, relax=True):
    pr = init()
    out_path = Path(f"/tmp/e43_{which}.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if which == "cr":
        # Read crystal peptide/pocket PDBs directly from the benchmark (survives /tmp wipes).
        bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
        items = [(b["pdb"].upper(), str((ROOT / b["peptide_pdb"]).resolve()),
                  str((ROOT / b["pocket_pdb"]).resolve()), b["dg_exp"])
                 for b in bench
                 if (ROOT / b["peptide_pdb"]).exists() and (ROOT / b["pocket_pdb"]).exists()]
    else:
        e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
        work = Path("/tmp/ppep_work")
        items = [(k, str(work / f"{k}_pep.pdb"), str(work / f"{k}_rec.pdb"), r["y"])
                 for k, r in e28.items() if (work / f"{k}_pep.pdb").exists()]
    for key, pep, rec, y in items:
        if key in out or not pep:
            continue
        try:
            terms = per_term_interface(pr, pep, rec, relax=relax)
            out[key] = dict(terms, y=y)
            out_path.write_text(json.dumps(out))
            if len(out) % 10 == 0:
                print(f"  {which} {len(out)} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {key} FAIL {type(e).__name__}: {str(e)[:50]}", flush=True)
    return out


def evaluate():
    from scipy.stats import pearsonr
    cr = json.loads(Path("/tmp/e43_cr.json").read_text())
    bp = Path("/tmp/e43_b98.json")
    b98 = json.loads(bp.read_text()) if bp.exists() else {}
    ycr = np.array([cr[k]["y"] for k in cr])
    print(f"\n=== ref2015 PER-TERM interface energy vs ΔG (crystal-65 n={len(cr)}"
          + (f", 98 n={len(b98)}" if b98 else "") + ") ===")
    print(f"  {'term':<14}{'crystal-65':>12}" + (f"{'the-98':>10}{'universal':>11}" if b98 else ""))
    for t in KEY_TERMS:
        vc = np.array([cr[k][t] for k in cr])
        rc = pearsonr(vc, ycr).statistic if vc.std() > 0 else 0.0
        line = f"  {t:<14}{rc:>+12.3f}"
        if b98 and len(b98) >= 5:
            y9 = np.array([b98[k]["y"] for k in b98]); v9 = np.array([b98[k][t] for k in b98])
            r9 = pearsonr(v9, y9).statistic if v9.std() > 0 else 0.0
            u = "YES" if rc * r9 > 0 and min(abs(rc), abs(r9)) > 0.1 else ""
            line += f"{r9:>+10.3f}{u:>11}"
        print(line)
    # full ref2015 total (sum) LOO
    def loo(rows, feats):
        y = np.array([r["y"] for r in rows]); X = np.array([[r[f] for f in feats] for r in rows]); p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return pearsonr(p, y).statistic
    rows = list(cr.values())
    print(f"\n  crystal-65 LOO: all 11 terms r={loo(rows,KEY_TERMS):+.3f} | "
          f"fa_elec+fa_sol r={loo(rows,['fa_elec','fa_sol']):+.3f} | "
          f"hbonds r={loo(rows,['hbond_sr_bb','hbond_lr_bb','hbond_bb_sc','hbond_sc']):+.3f}")
    print("  >> which terms are FlexPepDock's real engine? does fa_elec+fa_sol drive it?")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cr"
    if which == "eval":
        evaluate()
    else:
        build(which, relax="--norelax" not in sys.argv)
        evaluate()
