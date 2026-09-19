#!/usr/bin/env python
"""Thread every peptide onto every receptor in a block and extract PAIR features.

EVERY FEATURE HERE IS A PROPERTY OF THE PAIR, NOT OF EITHER PARTNER.  That is the whole design
constraint, and it is what our production dG model violates.  A feature like "peptide
hydrophobicity" or "peptide net charge" is constant down a row of the grid, so it can encode
how good a peptide is in general but says nothing about which receptor it belongs on.  Those
features are why our model moves only 0.22 kcal/mol across an entire row.

Concretely the features fall into four families, all of which the archive says carry
within-pair signal:

  ref2015 interface terms, per term.  E(complex) - E(receptor) - E(peptide) for each scoring
    term separately.  The total already ranks better than our dG model on Coventry; splitting
    it lets a learner reweight the terms.  There is a specific reason to expect that helps:
    fa_atr alone separated cognates better than the total did, because fa_rep and fa_sol both
    scale with burial and cancel it.  A learned weighting can keep the attraction and discount
    the two that merely track how buried the peptide is.

  typed contact chemistry.  Contacts split by the chemistry of BOTH partners -- apolar-apolar,
    polar-polar, opposite-charge, like-charge, aromatic.  Chemistry-typed contacts were the
    reason hydrophobic complementarity worked where chemistry-blind BSA did not.

  complementarity scores.  Charge and hydrophobic complementarity between the peptide and the
    pocket it is sitting in.  charge_complementarity.py exists precisely for this regime and
    reaches r=0.755 within-pocket on SKEMPI charge-changing mutations.

  burial geometry.  How the peptide's burial is distributed along its length, not just how much
    of it there is.  Total burial is a size proxy; its shape is a fit proxy.

The threading protocol is the one from coventry_threading_test.py, including the soft-repulsive
pre-pack: threading a bulky sequence onto a backbone shaped for a small one creates clashes a
single hard pass cannot escape, and the resulting 10^4 REU swamps every comparison.  Genuine
steric incompatibility survives the soft pass; recoverable rotamer conflicts do not.  Every
sequence in a block gets identical freedom, so the comparison stays controlled.

Usage: sel_features.py [workers] [--blocks data/sel_blocks.json] [--limit N]
Output: logs/sel_features.jsonl  (one row per peptide x receptor pair)
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
BLOCKS = ROOT / os.environ.get("SEL_BLOCKS", "data/sel_blocks.json")
OUT = ROOT / os.environ.get("SEL_OUT", "logs/sel_features.jsonl")
INTERFACE_CUT = 8.0
CONTACT_CUT = 4.5

#: ref2015 terms worth decomposing. Intra-residue terms (fa_dun, p_aa_pp, rama) very nearly
#: cancel in the interface difference and are kept only as a sanity channel.
TERMS = ("fa_atr", "fa_rep", "fa_sol", "fa_elec", "lk_ball_wtd", "fa_intra_rep",
         "hbond_bb_sc", "hbond_sc", "hbond_sr_bb", "hbond_lr_bb", "fa_dun", "p_aa_pp")

APOLAR = set("AVLIMFWPCG")
POLAR = set("STNQYH")
POS = set("KR")
NEG = set("DE")
AROMATIC = set("FWYH")
#: Kyte-Doolittle, for hydrophobic complementarity
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")


def _terms_of(pose) -> dict:
    """Weighted per-term energies of a pose."""
    from pyrosetta.rosetta.core.scoring import score_type_from_name
    _SF(pose)
    e = pose.energies().total_energies()
    out = {}
    for t in TERMS:
        try:
            st = score_type_from_name(t)
            out[t] = float(e[st] * _SF.get_weight(st))
        except Exception:  # noqa: BLE001 -- a term absent from this weight set is not fatal
            out[t] = 0.0
    return out


def _geometry(pose, n_rec: int) -> dict:
    """Typed contact chemistry, complementarity and burial shape, from the refined pose."""
    import numpy as np
    from scipy.spatial import cKDTree

    rec_xyz, rec_aa, pep_xyz, pep_aa, pep_res = [], [], [], [], []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        aa = AA3to1.get(res.name3().strip(), "X")
        for a in range(1, res.natoms() + 1):
            if res.atom_type(a).element().strip() == "H":
                continue
            v = res.xyz(a)
            if i <= n_rec:
                rec_xyz.append((v.x, v.y, v.z)); rec_aa.append(aa)
            else:
                pep_xyz.append((v.x, v.y, v.z)); pep_aa.append(aa); pep_res.append(i - n_rec)
    if not rec_xyz or not pep_xyz:
        return {}
    R = np.asarray(rec_xyz); P = np.asarray(pep_xyz)
    tree = cKDTree(R)
    hits = tree.query_ball_point(P, CONTACT_CUT)

    counts = dict(c_apolar=0, c_polar=0, c_opp_charge=0, c_like_charge=0,
                  c_aromatic=0, c_mixed=0, n_contact=0)
    per_res = {}
    rec_touch = set()
    for k, hs in enumerate(hits):
        pa = pep_aa[k]
        per_res.setdefault(pep_res[k], 0)
        for j in hs:
            ra = rec_aa[j]
            rec_touch.add(j)
            counts["n_contact"] += 1
            per_res[pep_res[k]] += 1
            if pa in APOLAR and ra in APOLAR:
                counts["c_apolar"] += 1
            elif pa in POLAR and ra in POLAR:
                counts["c_polar"] += 1
            elif (pa in POS and ra in NEG) or (pa in NEG and ra in POS):
                counts["c_opp_charge"] += 1
            elif (pa in POS and ra in POS) or (pa in NEG and ra in NEG):
                counts["c_like_charge"] += 1
            elif pa in AROMATIC and ra in AROMATIC:
                counts["c_aromatic"] += 1
            else:
                counts["c_mixed"] += 1

    n_pep_res = max(pep_res) if pep_res else 1
    tot = max(counts["n_contact"], 1)
    out = {f"{k}_frac": v / tot for k, v in counts.items() if k != "n_contact"}
    out["n_contact_per_res"] = counts["n_contact"] / n_pep_res
    out["rec_atoms_touched_per_res"] = len(rec_touch) / n_pep_res

    # burial SHAPE along the peptide: total burial is a size proxy, its distribution is a fit
    # proxy -- a peptide that fits a groove is buried evenly, one perched on a rim is not.
    b = np.array([per_res.get(i, 0) for i in range(1, n_pep_res + 1)], dtype=float)
    out["burial_mean"] = float(b.mean())
    out["burial_sd"] = float(b.std())
    out["burial_cv"] = float(b.std() / b.mean()) if b.mean() > 0 else 0.0
    out["burial_max"] = float(b.max())
    out["frac_res_contacting"] = float((b > 0).mean())

    # complementarity: peptide chemistry vs the chemistry of the pocket it actually touches
    pocket_aa = [rec_aa[j] for j in rec_touch]
    pep_seq_aa = [pep_aa[k] for k in range(len(pep_aa))]
    if pocket_aa and pep_seq_aa:
        pk = float(np.mean([KD.get(a, 0.0) for a in pocket_aa]))
        pp = float(np.mean([KD.get(a, 0.0) for a in pep_seq_aa]))
        out["hydro_pocket"] = pk
        out["hydro_product"] = pk * pp          # matched hydrophobicity (the complementarity)
        out["hydro_absdiff"] = abs(pk - pp)     # mismatch
        qp = sum(1 for a in pep_seq_aa if a in POS) - sum(1 for a in pep_seq_aa if a in NEG)
        qr = sum(1 for a in pocket_aa if a in POS) - sum(1 for a in pocket_aa if a in NEG)
        out["charge_product"] = -float(qp * qr) / max(n_pep_res, 1)   # opposite = favourable
        out["pocket_charge_per_res"] = float(qr) / max(len(set(rec_touch)), 1)
    return out


def do_pair(args: tuple) -> dict:
    """Thread peptide `seq` onto receptor/backbone `j` and return every pair feature."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
    from pyrosetta.toolbox import mutate_residue

    block, i, j, seq, receptor, backbone, pep_pdb_i, pdb_i, pdb_j = args
    row = {"block": block, "pep_idx": i, "rec_idx": j, "cognate": int(i == j),
           "pep_pdb": pdb_i, "rec_pdb": pdb_j, "seq": seq}
    t0 = time.time()
    try:
        rec = _PR.pose_from_pdb(receptor)
        pep = _PR.pose_from_pdb(backbone)
        if pep.total_residue() != len(seq):
            row["error"] = f"len {pep.total_residue()} != {len(seq)}"
            return row
        info = pep.pdb_info()
        for k in range(1, pep.total_residue() + 1):
            info.chain(k, "p")
        n_rec = rec.total_residue()
        pose = rec.clone()
        pose.append_pose_by_jump(pep, n_rec)
        for k, aa in enumerate(seq, start=1):
            mutate_residue(pose, n_rec + k, aa, pack_radius=6.0)

        pep_sel = rs.ChainSelector("p")
        near = rs.NeighborhoodResidueSelector(pep_sel, INTERFACE_CUT, False)
        movable = rs.OrResidueSelector(pep_sel, near)
        frozen = rs.NotResidueSelector(movable)
        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())
        tf.push_back(operation.OperateOnResidueSubset(operation.PreventRepackingRLT(), frozen))

        mm = _PR.rosetta.core.kinematics.MoveMap()
        mm.set_bb(False); mm.set_chi(False); mm.set_jump(True)
        sub = movable.apply(pose)
        for k in range(1, pose.total_residue() + 1):
            if sub[k]:
                mm.set_chi(k, True)
                if k > n_rec:
                    mm.set_bb(k, True)

        soft = _PR.create_score_function("ref2015_soft")
        PackRotamersMover(soft, tf.create_task_and_apply_taskoperations(pose)).apply(pose)
        m0 = MinMover(mm, soft, "lbfgs_armijo_nonmonotone", 0.01, True)
        m0.max_iter(100); m0.apply(pose)
        PackRotamersMover(_SF, tf.create_task_and_apply_taskoperations(pose)).apply(pose)
        mn = MinMover(mm, _SF, "lbfgs_armijo_nonmonotone", 0.01, True)
        mn.max_iter(300); mn.apply(pose)

        split = list(pose.split_by_chain())
        pep_out = split[-1]
        rec_only = split[0].clone()
        for c in split[1:-1]:
            rec_only.append_pose_by_jump(c, rec_only.total_residue())
        tc, tr, tp = _terms_of(pose), _terms_of(rec_only), _terms_of(pep_out)
        for t in TERMS:
            row[f"i_{t}"] = tc[t] - tr[t] - tp[t]
        row["i_total"] = float(_SF(pose) - _SF(rec_only) - _SF(pep_out))
        row.update(_geometry(pose, n_rec))
        row["seconds"] = round(time.time() - t0, 1)
    except Exception as exc:  # noqa: BLE001 -- one bad pair must not stop the block
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    blocks = json.loads(BLOCKS.read_text())

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["block"], r["pep_idx"], r["rec_idx"]))

    jobs = []
    for b in blocks:
        m = b["members"]
        for i, pi in enumerate(m):
            for j, pj in enumerate(m):
                if (b["block_id"], i, j) in done:
                    continue
                jobs.append((b["block_id"], i, j, pi["seq"], pj["receptor"],
                             pj["peptide_pdb"], pi["peptide_pdb"], pi["pdb"], pj["pdb"]))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} pairs to thread ({len(done)} done), {workers} workers", flush=True)
    if not jobs:
        return

    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, row in enumerate(ex.map(do_pair, jobs, chunksize=2), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if k % 50 == 0 or k == 1:
                rate = k / max(time.time() - t0, 1)
                if "error" in row:
                    status = "ERR " + row["error"][:50]
                else:
                    status = f"i_total {row.get('i_total', float('nan')):8.1f}"
                print(f"  [{k}/{len(jobs)}] block {row['block']:2d} "
                      f"{row['pep_pdb']}->{row['rec_pdb']}  {status}  "
                      f"{rate:.2f}/s  eta {(len(jobs) - k) / max(rate, 1e-6) / 60:.0f} min",
                      flush=True)
    print("SEL_FEATURES_DONE")


if __name__ == "__main__":
    main()
