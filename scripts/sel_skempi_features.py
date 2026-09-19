#!/usr/bin/env python
"""Structure-based ddG features: repack the mutant interface and difference it against wild-type.

WHAT IS BEING TESTED.  The shipped ddG head (scoring/mutation_ddg.py) is structure-free -- two
residue identities plus a five-category location label -- and reaches r = 0.385 leave-complex-out.
The June assessment named the unexecuted next step outright: "CRUDE proxy (5 categories); full
structure ddG (dock WT-vs-mut, OUR pipeline) is the lever to beat 0.38."  This runs that lever
using the machinery the selectivity work already built.

WHY IT MIGHT BEAT A PROPERTY MODEL WHERE OUR ABSOLUTE SCORER COULD NOT.  Congeneric blindness is
real and general: across a one-residue change the median feature of the absolute scorer moves 14%
of its dataset-wide sd, and on 749 near-twin pairs we track the true difference at r = 0.182 while
moving only 35% as far as reality.  But those were complex-level, peptide-dominated descriptors.
PER-TERM INTERFACE energies computed on a repacked mutant are not blind to a substitution by
construction -- swap a lysine for a glutamate and i_fa_elec moves directly, delete a donor and the
hbond terms move directly.  Whether that converts into out-of-fold skill is exactly what this
measures, and it may not: importance is not skill, as the SKEMPI block already taught us once
(0.069 importance, zero added skill).

PROTOCOL.  Wild-type is repacked and scored ONCE per complex; each mutant is the same structure
with one residue mutated, repacked under the identical protocol (soft-repulsive pre-pass, then
hard repack and minimise). The feature vector is the DIFFERENCE, mutant minus wild-type, so every
complex-level constant cancels -- the same main-effect logic that the selectivity model relies on.

Label: ddG = RT ln(Kd_mut / Kd_wt), positive = the mutation WEAKENS binding.

Usage: sel_skempi_features.py [workers]
Output: logs/sel_skempi_features.jsonl
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/igem/unknown_software")
sys.path.insert(0, str(ROOT / "scripts"))
SRC = ROOT / "data/skempi_single_peptide.csv"
PDBS = ROOT / "datasets/skempi"
OUT = ROOT / "logs/sel_skempi_features.jsonl"
R_GAS = 0.0019872041          # kcal/mol/K

_PR = None
_SF = None


def _init() -> None:
    global _PR, _SF
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "scripts"))
    from coventry_refine import init_rosetta
    _PR = init_rosetta()
    _SF = _PR.create_score_function("ref2015")
    import sel_features
    sel_features._PR = _PR
    sel_features._SF = _SF


def ddg(row: dict) -> float | None:
    """RT ln(Kd_mut/Kd_wt); positive = mutation weakens binding."""
    try:
        km, kw = float(row["affinity_mut"]), float(row["affinity_wt"])
        t = float(row["temperature"] or 298)
    except (TypeError, ValueError):
        return None
    if km <= 0 or kw <= 0 or not (250 <= t <= 350):
        return None
    return R_GAS * t * math.log(km / kw)


def _build(pdb: str, small: str, big: str):
    """Two-chain pose: big chains first (receptor), then the small partner as chain 'p'."""
    import tempfile
    src = (PDBS / f"{pdb}.pdb").read_text(errors="ignore").splitlines()
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000") as td:
        rp, pp = Path(td) / "rec.pdb", Path(td) / "pep.pdb"
        rp.write_text("\n".join(l for l in src if l.startswith("ATOM") and l[21] in big)
                      + "\nEND\n")
        pp.write_text("\n".join(l for l in src if l.startswith("ATOM") and l[21] in small)
                      + "\nEND\n")
        rec = _PR.pose_from_pdb(str(rp))
        pep = _PR.pose_from_pdb(str(pp))
    return rec, pep


def _score(pose, n_rec: int) -> dict:
    """Repack + minimise the interface, then the full pair-feature vector."""
    from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
    from pyrosetta.rosetta.core.select import residue_selector as rs
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover
    import sel_features as SF

    pep_sel = rs.ChainSelector("p")
    near = rs.NeighborhoodResidueSelector(pep_sel, SF.INTERFACE_CUT, False)
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
    tc, tr, tp = SF._terms_of(pose), SF._terms_of(rec_only), SF._terms_of(pep_out)
    out = {f"i_{t}": tc[t] - tr[t] - tp[t] for t in SF.TERMS}
    out["i_total"] = float(_SF(pose) - _SF(rec_only) - _SF(pep_out))
    out.update(SF._geometry(pose, n_rec))
    return out


def do_complex(args: tuple) -> list[dict]:
    """Score wild-type once, then every mutation of this complex against it."""
    from pyrosetta.toolbox import mutate_residue
    pdb, small, big, muts = args
    results = []
    try:
        rec, pep = _build(pdb, small, big)
        n_rec = rec.total_residue()
        info = pep.pdb_info()
        for k in range(1, pep.total_residue() + 1):
            info.chain(k, "p")
        base = rec.clone()
        base.append_pose_by_jump(pep, n_rec)
        # keep a map from (original chain, resnum) to pose index BEFORE the chain rename
        wt_feats = _score(base.clone(), n_rec)
    except Exception as exc:  # noqa: BLE001
        return [{"pdb": pdb, "error": f"wt {type(exc).__name__}: {exc}"}]

    for m in muts:
        row = {"pdb": pdb, "mutation": m["mutation"], "location": m["location"],
               "wt_aa": m["wt_aa"], "mut_aa": m["mut_aa"], "ddg": m["ddg"],
               "on_small": int(m["mut_chain"] in small)}
        t0 = time.time()
        try:
            pose = rec.clone()
            p2 = pep.clone()
            pose.append_pose_by_jump(p2, n_rec)
            # Locate the mutated residue by scanning pdb_info directly. pdb2pose is unreliable
            # here because the small partner was renamed to chain 'p' before the two poses were
            # joined, so neither the original chain id nor 'p' resolves consistently. Matching on
            # the PDB residue NUMBER plus the wild-type identity is unambiguous in practice, and
            # the identity check below is the guard against a wrong hit.
            info = pose.pdb_info()
            want = m["resnum"].strip()
            num = int("".join(c for c in want if c.isdigit() or c == "-"))
            icode = want[-1] if want and want[-1].isalpha() else " "
            cands = []
            for k in range(1, pose.total_residue() + 1):
                if info.number(k) != num:
                    continue
                if icode != " " and info.icode(k).strip() != icode.strip():
                    continue
                cands.append(k)
            # prefer a candidate whose residue identity matches the SKEMPI wild type
            idx = next((k for k in cands if pose.residue(k).name1() == m["wt_aa"]), 0)
            if not idx:
                row["error"] = (f"residue not found (num {num}{icode.strip()}, "
                                f"{len(cands)} number matches, none is {m['wt_aa']})")
                results.append(row); continue
            obs = pose.residue(idx).name1()
            if obs != m["wt_aa"]:
                row["error"] = f"wt mismatch: pose has {obs}, SKEMPI says {m['wt_aa']}"
                results.append(row); continue
            mutate_residue(pose, idx, m["mut_aa"], pack_radius=6.0)
            mt = _score(pose, n_rec)
            for k, v in mt.items():
                row[f"d_{k}"] = v - wt_feats.get(k, 0.0)
            row["mut_i_total"] = mt["i_total"]
            row["wt_i_total"] = wt_feats["i_total"]
            row["seconds"] = round(time.time() - t0, 1)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return results


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["pdb"], r.get("mutation")))

    by_pdb = defaultdict(list)
    n_nolabel = 0
    for r in csv.DictReader(open(SRC)):
        d = ddg(r)
        if d is None:
            n_nolabel += 1
            continue
        if (r["pdb"], r["mutation"]) in done:
            continue
        r["ddg"] = d
        by_pdb[(r["pdb"], r["small_chains"], r["big_chains"])].append(r)
    jobs = [(p, s, b, m) for (p, s, b), m in by_pdb.items()]
    n_mut = sum(len(j[3]) for j in jobs)
    print(f"{n_mut} mutations over {len(jobs)} complexes "
          f"({len(done)} done, {n_nolabel} without a usable ddG), {workers} workers", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with OUT.open("a") as fh, ProcessPoolExecutor(workers, initializer=_init) as ex:
        n = 0
        for rows in ex.map(do_complex, jobs, chunksize=1):
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                n += 1
            fh.flush()
            rate = n / max(time.time() - t0, 1)
            print(f"  {n}/{n_mut}  {rows[0]['pdb']}  "
                  f"{sum(1 for r in rows if 'error' in r)} errors  "
                  f"eta {(n_mut - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
    print("SEL_SKEMPI_DONE")


if __name__ == "__main__":
    main()
