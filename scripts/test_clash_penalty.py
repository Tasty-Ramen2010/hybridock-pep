#!/usr/bin/env python
"""Unit-test the clash penalty before any training run is spent on it.

A loss term that silently returns zero, or that is not differentiable with respect to the
prediction, costs a full training run to discover and looks exactly like "the idea did not
work". Four checks, on synthetic geometry where the right answer is known by construction:

  1. NO OVERLAP -> exactly zero. A term that fires on well-separated structures would penalise
     every pose and just shrink the translation head, which is the gate-collapse failure again
     under a different name.
  2. OVERLAP -> strictly positive, and larger for deeper interpenetration.
  3. DIFFERENTIABLE w.r.t. tr_pred, with a nonzero gradient -- otherwise it is decoration.
  4. THE GRADIENT PUSHES THE PEPTIDE OUT, not further in. Sign errors here would actively
     train the model to clash, and the loss curve would look fine while doing it.

Usage: test_clash_penalty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party/RAPiDock_finetuned"))


def make_batch(pep_xyz, rec_xyz):
    from torch_geometric.data import HeteroData, Batch
    d = HeteroData()
    d["pep_a"].pos = torch.tensor(pep_xyz, dtype=torch.float)
    d["receptor"].pos = torch.tensor(rec_xyz, dtype=torch.float)
    return Batch.from_data_list([d])


def main() -> None:
    from train_lastlayer import _clash_penalty
    dev = "cpu"

    # a compact receptor blob at the origin, on a 3 A lattice
    rec = [[x * 3.0, y * 3.0, z * 3.0]
           for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)]
    pep = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]]

    print(f"receptor {len(rec)} residues, peptide {len(pep)} atoms, cutoff 3.5 A\n")
    ok = True

    # ---- 1. far away -> zero
    far = make_batch([[p[0] + 60.0, p[1], p[2]] for p in pep], rec)
    tr = torch.zeros(1, 3, requires_grad=True)
    sig = torch.ones(1)
    v = _clash_penalty(far, tr, sig, dev)
    print(f"  1. peptide 60 A away          penalty = {float(v):.6f}   "
          f"{'PASS' if float(v) == 0.0 else 'FAIL — fires on separated structures'}")
    ok &= float(v) == 0.0

    # ---- 2. overlapping -> positive, and monotone in depth
    vals = []
    for shift in (10.0, 4.0, 0.0):
        b = make_batch([[p[0] + shift, p[1], p[2]] for p in pep], rec)
        tr = torch.zeros(1, 3, requires_grad=True)
        vals.append(float(_clash_penalty(b, tr, torch.ones(1), dev)))
    print(f"  2. shifted +10 / +4 / 0 A     penalty = {vals[0]:.4f} / {vals[1]:.4f} / "
          f"{vals[2]:.4f}")
    mono = vals[0] <= vals[1] <= vals[2] and vals[2] > 0
    print(f"     {'PASS — deeper overlap costs more' if mono else 'FAIL — not monotone in depth'}")
    ok &= mono

    # ---- 3 & 4. gradient exists and points OUTWARD.
    # ONE receptor residue and ONE peptide atom overlapping it. The first version of this test
    # used a 3 A lattice slab, and that geometry cannot answer the question: with a 3.5 A cutoff
    # every direction out of the slab lands on the next plane of receptor points, so the local
    # gradient tracks lattice structure rather than "away from the protein" and the direction
    # check failed on a test artefact. A single pair has exactly one correct answer -- directly
    # along +x -- so a sign error has nowhere to hide.
    one_rec = [[0.0, 0.0, 0.0]]
    one_pep = [[1.2, 0.0, 0.0]]             # 1.2 A apart, well inside the 3.5 A cutoff
    b = make_batch(one_pep, one_rec)
    tr = torch.zeros(1, 3, requires_grad=True)
    sig = torch.full((1,), 2.0)             # move = sigma^2 * tr_pred = 4 * tr_pred
    v = _clash_penalty(b, tr, sig, dev)
    v.backward()
    g = tr.grad[0]
    print(f"\n  (one receptor residue at the origin, one peptide atom 1.2 A away; escape is +x)")
    print(f"  3. d(penalty)/d(tr_pred)      = [{g[0]:+.4f}, {g[1]:+.4f}, {g[2]:+.4f}]")
    nonzero = float(g.norm()) > 1e-4
    print(f"     {'PASS — differentiable, gradient is real' if nonzero else 'FAIL — no usable gradient reaches tr_pred'}")
    ok &= nonzero
    # descending must move the peptide along +x, away from the occupied half-space
    escapes = float(-g[0]) > 0
    print(f"     descent direction in x is {float(-g[0]):+.4f}  "
          f"{'PASS — points away from the receptor' if escapes else 'FAIL — points INTO the receptor'}")
    ok &= escapes

    step = -g / max(float(g.norm()), 1e-8)
    moved = make_batch([[q[0] + float(step[0]) * 1.0,
                         q[1] + float(step[1]) * 1.0,
                         q[2] + float(step[2]) * 1.0] for q in one_pep], one_rec)
    after = float(_clash_penalty(moved, torch.zeros(1, 3), torch.ones(1), dev))
    before = float(v)
    print(f"\n  4. penalty before {before:.4f} -> after one descent step {after:.4f}")
    outward = after < before
    print(f"     {'PASS — the gradient pushes the peptide OUT' if outward else 'FAIL — gradient drives it further IN'}")
    ok &= outward

    print("\n  => " + ("all four checks pass; the term is safe to train with."
                       if ok else "DO NOT TRAIN WITH THIS TERM until the failures above are fixed."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
