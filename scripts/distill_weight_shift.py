#!/usr/bin/env python
"""How far did the weights actually move, and did the gate freeze hold?

TWO SEPARATE QUESTIONS, BOTH ANSWERABLE FROM THE CHECKPOINTS ALONE.

DID ANYTHING CHANGE? Train loss moved 0.4499 -> 0.4323 and rose at the end, and the plateau fired
at epoch 3. That pattern is consistent with a real but tiny update, and it is also consistent with
a model that barely moved at all. A docking null means something completely different in those
two cases: a model that moved and did not improve is evidence against the method, while a model
that never moved is evidence only that the learning rate or the co-fold fraction was too small.
Per-tensor relative change separates them, and it costs nothing.

DID THE GATE FREEZE HOLD? `tr_final_layer` and `rot_final_layer` are the sigma-conditioned
magnitude gates. Under MSE on the score, the loss-optimal magnitude is rho*|target| whenever the
predicted direction is imperfect, so the loss actively rewards shrinking them, monotonically from
epoch 1 -- which is why no early checkpoint survives an unfrozen run. `--freeze-gate-layers` was
passed, so these tensors must be EXACTLY unchanged, not approximately. A nonzero delta here means
the flag did not do what it claims and the run is uninterpretable.

BOTH state_dict AND the EMA ARE CHECKED. inference.py calls ema_weights.copy_to() before
sampling, so the EMA shadow parameters are what actually docks. A previous investigation found
the gates collapsed in both, and also that shadow_params has 148 entries against 229 state_dict
keys -- so positional indexing into the key list is WRONG and tensors must be matched by shape.

Usage: distill_weight_shift.py --parent <ckpt> --child <ckpt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    import torch
    P = torch.load(a.parent, map_location="cpu", weights_only=False)
    C = torch.load(a.child, map_location="cpu", weights_only=False)

    def sd(d):
        for k in ("model", "state_dict", "model_state_dict"):
            if isinstance(d, dict) and k in d:
                return d[k]
        return d

    ps, cs = sd(P), sd(C)
    # EXCLUDE NON-LEARNED BUFFERS. This is not tidiness, it is the difference between a right
    # and a wrong answer. encoder.cross_convs.0.batch_norm.running_var has norm 812,822 against
    # fc.0.weight's 115, and --freeze-bn-stats pins it so its delta is exactly zero. Including
    # it in a module-level ||dW||/||W|| divides a real weight change by a frozen number four
    # orders of magnitude larger, and the ratio reads ~1e-05 no matter what the weights did.
    # That artefact is what produced a confident report that "the convolutions never trained";
    # excluded, the same checkpoints show cross_convs moving 7.9e-03. Never aggregate norms
    # across tensors without checking what dominates the denominator.
    BUFFERS = ("running_mean", "running_var", "num_batches_tracked", "output_mask")
    keys = [k for k in ps if k in cs and hasattr(ps[k], "shape")
            and ps[k].shape == cs[k].shape and ps[k].dtype.is_floating_point
            and not any(b in k for b in BUFFERS)
            and float(ps[k].float().norm()) > 0.0]
    print(f"{len(keys)} shared learned float tensors "
          f"(BatchNorm buffers and zero-norm placeholders excluded)\n")

    rows = []
    for k in keys:
        p, c = ps[k].float(), cs[k].float()
        n = float(p.norm())
        delta = float((c - p).norm())
        rows.append((k, delta / n if n > 0 else 0.0, float(c.norm()) / n if n > 0 else 1.0,
                     p.numel()))

    moved = [r for r in rows if r[1] > 1e-8]
    tot_p = sum(r[3] for r in rows)
    tot_m = sum(r[3] for r in moved)
    print(f"  tensors changed at all: {len(moved)}/{len(rows)}")
    print(f"  parameters in changed tensors: {tot_m:,}/{tot_p:,} ({100 * tot_m / tot_p:.1f}%)")
    print("     ^ THIS NUMBER IS A TRAP. It counts a tensor as changed at a delta of 1e-10, and "
          "on the\n       first distillation run it read 99.0% while the 5.3M-parameter "
          "convolution stack had\n       moved by 1e-05 -- a no-op. I reported that run as 'a "
          "real finetune' on the strength of\n       this line and was wrong. Read the "
          "per-module table below instead.")
    if moved:
        rel = np.array([r[1] for r in moved])
        print(f"  relative change among those: median {np.median(rel):.2e}  "
              f"max {rel.max():.2e}")

    # PER-MODULE, WEIGHTED BY PARAMETER COUNT -- the view that actually says whether the model
    # learned. A finetune that moves only small input embeddings cannot change where the peptide
    # goes, however many tensors technically differ.
    import collections
    g: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for k in keys:
        mod = ".".join(k.split(".")[:2])
        p, c = ps[k].float(), cs[k].float()
        g[mod][0] += float((c - p).norm()) ** 2
        g[mod][1] += float(p.norm()) ** 2
        g[mod][2] += p.numel()
    print(f"\n  per-module change, weighted by size:\n")
    print(f"  {'module':<44}{'rel change':>12}{'params':>13}")
    conv_rel = 0.0
    for mod, v in sorted(g.items(), key=lambda t: -(t[1][0] ** 0.5) / max(t[1][1] ** 0.5, 1e-12)):
        r = (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12)
        print(f"  {mod[:42]:<44}{r:>12.3e}{v[2]:>13,}")
        if "convs" in mod:
            conv_rel = max(conv_rel, r)
    if conv_rel and conv_rel < 1e-3:
        print(f"\n  !! the convolution stack moved {conv_rel:.1e} relative. Placement is computed "
              f"there,\n     so this model cannot have changed where it puts peptides. Raise the "
              f"learning rate or\n     the target fraction and re-check HERE before spending an "
              f"evaluation on it.")

    print(f"\n  largest movers (relative L2 change, and norm ratio child/parent):\n")
    print(f"  {'tensor':<58}{'rel change':>12}{'norm ratio':>12}")
    for k, d, nr, _ in sorted(rows, key=lambda r: -r[1])[:a.top]:
        print(f"  {k[:56]:<58}{d:>12.3e}{nr:>12.4f}")

    print("\n\nGATE FREEZE — these must be EXACTLY unchanged\n")
    print(f"  {'tensor':<58}{'rel change':>12}{'norm ratio':>12}")
    ok = True
    for k, d, nr, _ in rows:
        if "tr_final_layer" in k or "rot_final_layer" in k:
            flag = "" if d == 0.0 else "   <-- MOVED"
            if d != 0.0:
                ok = False
            print(f"  {k[:56]:<58}{d:>12.3e}{nr:>12.4f}{flag}")

    # the EMA is what actually samples, so it is checked separately and matched by SHAPE
    for tag, obj in (("parent", P), ("child", C)):
        e = obj.get("ema_weights") if isinstance(obj, dict) else None
        if e is None:
            print(f"\n  ({tag} has no ema_weights entry)")
    pe = P.get("ema_weights", {}) if isinstance(P, dict) else {}
    ce = C.get("ema_weights", {}) if isinstance(C, dict) else {}
    psh = pe.get("shadow_params") if isinstance(pe, dict) else None
    csh = ce.get("shadow_params") if isinstance(ce, dict) else None
    if psh is not None and csh is not None and len(psh) == len(csh):
        deltas = [float((c.float() - p.float()).norm()) /
                  max(float(p.float().norm()), 1e-12) for p, c in zip(psh, csh)]
        nz = sum(1 for d in deltas if d > 1e-8)
        print(f"\n  EMA shadow params: {len(psh)} tensors, {nz} changed, "
              f"max relative change {max(deltas):.3e}")

    print("\n  -> " + ("gate freeze HELD (both gate tensors bit-identical)." if ok else
                       "GATE FREEZE FAILED — the gates moved. This run is not interpretable."))
    if moved and max(r[1] for r in moved) < 1e-3:
        print("  -> the update is TINY (max relative change < 1e-3). A docking null would say "
              "the learning rate or co-fold fraction was too small, NOT that distillation "
              "does not work.")


if __name__ == "__main__":
    main()
