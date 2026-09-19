#!/usr/bin/env python
"""Extract full-atom interface graphs — every atom within 10 A of a peptide-receptor contact.

WHY A GRAPH AT ALL, GIVEN THE ADVICE AGAINST IT. Coventry was explicit (21:40): "you've got 1500
data points, maybe 10,000 with my data. There's not enough information to learn from the graph."
Our own e436 tensor-product model collapsed to a near-constant. Both are good reasons for a low
prior. But two things have changed. Capacity turned out to matter after all -- the same 59
features go from r 0.239 linear to r 0.475 nonlinear -- so the blanket "we are data-limited, keep
it simple" argument is weaker than it looked. And the residual of that nonlinear model has NO
nameable axis left: not length, not isolation, not assay disagreement. If the missing structure
cannot be written as a descriptor, a model that reads atoms is the only way to find out whether
it exists.

WHAT IS EXTRACTED. Contacts first: peptide and receptor heavy atoms within 4.5 A of each other.
Then every atom within 10 A of any of those contact atoms, from both sides. That is Ram's
specification, and it is the right region -- it keeps the second shell that shapes desolvation and
electrostatics while discarding the bulk of the protein that our descriptors already summarise
and that would otherwise dominate the node count.

NODE FEATURES are deliberately primitive -- element, side chain or backbone, which chain, residue
class, and a burial proxy -- because the point is to test whether GEOMETRY carries information our
descriptors miss. Feeding it engineered chemistry would confound that: a win could then be the
features rather than the graph.

CHAIN ASSIGNMENT is the failure mode that would silently poison everything. A PDBbind entry does
not label which chain is the peptide, so the shortest chain within the in-scope length range is
taken, and its length is checked against the sequence we already have for that complex. A
mismatch drops the entry rather than guessing.

Usage: gnn_interface_build.py [--cut 10.0] [--limit N]
Output: data/gnn_graphs.npz
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/igem/unknown_software")
PDBDIR = ROOT / "datasets/pdbbind925"
CORPUS = ROOT / "data/e432/corpus.npz"
OUT = ROOT / "data/gnn_graphs.npz"
ELEMENTS = ["C", "N", "O", "S", "P", "OTHER"]
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
CHARGED = set("DEKR")
POLAR = set("STNQHY")
AROM = set("FWY")
BACKBONE = {"N", "CA", "C", "O"}


def parse(pdb: Path):
    """{chain: [(resname, resseq, [(atomname, element, xyz)])]} — first model, altloc A."""
    ch = defaultdict(list)
    cur = None
    for l in pdb.read_text(errors="ignore").splitlines():
        if l.startswith("ENDMDL"):
            break
        if not l.startswith("ATOM"):
            continue
        if l[16] not in (" ", "A"):
            continue
        rn = l[17:20].strip()
        if rn not in AA3:
            continue
        el = (l[76:78].strip() or l[12:16].strip()[:1]).upper()
        if el == "H":
            continue
        key = (l[21], l[22:27])
        if key != cur:
            ch[l[21]].append((rn, l[22:27], []))
            cur = key
        ch[l[21]][-1][2].append((l[12:16].strip(), el,
                                 (float(l[30:38]), float(l[38:46]), float(l[46:54]))))
    return ch


def node_feats(rn: str, atom: str, el: str, is_pep: float, burial: float) -> list[float]:
    one = AA3.get(rn, "X")
    e = [1.0 if el == x else 0.0 for x in ELEMENTS[:-1]]
    e.append(1.0 if el not in ELEMENTS[:-1] else 0.0)
    return e + [
        is_pep,
        1.0 if atom in BACKBONE else 0.0,
        1.0 if one in CHARGED else 0.0,
        1.0 if one in POLAR else 0.0,
        1.0 if one in AROM else 0.0,
        1.0 if one == "G" else 0.0,
        1.0 if one == "P" else 0.0,
        burial,
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut", type=float, default=10.0)
    ap.add_argument("--contact", type=float, default=4.5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = np.load(CORPUS, allow_pickle=True)
    m = d["has_feats"].astype(bool) & (d["source"] == "pdbbind")
    meta = {str(p).lower()[:4]: (str(s), float(yy), str(gg))
            for p, s, yy, gg in zip(d["pdb"][m], d["seq"][m], d["y"][m], d["groups"][m])}

    files = sorted(PDBDIR.glob("*.pdb"))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} structures on disk, {len(meta)} have labels", flush=True)

    X, E, Y, G, NAMES, skipped = [], [], [], [], [], defaultdict(int)
    for f in files:
        pid = f.stem.lower()[:4]
        if pid not in meta:
            skipped["no label"] += 1
            continue
        seq, y, grp = meta[pid]
        ch = parse(f)
        if len(ch) < 2:
            skipped["<2 chains"] += 1
            continue
        lens = {c: len(v) for c, v in ch.items()}
        # the peptide is the shortest chain in scope, and its length must match what we expect
        cands = [c for c, n in lens.items() if 4 <= n <= 30]
        if not cands:
            skipped["no chain in 4-30"] += 1
            continue
        pc = min(cands, key=lambda c: abs(lens[c] - len(seq)))
        if abs(lens[pc] - len(seq)) > 3:
            skipped["peptide length mismatch"] += 1
            continue
        rec = [c for c in ch if c != pc]
        if not rec:
            skipped["no receptor"] += 1
            continue

        pep_at = [(rn, an, el, xyz) for rn, _, ats in ch[pc] for an, el, xyz in ats]
        rec_at = [(rn, an, el, xyz) for c in rec for rn, _, ats in ch[c] for an, el, xyz in ats]
        if len(pep_at) < 20 or len(rec_at) < 50:
            skipped["too few atoms"] += 1
            continue
        P = np.array([x[3] for x in pep_at])
        R = np.array([x[3] for x in rec_at])
        dm = np.linalg.norm(P[:, None, :] - R[None, :, :], axis=2)
        cp, cr = np.where(dm < a.contact)
        if len(cp) < 5:
            skipped["no contacts"] += 1
            continue
        anchor = np.vstack([P[np.unique(cp)], R[np.unique(cr)]])

        keep_p = np.unique(np.where(
            np.linalg.norm(P[:, None, :] - anchor[None, :, :], axis=2).min(1) < a.cut)[0])
        keep_r = np.unique(np.where(
            np.linalg.norm(R[:, None, :] - anchor[None, :, :], axis=2).min(1) < a.cut)[0])
        xyz = np.vstack([P[keep_p], R[keep_r]])
        if not (30 <= len(xyz) <= 3000):
            skipped["graph size out of range"] += 1
            continue

        nb = (np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2) < 8.0).sum(1) - 1
        bur = nb / max(nb.max(), 1)
        feats = []
        for k, i in enumerate(keep_p):
            rn, an, el, _ = pep_at[i]
            feats.append(node_feats(rn, an, el, 1.0, float(bur[k])))
        for k, i in enumerate(keep_r):
            rn, an, el, _ = rec_at[i]
            feats.append(node_feats(rn, an, el, 0.0, float(bur[len(keep_p) + k])))

        X.append(np.array(feats, dtype=np.float32))
        E.append(xyz.astype(np.float32))
        Y.append(y); G.append(grp); NAMES.append(pid)

    print(f"\nbuilt {len(X)} interface graphs")
    if X:
        ns = [len(x) for x in X]
        print(f"  atoms per graph: median {int(np.median(ns))}, "
              f"range {min(ns)}-{max(ns)}")
        print(f"  node feature dim: {X[0].shape[1]}")
        np.savez_compressed(OUT, X=np.array(X, dtype=object), P=np.array(E, dtype=object),
                            y=np.array(Y), g=np.array(G), names=np.array(NAMES))
        print(f"  -> {OUT.relative_to(ROOT)}")
    print("\n  skipped:")
    for k, v in sorted(skipped.items(), key=lambda t: -t[1]):
        print(f"    {v:>5}  {k}")


if __name__ == "__main__":
    main()
