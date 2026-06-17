"""E261 — assemble the per-receptor peptide-Kd anchor library (Phase-1 benchmark + deployment library).

Groups the 925 PDBbind peptide-Kd complexes by RECEPTOR (exact receptor-sequence identity — the safe key:
identical sequence => identical offset b(R)). A receptor with >=2 DISTINCT peptides is "anchorable":
at inference, score a query peptide on receptor R by anchoring to the other known-Kd peptides on R.

Output:
  data/e261_anchor_library.json  — {receptor_id: {seq_len, n_members, n_distinct_pep, members:[...]}}
  console: distribution of anchorable receptors (>=2 / >=3 / >=5 distinct peptides).
"""
from __future__ import annotations
import json, glob, os, hashlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
         "SEC": "U", "PYL": "O"}


def receptor_seq(pdb_path: str) -> str:
    """One-letter receptor sequence (all protein chains, CA atoms, chain+resseq ordered)."""
    seen = set()
    per_chain: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(pdb_path) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            if ln[12:16].strip() != "CA":
                continue
            res = ln[17:20].strip()
            if res not in _3to1:
                continue
            ch = ln[21]
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            key = (ch, resseq, ln[26])  # include insertion code
            if key in seen:
                continue
            seen.add(key)
            per_chain[ch].append((resseq, _3to1[res]))
    chains = []
    for ch in sorted(per_chain):
        chains.append("".join(a for _, a in sorted(per_chain[ch])))
    return "/".join(chains)


def main() -> None:
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/pdbbind_peptides.jsonl"))]
    idx = {os.path.basename(p).split("_")[0].lower(): p
           for p in glob.glob(os.path.join(ROOT, "data/drive_pull/pl/P-L/**/*_protein.pdb"),
                              recursive=True)}
    by_receptor: dict[str, list[dict]] = defaultdict(list)
    n_noseq = 0
    for r in rows:
        p = idx.get(r["pdb"].lower())
        if not p:
            continue
        rseq = receptor_seq(p)
        if len(rseq.replace("/", "")) < 20:   # too short to be a real receptor
            n_noseq += 1
            continue
        rid = hashlib.md5(rseq.encode()).hexdigest()[:10]
        by_receptor[rid].append(
            {"pdb": r["pdb"], "pep": r["seq"], "y": r["y"], "atype": r["affinity_type"],
             "rec_len": len(rseq.replace("/", ""))})

    library = {}
    for rid, mem in by_receptor.items():
        distinct = len({m["pep"] for m in mem})
        library[rid] = {"seq_len": mem[0]["rec_len"], "n_members": len(mem),
                        "n_distinct_pep": distinct, "members": mem}
    json.dump(library, open(os.path.join(ROOT, "data/e261_anchor_library.json"), "w"), indent=1)

    n_rec = len(library)
    ge = lambda k: sum(1 for v in library.values() if v["n_distinct_pep"] >= k)
    pept_in = lambda k: sum(v["n_members"] for v in library.values() if v["n_distinct_pep"] >= k)
    print(f"925 PDBbind peptide complexes -> {n_rec} distinct receptors (exact-seq grouping)")
    print(f"  receptors with >=2 distinct peptides (ANCHORABLE): {ge(2):4d}  "
          f"({pept_in(2)} complexes)")
    print(f"  receptors with >=3 distinct peptides:              {ge(3):4d}  "
          f"({pept_in(3)} complexes)")
    print(f"  receptors with >=5 distinct peptides:              {ge(5):4d}  "
          f"({pept_in(5)} complexes)")
    print(f"  singletons (orphan receptors, fall back to absolute): {ge(1) - ge(2)}")
    # show the biggest panels (the best Phase-1 / deployment anchors)
    top = sorted(library.items(), key=lambda kv: kv[1]["n_distinct_pep"], reverse=True)[:10]
    print("\n  largest same-receptor panels (rec_id  len  #distinct_pep  #complexes  Kd-span):")
    for rid, v in top:
        ys = [m["y"] for m in v["members"]]
        print(f"    {rid}  {v['seq_len']:4d}  {v['n_distinct_pep']:3d}  {v['n_members']:3d}  "
              f"ΔG[{min(ys):.1f},{max(ys):.1f}]")
    print(f"\n  saved data/e261_anchor_library.json  ({n_noseq} skipped, no receptor seq)")


if __name__ == "__main__":
    main()
