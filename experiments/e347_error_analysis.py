"""E347 — where does the charged tier lack? Correlate physical + 3D-structural features with the SIGNED error.

Reads the E345 campaign checkpoint and, for each completed case, extracts structural/physical descriptors of the
mutated residue's interface environment, then correlates each descriptor with the SIGNED error (calc − exp) of
every engine (explicit / ecc / gb / qm). Signed (not abs) so we see the DIRECTION: which environments make each
method over- vs under-estimate. Runs incrementally (whatever's checkpointed); re-run when the campaign completes.

Features
  physical:   exp_ddg, is_glu (Asp=0/Glu=1)
  3D interface: buried_frac (SASA loss on binding), n_contacts (partner heavy atoms ≤4.5Å), cation_dist (to nearest
                partner Arg/Lys/His N — salt-bridge proximity), has_saltbridge, n_aromatic (partner F/Y/W/H ≤5Å),
                n_polar_neutral (partner S/T/Y/N/Q O/N ≤3.5Å — the buried-H-bond signal), metal_near, complex_atoms

Run: /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e347_error_analysis.py
"""
from __future__ import annotations
import sys, json
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch

JSON = "/home/igem/unknown_software/data/e345_charged_campaign.json"
ACID = {"D": ("ASP", ("OD1", "OD2")), "E": ("GLU", ("OE1", "OE2"))}
CATION = {"ARG": ("NH1", "NH2", "NE"), "LYS": ("NZ",), "HIS": ("ND1", "NE2")}
AROM = {"PHE", "TYR", "TRP", "HIS"}
POLARN = {"SER", "THR", "TYR", "ASN", "GLN"}
METALS = {"MG", "ZN", "CA", "MN", "FE", "NA", "K", "NI", "CU", "CO"}
_cache = {}


def _struct(pdb):
    if pdb not in _cache:
        _cache[pdb] = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))[0]
    return _cache[pdb]


def features(tag, mut):
    pdb = tag.split("_")[0]; groups = tag.split("_")[1:]
    wt, ch, resid = mut[0], mut[1], int(mut[2:-1])
    st = _struct(pdb)
    res = next((r for r in st[ch] if r.id[1] == resid and r.get_resname().strip() == ACID[wt][0]), None)
    if res is None:
        return None
    tip = [res[n] for n in ACID[wt][1] if n in res]
    partner = set("".join(groups)) - {ch}
    heavy = [a for c in st for a in c.get_atoms() if a.element.strip() not in ("", "H")]
    ns = NeighborSearch(heavy)

    n_contacts = cation_dist = 0; cation_dist = 99.0
    arom, polarn, metal = set(), 0, 0
    for a in tip:
        for nb in ns.search(a.coord, 5.0):
            r = nb.get_parent(); pc = r.get_parent().id; rn = r.get_resname().strip()
            if pc in partner:
                d = float(np.linalg.norm(a.coord - nb.coord))
                if d <= 4.5:
                    n_contacts += 1
                if rn in CATION and nb.name in CATION[rn]:
                    cation_dist = min(cation_dist, d)
                if rn in AROM and nb.name.startswith("C"):
                    arom.add((pc, r.id[1]))
                if rn in POLARN and nb.name[0] in ("O", "N") and d <= 3.5:
                    polarn += 1
        for nb in ns.search(a.coord, 8.0):
            if nb.get_parent().get_resname().strip() in METALS:
                metal = 1
    # SASA burial: mutated-residue SASA in complex vs isolated own-chain
    sr = ShrakeRupley()
    sr.compute(st, level="A")
    sasa_cplx = sum(a.sasa for a in res)
    # isolated chain SASA
    ownchain = st[ch].copy()
    from Bio.PDB.Structure import Structure as _S
    from Bio.PDB.Model import Model as _M
    m = _M(0); m.add(ownchain); s2 = _S("iso"); s2.add(m)
    sr.compute(s2, level="A")
    res_iso = next((r for r in s2[0][ch] if r.id[1] == resid), None)
    sasa_free = sum(a.sasa for a in res_iso) if res_iso else sasa_cplx
    buried = 1.0 - (sasa_cplx / sasa_free) if sasa_free > 1 else 0.0
    return {"exp_ddg": None, "is_glu": 1 if wt == "E" else 0, "buried_frac": round(buried, 2),
            "n_contacts": n_contacts, "cation_dist": round(cation_dist, 2),
            "has_saltbridge": 1 if cation_dist < 4.0 else 0, "n_aromatic": len(arom),
            "n_polar_neutral": polarn, "metal_near": metal,
            "complex_atoms": sum(1 for _ in st.get_atoms())}


def main():
    d = json.load(open(JSON))
    rows = d if isinstance(d, list) else d.get("results", [])
    rows = [r for r in rows if any(r.get(m) is not None for m in ("explicit", "ecc", "gb", "qm"))]
    print(f"=== E347 error analysis on {len(rows)} completed cases ===\n", flush=True)
    feats, table = [], []
    for r in rows:
        f = features(r["tag"], r["mut"])
        if f is None:
            continue
        f["exp_ddg"] = r["exp"]
        errs = {m: (r[m] - r["exp"]) if r.get(m) is not None else None for m in ("explicit", "ecc", "gb", "qm")}
        feats.append((f, errs))
        table.append((r["tag"], r["mut"], r["exp"], errs, f))
    # per-case signed-error table
    print(f"{'case':16s} {'exp':>5s} | {'expl':>6s} {'ecc':>6s} {'gb':>6s} {'qm':>6s} | bur nc satD arom pol M")
    for tag, mut, exp, errs, f in table:
        e = lambda k: f"{errs[k]:+6.1f}" if errs[k] is not None else "   -- "
        print(f"{tag[:15]+' '+mut[:0]:16s} {exp:+5.2f} | {e('explicit')} {e('ecc')} {e('gb')} {e('qm')} | "
              f"{f['buried_frac']:.1f} {f['n_contacts']:2d} {f['cation_dist']:4.1f} {f['n_aromatic']:2d} "
              f"{f['n_polar_neutral']:2d} {f['metal_near']}")
    # correlations: each feature vs signed error, per method
    if len(feats) >= 5:
        from scipy.stats import pearsonr
        fkeys = ["exp_ddg", "is_glu", "buried_frac", "n_contacts", "cation_dist", "has_saltbridge",
                 "n_aromatic", "n_polar_neutral", "metal_near", "complex_atoms"]
        print("\n=== corr(feature, SIGNED error)  [+ = feature drives OVER-estimation] ===")
        print(f"{'feature':16s} " + " ".join(f"{m:>8s}" for m in ("explicit", "ecc", "gb", "qm")))
        for fk in fkeys:
            line = f"{fk:16s} "
            for m in ("explicit", "ecc", "gb", "qm"):
                xs = [ff[fk] for ff, ee in feats if ee[m] is not None]
                ys = [ee[m] for ff, ee in feats if ee[m] is not None]
                if len(set(xs)) > 1 and len(xs) >= 5:
                    line += f"{pearsonr(xs, ys)[0]:+8.2f} "
                else:
                    line += f"{'--':>8s} "
            print(line)
    else:
        print(f"\n(only {len(feats)} cases — correlations need ≥5; re-run when the campaign advances)")


if __name__ == "__main__":
    main()
