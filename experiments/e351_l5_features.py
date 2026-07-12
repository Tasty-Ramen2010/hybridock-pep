"""E351 — PRISM Layer 5 feature extraction over the 1445 ±1 charge-change SKEMPI cases (data/e350).

Fast structural + mutation-type descriptors for a Δ-learning charged-ΔΔG model. SASA is computed ONCE per PDB
(216 structures) and reused across all its mutations, so the whole set extracts in minutes on CPU (no GPU, no MD).
The expensive physics-engine outputs (GB/QM/RISM) are added later as extra feature columns for the subset where
we run them; this file is the cheap, at-scale backbone L5 trains on.

Features per case
  mutation-type: wt_charge, mut_charge, dq, d_volume, d_hydropathy, is_alanine, is_isosteric
  interface 3D: buried_frac (SASA loss on binding), n_contacts, opp_charge_dist (nearest opposite-charge partner
                atom), same_charge_dist, n_aromatic, n_polar_neutral, n_hydrophobic, metal_near, complex_atoms
  label: exp ΔΔG (SKEMPI)

Run: OMP_NUM_THREADS=2 /home/igem/miniconda3/envs/openmm-env/bin/python experiments/e351_l5_features.py
"""
from __future__ import annotations
import sys, json, time
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import fetch

CASES = json.load(open("/home/igem/unknown_software/data/e350_charged_expanded.json"))
OUT = "/home/igem/unknown_software/data/e351_l5_features.jsonl"
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "E": 138.4, "Q": 143.8, "G": 60.1, "H": 153.2,
       "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
       "Y": 193.6, "V": 140.0}
HYD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
       "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
AA3 = {"D": "ASP", "E": "GLU", "K": "LYS", "R": "ARG", "N": "ASN", "Q": "GLN", "H": "HIS"}
CATION_AT = {"ARG": ("NH1", "NH2", "NE"), "LYS": ("NZ",), "HIS": ("ND1", "NE2")}
ANION_AT = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
AROM = {"PHE", "TYR", "TRP", "HIS"}
POLARN = {"SER", "THR", "TYR", "ASN", "GLN"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
METALS = {"MG", "ZN", "CA", "MN", "FE", "NA", "K", "NI", "CU", "CO"}
_struct, _sasa_done = {}, set()
_parser = PDBParser(QUIET=True)
_sr = ShrakeRupley()


def get_struct(pdb):
    if pdb not in _struct:
        st = _parser.get_structure(pdb, fetch(pdb))[0]
        _sr.compute(st, level="A")          # SASA once per PDB (complex context)
        _struct[pdb] = st
    return _struct[pdb]


def features(tag, mut, exp):
    pdb = tag.split("_")[0]; groups = tag.split("_")[1:]
    wt, ch, resid, mt = mut[0], mut[1], int(mut[2:-1]), mut[-1]
    st = get_struct(pdb)
    if ch not in st:
        return None
    res = next((r for r in st[ch] if r.id[1] == resid and r.get_resname().strip() == AA3.get(wt, "")), None)
    if res is None:
        res = next((r for r in st[ch] if r.id[1] == resid), None)
    if res is None:
        return None
    side = [a for a in res if a.name not in ("N", "CA", "C", "O")] or list(res)
    partner = set("".join(groups)) - {ch}
    heavy = [a for c in st for a in c.get_atoms() if a.element.strip() not in ("", "H")]
    ns = NeighborSearch(heavy)
    n_contacts = arom = polarn = hydrophobic = metal = 0
    opp_d = same_d = 99.0
    wt_q = CHARGE.get(wt, 0)
    for a in side:
        for nb in ns.search(a.coord, 6.0):
            r = nb.get_parent(); pc = r.get_parent().id; rn = r.get_resname().strip()
            if pc not in partner:
                continue
            d = float(np.linalg.norm(a.coord - nb.coord))
            if d <= 4.5:
                n_contacts += 1
            # nearest opposite/same-charge partner sidechain N/O relative to the WT charge sign
            is_cat = rn in CATION_AT and nb.name in CATION_AT[rn]
            is_ani = rn in ANION_AT and nb.name in ANION_AT[rn]
            if is_cat or is_ani:
                partner_q = 1 if is_cat else -1
                if wt_q * partner_q < 0:
                    opp_d = min(opp_d, d)
                elif wt_q * partner_q > 0:
                    same_d = min(same_d, d)
            if d <= 5.0 and rn in AROM and nb.name.startswith("C"):
                arom += 1
            if d <= 3.5 and rn in POLARN and nb.name[0] in ("O", "N"):
                polarn += 1
            if d <= 4.5 and rn in HYDROPHOBIC and nb.name.startswith("C"):
                hydrophobic += 1
        for nb in ns.search(a.coord, 8.0):
            if nb.get_parent().get_resname().strip() in METALS:
                metal = 1
    sasa_cplx = sum(a.sasa for a in res)
    # isolated-chain SASA (burial on binding): compute once per (pdb,chain) lazily
    key = (pdb, ch)
    if key not in _sasa_done:
        _iso_cache[key] = _iso_chain_sasa(st, ch)
        _sasa_done.add(key)
    sasa_free = _iso_cache[key].get(resid, sasa_cplx)
    buried = round(float(1.0 - sasa_cplx / sasa_free), 3) if sasa_free > 1 else 0.0
    return {"tag": tag, "mut": mut, "exp": exp,
            "wt_charge": wt_q, "mut_charge": CHARGE.get(mt, 0), "dq": CHARGE.get(mt, 0) - wt_q,
            "d_volume": round(VOL.get(mt, 110) - VOL.get(wt, 110), 1),
            "d_hydropathy": round(HYD.get(mt, 0) - HYD.get(wt, 0), 1),
            "is_alanine": int(mt == "A"),
            "is_isosteric": int((wt, mt) in {("D", "N"), ("E", "Q"), ("N", "D"), ("Q", "E")}),
            "buried_frac": buried, "n_contacts": n_contacts,
            "opp_charge_dist": round(opp_d, 2), "same_charge_dist": round(same_d, 2),
            "n_aromatic": arom, "n_polar_neutral": polarn, "n_hydrophobic": hydrophobic,
            "metal_near": metal, "complex_atoms": sum(1 for _ in st.get_atoms())}


_iso_cache = {}


def _iso_chain_sasa(st, ch):
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    m = Model(0); m.add(st[ch].copy()); s2 = Structure("iso"); s2.add(m)
    _sr.compute(s2, level="A")
    return {r.id[1]: sum(a.sasa for a in r) for r in s2[0][ch]}


def main():
    done = set()
    try:
        for ln in open(OUT):
            d = json.loads(ln); done.add((d["tag"], d["mut"]))
    except FileNotFoundError:
        pass
    print(f"=== E351 L5 features: {len(CASES)} cases, {len(done)} already done ===", flush=True)
    t0 = time.time(); n = 0
    with open(OUT, "a") as fh:
        for tag, mut, exp in CASES:
            if (tag, mut) in done:
                continue
            try:
                f = features(tag, mut, exp)
                if f is not None:
                    fh.write(json.dumps(f) + "\n"); fh.flush(); n += 1
            except Exception as e:
                print(f"  {tag} {mut} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
            if n and n % 50 == 0:
                print(f"  {n} done ({(time.time()-t0)/60:.0f}min, {len(_struct)} PDBs cached)", flush=True)
    print(f"wrote {OUT} (+{n} rows, {(time.time()-t0)/60:.0f}min)")


if __name__ == "__main__":
    main()
