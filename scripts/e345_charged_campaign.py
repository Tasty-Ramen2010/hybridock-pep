"""E345 — overnight charged-FEP validation campaign: r / MAE / RMSE for all 3 methods on 22 clean SKEMPI cases.

Locks the charged-term formula and validates it at scale. 22 unique isosteric charge-neutralizing interface
mutations (D→N / E→Q, single, |ΔΔG_exp| 0.3–4.0 kcal, 18 PDBs) — the clean, well-posed benchmark everyone else
uses, not the cherry-picked extremes. For each case runs three engines and compares to experiment:
  1. explicit-frozen  — charge-morph TI in explicit TIP3P/PME (baseline; E341)
  2. ECC-explicit     — same + charges ×0.75 (electronic-continuum screening; E343)   <-- the --ultra-charged engine
  3. GB-implicit      — charge-morph TI in GBn2 continuum solvent (E344)

Each case is also given a CONFIDENCE gate from structure: a charge-neutralizing mutation is HIGH-confidence when
the WT carboxylate has a cationic salt-bridge partner (Arg/Lys/His) within 4.5 Å across the interface (ECC handles
these), and LOW-confidence when the interface partners are polar-neutral only (Tyr-OH/Ser/Thr/backbone → the 1IAR
buried-H-bond class our electrostatic corrections can't fix) or a metal sits within 8 Å (setup-sensitive).

Reports Pearson r, MAE, RMSE per method over ALL cases and over the HIGH-confidence subset.

Run: OMP_NUM_THREADS=1 /home/igem/miniconda3/envs/openmm-env/bin/python scripts/e345_charged_campaign.py
"""
from __future__ import annotations
import sys, time, json, tempfile
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
sys.path.insert(0, "/home/igem/unknown_software/scripts")
from e334_skempi_validation import build, deriv_curve, fetch
from e343_ecc_scaling import apply_ecc
from e332_g1_charged_corrected import rocklin_correction
import e344_implicit_charged as gbmod
from openmm import unit

MORPHS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]   # 9 windows (charge-change needs denser λ)
OUT = "/home/igem/unknown_software/data/e345_charged_campaign.json"
_trap = getattr(np, "trapezoid", None) or np.trapz

# (tag, mutation, exp ΔΔG) — 22 clean isosteric charge-neutralizing cases from the SKEMPI scan
CASES = [
    ("1AO7_ABC_DE", "DD93N", 2.46), ("1BD2_ABC_DE", "DD30N", 1.94), ("1BRS_A_D", "EA71Q", 1.45),
    ("1CHO_EFG_I", "EI16Q", 1.32), ("1E96_A_B", "DA38N", 2.16),   ("1IAR_A_B", "EA9Q", 3.11),
    ("1K8R_A_B", "DA38N", 1.97),   ("1KTZ_A_B", "DB8N", 2.00),    ("1KTZ_A_B", "EB95Q", 1.63),
    ("1MAH_A_F", "DA71N", 1.88),   ("1NMB_N_LH", "DH57N", 2.95),  ("1PPF_E_I", "EI19Q", 0.65),
    ("1R0R_E_I", "EI14Q", 1.26),   ("2JEL_LH_P", "EP5Q", 0.71),   ("2PCB_A_B", "EA32Q", 0.61),
    ("2PCB_A_B", "EA35Q", 0.68),   ("2PCB_A_B", "DA34N", 0.82),   ("3HFM_HL_Y", "DY101N", 1.49),
    ("4NKQ_C_AB", "DB176N", 1.80), ("4NM8_ABCDEF_HL", "DB19N", 0.66), ("4RS1_A_B", "EA88Q", 0.86),
    ("4RS1_A_B", "DA92N", 1.33),
]
ACID_SIDE = {"D": ("OD1", "OD2"), "E": ("OE1", "OE2")}
CATION = {"ARG": ("NH1", "NH2", "NE"), "LYS": ("NZ",), "HIS": ("ND1", "NE2")}
METALS = {"MG", "ZN", "CA", "MN", "FE", "NA", "K", "NI", "CU", "CO"}


def confidence(tag, mut):
    """Structure-based gate: cationic salt-bridge partner across interface (HIGH) vs buried-polar/metal (LOW)."""
    pdb = tag.split("_")[0]; groups = tag.split("_")[1:]
    wt, ch, resid = mut[0], mut[1], int(mut[2:-1])
    st = PDBParser(QUIET=True).get_structure(pdb, fetch(pdb))[0]
    if ch not in st:
        return {"ok": False, "reason": "chain-missing", "high": False}
    try:
        res = next(r for r in st[ch] if r.id[1] == resid and r.get_resname()[0] == ("A" if wt == "?" else wt))
    except StopIteration:
        try:
            res = next(r for r in st[ch] if r.id[1] == resid)
        except StopIteration:
            return {"ok": False, "reason": "resid-missing", "high": False}
    carbox = [a for a in res if a.name in ACID_SIDE.get(wt, ())]
    if not carbox:
        return {"ok": False, "reason": "no-carboxylate", "high": False}
    other_atoms = [a for c in st for a in c.get_atoms() if c.id != ch]
    ns = NeighborSearch(other_atoms + [a for a in st[ch].get_atoms()])
    cation = None; metal = None
    for a in carbox:
        for nb in ns.search(a.coord, 4.5):
            rn = nb.get_parent().get_resname().strip()
            if rn in CATION and nb.name in CATION[rn] and nb.get_parent().get_parent().id != ch:
                cation = (nb.get_parent().get_parent().id, rn, nb.get_parent().id[1])
        for nb in ns.search(a.coord, 8.0):
            if nb.get_parent().get_resname().strip() in METALS:
                metal = nb.get_parent().get_resname().strip()
    high = cation is not None and metal is None
    return {"ok": True, "high": high, "cation": cation, "metal": metal,
            "reason": ("salt-bridge:" + str(cation)) if high else ("metal:" + str(metal) if metal else "no-cation-partner")}


def explicit_pair(tag, mut, ecc):
    sysb, modb, ab, dQ = build(tag, mut, "bound")
    if ecc:
        apply_ecc(sysb, 0.75)
    db, Lb = deriv_curve(sysb, modb, MORPHS, 1500, 60, 100)
    sysf, modf, af, _ = build(tag, mut, "free")
    if ecc:
        apply_ecc(sysf, 0.75)
    df, Lf = deriv_curve(sysf, modf, MORPHS, 1500, 90, 100)
    bnd = np.array([v[0] for v in db]); fre = np.array([v[0] for v in df])
    dQs = dQ * (0.75 if ecc else 1.0)
    return float(_trap(bnd - fre, MORPHS)) + (rocklin_correction(dQs, Lb) - rocklin_correction(dQs, Lf))


def gb_case(tag, mut):
    sb = gbmod.build_implicit(tag, mut, "bound", False)
    db = gbmod.deriv_curve(*sb, MORPHS, 1500, 60, 100, False)
    sf = gbmod.build_implicit(tag, mut, "free", False)
    df = gbmod.deriv_curve(*sf, MORPHS, 1500, 90, 100, False)
    return float(_trap(np.array(db) - np.array(df), MORPHS))


def metrics(pairs):
    """pairs = list of (calc, exp). Returns r, MAE, RMSE, n."""
    pairs = [(c, e) for c, e in pairs if c is not None and np.isfinite(c)]
    if len(pairs) < 3:
        return {"n": len(pairs), "r": None, "MAE": None, "RMSE": None}
    c = np.array([p[0] for p in pairs]); e = np.array([p[1] for p in pairs])
    from scipy.stats import pearsonr
    return {"n": len(pairs), "r": round(float(pearsonr(c, e)[0]), 3),
            "MAE": round(float(np.mean(np.abs(c - e))), 3),
            "RMSE": round(float(np.sqrt(np.mean((c - e) ** 2))), 3)}


def main():
    print(f"=== E345 charged-FEP campaign: {len(CASES)} cases × 3 engines, 9 λ-windows ===", flush=True)
    results = []
    t0 = time.time()
    for i, (tag, mut, exp) in enumerate(CASES):
        rec = {"tag": tag, "mut": mut, "exp": exp}
        try:
            rec["conf"] = confidence(tag, mut)
        except Exception as e:
            rec["conf"] = {"ok": False, "reason": f"conf-err:{str(e)[:40]}", "high": False}
        for name, fn in (("explicit", lambda: explicit_pair(tag, mut, False)),
                         ("ecc", lambda: explicit_pair(tag, mut, True)),
                         ("gb", lambda: gb_case(tag, mut))):
            t = time.time()
            try:
                rec[name] = round(fn(), 3)
                print(f"[{i+1:2d}/{len(CASES)}] {tag:15s} {mut:8s} {name:8s} calc={rec[name]:+.2f} "
                      f"exp={exp:+.2f} ({(time.time()-t)/60:.1f}m) conf={'HI' if rec['conf'].get('high') else 'lo'}",
                      flush=True)
            except Exception as e:
                rec[name] = None
                print(f"[{i+1:2d}/{len(CASES)}] {tag:15s} {mut:8s} {name:8s} FAIL {type(e).__name__}:{str(e)[:60]}",
                      flush=True)
        results.append(rec)
        json.dump(results, open(OUT, "w"), indent=1)     # checkpoint after every case
    # metrics
    print(f"\n=== METRICS (wall {(time.time()-t0)/60:.0f} min) ===")
    summary = {}
    for name in ("explicit", "ecc", "gb"):
        allp = [(r.get(name), r["exp"]) for r in results]
        hip = [(r.get(name), r["exp"]) for r in results if r["conf"].get("high")]
        summary[name] = {"all": metrics(allp), "high_conf": metrics(hip)}
        a, h = summary[name]["all"], summary[name]["high_conf"]
        print(f"{name:9s} ALL  n={a['n']:2d} r={a['r']}  MAE={a['MAE']}  RMSE={a['RMSE']}   |  "
              f"HIGH-CONF n={h['n']:2d} r={h['r']}  MAE={h['MAE']}  RMSE={h['RMSE']}")
    json.dump({"results": results, "summary": summary}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print("The --ultra-charged engine is ECC-explicit; report its HIGH-CONF row as the validated charged tier.")


if __name__ == "__main__":
    main()
