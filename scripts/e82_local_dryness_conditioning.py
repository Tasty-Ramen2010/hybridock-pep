"""E82 — local per-charge dryness conditioning, with PENALTY/REWARD separated (Ram's wet/dry function).

Prior conditioning (E81) used the receptor's GLOBAL pocket hydrophobicity -> flipped. Ram: use the LOCAL
dryness each charge sits in. And the key physical insight that the net-charge features all miss:

  a buried charge UNPAIRED in a DRY local patch  -> desolvation paid for nothing -> ALWAYS weakens (penalty)
  a buried charge PAIRED  in a DRY local patch  -> salt bridge in low dielectric -> ALWAYS strengthens (reward)

The NET of these flips across datasets (their balance differs), but EACH HALF should be sign-stable by
physics. So we DECOMPOSE, not sum. Per peptide charged group, from the static structure:
  dsasa        = burial (free - complex SASA)
  local_dry    = receptor hydrophobic heavy atoms / (hyd + polar) within 6 Å of the charged atom  (0..1)
  paired       = opposite-sign receptor charged atom within 4.5 Å
Features (intensive, /L), each tested for SIGN-STABILITY across charged-cr65 AND charged-the98:
  desolv_penalty   = Σ [buried]·[unpaired]·local_dry            (expect +corr, weakens)   <- the penalty
  saltbridge_reward= Σ [buried]·[paired]·local_dry              (expect -corr, strengthens)<- the reward
  + many variants (thresholds, dryness weighting on/off, Ram's hyd-contact × charge-gradient).
If penalty AND reward are EACH sign-stable, a model with BOTH beats the flipping net -> real charged lever.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from Bio.PDB.Structure import Structure  # noqa: E402
from Bio.PDB.Model import Model  # noqa: E402

P = PDBParser(QUIET=True)
SR = ShrakeRupley()
CACHE = Path("/tmp/e82_local_dry.json")
POS3, NEG3 = {"LYS", "ARG", "HIS"}, {"ASP", "GLU"}
CHG_ATOMS = {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"], "HIS": ["ND1", "NE2"],
             "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}
NONPOLAR_EL = {"C", "S"}
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
      "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
      "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def sign(rn):
    return 1 if rn in POS3 else (-1 if rn in NEG3 else 0)


def featurize(pep_pdb, rec_pdb, y, ds, net_charge):
    pep = P.get_structure("p", str(pep_pdb))[0]
    rec = P.get_structure("r", str(rec_pdb))[0]
    SR.compute(pep, level="R")
    free = {(r.get_parent().id, r.id[1]): r.sasa for r in pep.get_residues() if r.id[0] == " "}
    cx = Structure("c"); m = Model(0); cx.add(m); used = set(); pep_cids = set()
    for tag, src in [("p", pep), ("r", rec)]:
        for ch in src.get_chains():
            cid = ch.id
            while cid in used:
                cid = chr((ord(cid) + 1) % 90 + 33)
            used.add(cid); c2 = ch.copy(); c2.id = cid; m.add(c2)
            if tag == "p":
                pep_cids.add(cid)
    SR.compute(cx, level="R")
    comp = {}
    for ch in cx.get_chains():
        if ch.id in pep_cids:
            for r in ch.get_residues():
                if r.id[0] == " ":
                    comp[(r.resname.upper(), r.id[1])] = r.sasa
    # receptor heavy atoms split nonpolar/polar, + charged atoms with sign
    rec_np, rec_pol, rec_chg = [], [], []
    for ch in rec.get_chains():
        for r in ch.get_residues():
            rn = r.resname.upper(); sgn = sign(rn)
            for a in r:
                if a.element == "H":
                    continue
                (rec_np if a.element in NONPOLAR_EL else rec_pol).append(a.coord.astype(float))
                if sgn and a.name in CHG_ATOMS.get(rn, []):
                    rec_chg.append((a.coord.astype(float), sgn))
    tnp = cKDTree(np.array(rec_np)) if rec_np else None
    tpol = cKDTree(np.array(rec_pol)) if rec_pol else None

    pep_res = [r for r in pep.get_residues() if r.id[0] == " "]
    seq = "".join(A3.get(r.resname.upper(), "X") for r in pep_res)
    L = max(1, len(seq))
    # accumulate decomposed terms over charged groups
    acc = {k: 0.0 for k in ["desolv_penalty", "sb_reward", "desolv_penalty_nb", "sb_reward_nb",
                            "penalty_count", "reward_count", "mean_local_dry_chg",
                            "buried_unpaired", "buried_paired", "dry_buried_all"]}
    nchg = 0
    for r in pep_res:
        rn = r.resname.upper(); sgn = sign(rn)
        if not sgn:
            continue
        nchg += 1
        key = (r.get_parent().id, r.id[1])
        fs = free.get(key, 0.0); cs = comp.get((rn, r.id[1]), fs)
        dsasa = max(0.0, fs - cs)
        catoms = [a.coord.astype(float) for a in r if a.name in CHG_ATOMS.get(rn, [])]
        if not catoms:
            continue
        ca = catoms[0]
        nnp = sum(len(tnp.query_ball_point(c, 6.0)) for c in catoms) if tnp else 0
        npol = sum(len(tpol.query_ball_point(c, 6.0)) for c in catoms) if tpol else 0
        local_dry = nnp / (nnp + npol) if (nnp + npol) else 0.0
        paired = any(rs == -sgn and np.linalg.norm(c - rc) < 4.5
                     for c in catoms for rc, rs in rec_chg)
        buried = dsasa > 30
        acc["mean_local_dry_chg"] += local_dry
        if buried:
            acc["dry_buried_all"] += local_dry
            if paired:
                acc["sb_reward"] += local_dry          # reward amplified by dryness
                acc["sb_reward_nb"] += 1.0
                acc["buried_paired"] += 1
                acc["reward_count"] += 1
            else:
                acc["desolv_penalty"] += local_dry      # penalty amplified by dryness
                acc["desolv_penalty_nb"] += 1.0
                acc["buried_unpaired"] += 1
                acc["penalty_count"] += 1
    f = dict(ds=ds, y=y, seq=seq, net_charge=net_charge)
    for k, v in acc.items():
        f[k] = v / L
    f["mean_local_dry_chg"] = acc["mean_local_dry_chg"] / max(1, nchg)
    f["net_sb_balance"] = (acc["sb_reward"] - acc["desolv_penalty"]) / L   # the NET (expected to flip)
    # Ram's hydrophobic-contact × charge-gradient: total receptor-hyd contact near ALL charges × |netQ|/L
    f["hydcontact_x_chggrad"] = acc["dry_buried_all"] * abs(net_charge) / L
    return f


def build():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    e78 = json.load(open("/tmp/e78_dewet.json"))
    e49 = json.load(open("/tmp/e49b_the98.json"))
    work = Path("/tmp/ppep_work")
    for k, v in e49.items():
        kk = "98_" + k
        if kk not in e78 or abs(e78[kk]["net_charge"]) < 2:
            continue
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            try:
                out[kk] = featurize(pep, rec, v["y"], "the98", e78[kk]["net_charge"])
            except Exception as e:  # noqa: BLE001
                print(f"  98 {k} {str(e)[:40]}")
    bench = json.loads((ROOT / "data/benchmark_crystal.json").read_text())
    for r in bench:
        kk = "cr_" + r["pdb"]
        if kk not in e78 or abs(e78[kk]["net_charge"]) < 2:
            continue
        try:
            out[kk] = featurize(r["peptide_pdb"], r["pocket_pdb"], r["dg_exp"], "cr65",
                                e78[kk]["net_charge"])
        except Exception as e:  # noqa: BLE001
            print(f"  cr {r['pdb']} {str(e)[:40]}")
    CACHE.write_text(json.dumps(out))
    return out


def main():
    feats = build()
    rows = list(feats.values())
    c = [r for r in rows if r["ds"] == "cr65"]; n = [r for r in rows if r["ds"] == "the98"]
    FEATS = [k for k in rows[0] if k not in ("ds", "y", "seq", "net_charge")]
    print(f"=== E82 local-dryness penalty/reward decomposition. cr65={len(c)} the98={len(n)} ===")

    def pr(rs, f):
        x = np.array([r[f] for r in rs], float); y = np.array([r["y"] for r in rs])
        mk = ~np.isnan(x)
        return pearsonr(x[mk], y[mk])[0] if mk.sum() > 4 and np.std(x[mk]) > 0 else np.nan

    print("\nSIGN-STABILITY across BOTH charged datasets (penalty expect +, reward expect -):")
    print(f"{'feature':<22}{'cr65':>9}{'the98':>9}  verdict")
    survivors = []
    for f in FEATS:
        rc, rn = pr(c, f), pr(n, f)
        stable = (not np.isnan(rc) and not np.isnan(rn) and rc * rn > 0)
        strong = stable and min(abs(rc), abs(rn)) > 0.2
        v = "STABLE-STRONG <==" if strong else ("stable" if stable else "flip")
        if strong:
            survivors.append(f)
        print(f"  {f:<20}{rc:>+9.3f}{rn:>+9.3f}  {v}")

    print(f"\nstable-strong survivors: {survivors or 'NONE'}")
    if survivors:
        # leave-dataset-out: do the separated terms beat the flipping net?
        def ldo(tr, te, cols):
            X = np.array([[r[c] for c in cols] for r in tr], float); y = np.array([r["y"] for r in tr])
            ok = ~np.isnan(X).any(1); X, y = X[ok], y[ok]
            mu, sd = X.mean(0), X.std(0) + 1e-9
            A = np.column_stack([np.ones(len(X)), (X - mu) / sd]); R = np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ y)
            Xe = np.array([[r[c] for c in cols] for r in te], float); oke = ~np.isnan(Xe).any(1)
            return pearsonr(np.column_stack([np.ones(oke.sum()), (Xe[oke] - mu) / sd]) @ w,
                            np.array([r["y"] for r in te])[oke])[0]
        print(f"\nleave-dataset-out charged (separated terms vs the flipping net):")
        print(f"  {'model':<34}{'the98->cr65':>13}{'cr65->the98':>13}")
        for nm, cols in [("net_sb_balance (the SUM)", ["net_sb_balance"]),
                         ("survivors (separated)", survivors),
                         ("desolv_penalty + sb_reward", ["desolv_penalty", "sb_reward"])]:
            try:
                print(f"  {nm:<34}{ldo(n, c, cols):>+13.3f}{ldo(c, n, cols):>+13.3f}")
            except Exception:  # noqa: BLE001
                pass
        print("\n  >> if separated terms transfer POSITIVE where net_sb_balance flips, the decomposition works.")


if __name__ == "__main__":
    main()
