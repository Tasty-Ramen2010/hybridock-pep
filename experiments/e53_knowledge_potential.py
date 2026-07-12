"""E53 — knowledge-based interaction potentials mined from 18,522 peppc structures (Ram's ideas).

Tests, in parallel, whether structure-mined contact statistics bridge the charged data gap:
  H1 (400 AA-pairs): directed 20x20 log-odds peptide↔receptor contact potential (the '400 AA
     interactions'). log-odds = log(N_obs(a,b)/N_exp(a,b)); N_exp from marginal freqs.
  H2 (TERMINAL-debiased): same, but peptide residues only at chain ends (pos 1,2,-2,-1) — Ram's
     insight that termini are peptide-like (less tertiary bias) -> cleaner free-peptide-like stats.
  H3 (CHARGED atom-pair): carboxylate(D/E)/amine(K)/guanidinium(R)/imidazole(H) group-pair propensity
     — the charge-specific potential residue-MJ may miss.
  H4: score crystal-65 + the-98 with each, vs Kd, CHARGED-stratified, vs existing MJ.
  H5: does a mined potential ADD to current features on the charged subset (the floor)?
KBPs encode favorability (propensity), still static/pairwise — honest test of whether charge/terminal
specificity beats residue-MJ. Mines a sample of peppc; tests on our labelled Kd sets.
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

P = PDBParser(QUIET=True)
A3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
AAS = "ACDEFGHIKLMNPQRSTVWY"
N_MINE = 4000
CUT = 6.0


def cb(res):
    for n in ("CB", "CA"):
        if n in res:
            return res[n].coord
    return None


def interface_pairs(pep_pdb, rec_pdb, terminal_only=False):
    """Return list of (pep_aa, rec_aa, pep_pos_frac) for peptide↔receptor Cβ contacts < CUT."""
    try:
        pm = P.get_structure("p", str(pep_pdb))[0]
        rm = P.get_structure("r", str(rec_pdb))[0]
    except Exception:
        return []
    pep = [r for r in pm.get_residues() if r.id[0] == " " and r.resname.upper() in A3]
    rec = [r for r in rm.get_residues() if r.id[0] == " " and r.resname.upper() in A3]
    if len(pep) < 2 or not rec:
        return []
    rec_cb = [(cb(r), A3[r.resname.upper()]) for r in rec if cb(r) is not None]
    if not rec_cb:
        return []
    coords = np.array([c for c, _ in rec_cb])
    out = []
    L = len(pep)
    for i, r in enumerate(pep):
        if terminal_only and not (i < 2 or i >= L - 2):
            continue
        c = cb(r)
        if c is None:
            continue
        d = np.sqrt(((coords - c) ** 2).sum(1))
        for j in np.where(d < CUT)[0]:
            out.append((A3[r.resname.upper()], rec_cb[j][1]))
    return out


def build_potential(complexes, terminal_only=False):
    obs = defaultdict(float); marg_p = defaultdict(float); marg_r = defaultdict(float); tot = 0.0
    for pep, rec in complexes:
        for a, b in interface_pairs(pep, rec, terminal_only):
            obs[(a, b)] += 1; marg_p[a] += 1; marg_r[b] += 1; tot += 1
    if tot < 100:
        return None
    pot = {}
    for a in AAS:
        for b in AAS:
            exp = (marg_p[a] / tot) * (marg_r[b] / tot) * tot
            pot[(a, b)] = float(np.log((obs[(a, b)] + 0.5) / (exp + 0.5)))  # log-odds, +0.5 smoothing
    return pot


def score(pep_pdb, rec_pdb, pot, terminal_only=False):
    pr = interface_pairs(pep_pdb, rec_pdb, terminal_only)
    if not pr:
        return None
    vals = [pot.get((a, b), 0.0) for a, b in pr]
    return float(np.mean(vals)), float(np.sum(vals))   # intensive (mean), extensive (sum)


def main():
    pepdirs = list((ROOT / "datasets/training_formatted_peppc").glob("peppc_*"))
    random.seed(0); random.shuffle(pepdirs)
    mine = []
    for d in pepdirs[:N_MINE * 2]:
        st = d.name.replace("peppc_", "")
        pep = d / f"peppc_{st}_peptide.pdb"; rec = d / f"peppc_{st}_protein_pocket.pdb"
        if pep.exists() and rec.exists():
            mine.append((pep, rec))
        if len(mine) >= N_MINE:
            break
    print(f"=== E53: mining {len(mine)} peppc interfaces ===", flush=True)
    pot_all = build_potential(mine, False)
    pot_term = build_potential(mine, True)
    print("  potentials built (all-residue + terminal-only)", flush=True)
    # save for reuse
    Path("/tmp/e53_pot.json").write_text(json.dumps(
        {"all": {f"{a}{b}": pot_all[(a, b)] for a in AAS for b in AAS},
         "term": {f"{a}{b}": pot_term[(a, b)] for a in AAS for b in AAS}}))

    # diagnostic: top favorable/unfavorable charged pairs (does it learn charge complementarity?)
    chg = [(p, pot_all[p]) for p in pot_all if p[0] in "DEKR" and p[1] in "DEKR"]
    chg.sort(key=lambda x: x[1])
    print("  charged AA-pair log-odds (− favorable): most favorable",
          [(f"{a}-{b}", round(v, 2)) for (a, b), v in chg[:3]],
          "| least", [(f"{a}-{b}", round(v, 2)) for (a, b), v in chg[-3:]], flush=True)

    # H4/H5: test on crystal-65 + the-98
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    rows = []
    for k, m in bench.items():
        pep, rec = ROOT / m["peptide_pdb"], ROOT / m["pocket_pdb"]
        if pep.exists() and rec.exists() and m.get("peptide_seq"):
            sa = score(pep, rec, pot_all); st = score(pep, rec, pot_term, True)
            if sa and st:
                seq = m["peptide_seq"]
                rows.append(dict(pdb=k, y=m["dg_exp"], kbp_all_mean=sa[0], kbp_all_sum=sa[1],
                                 kbp_term_mean=st[0], cf=sum(c in "DEKR" for c in seq) / len(seq), ds="cr"))
    for k, v in e28.items():
        pep, rec = work / f"{k}_pep.pdb", work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            sa = score(pep, rec, pot_all); st = score(pep, rec, pot_term, True)
            if sa and st:
                seq = "".join(A3.get(r.resname.upper(), "X")
                              for r in P.get_structure("p", str(pep))[0].get_residues() if r.id[0] == " ")
                rows.append(dict(pdb=k, y=v["y"], kbp_all_mean=sa[0], kbp_all_sum=sa[1],
                                 kbp_term_mean=st[0], cf=sum(c in "DEKR" for c in seq) / max(1, len(seq)), ds="b98"))
    Path("/tmp/e53_scored.json").write_text(json.dumps(rows))
    y = np.array([r["y"] for r in rows]); cf = np.array([r["cf"] for r in rows]); h = cf >= 0.3
    print(f"\n=== mined potential vs Kd (n={len(rows)}, charged={h.sum()}) — does charge/terminal beat MJ? ===")
    print(f"  {'feature':<16}{'all Spear':>11}{'charged Spear':>15}")
    for f in ["kbp_all_mean", "kbp_all_sum", "kbp_term_mean"]:
        v = np.array([r[f] for r in rows])
        print(f"  {f:<16}{spearmanr(v,y).statistic:>+11.3f}{spearmanr(v[h],y[h]).statistic:>+15.3f}")
    print("  (reference: MJ residue potential ~0.3 within-set; charged floor ~0.07-0.16)")
    print("  >> if kbp_term_mean charged > 0.2 AND beats all-residue, terminal-debiasing captured")
    print("     charged signal residue-MJ misses — the Ram fragment hypothesis works.")


if __name__ == "__main__":
    main()
