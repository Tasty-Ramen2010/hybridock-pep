"""E25 — ESM per-CONTACT energy (Ram's idea: ESM models two amino acids touching).

For each peptide-receptor residue contact, score the interaction from ESM-2 per-residue
embeddings of the two residues (peptide residue in peptide context, receptor residue in
receptor-pocket context). Test whether ESM beats the fixed MJ contact potential (0.615
with geometry). NO per-contact FITTING (overfits 65 pts) — fixed reductions only:
  esm_dot   : Σ_contacts <emb_i, emb_j>        (biochemical compatibility)
  esm_cos   : Σ_contacts cos(emb_i, emb_j)
  esm_min   : most-favorable (max-similarity) single contact = dominant hotspot

Two stages: (1) extract per-residue embeddings (rapidock env, CPU) -> /tmp/e25_emb.json;
(2) eval (score-env). Run stage 1 with the rapidock python, stage 2 with score-env python.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


# ---------------- STAGE 1: extract per-residue embeddings (rapidock env) ----------------

def extract():
    import torch
    import esm
    from Bio.PDB import PDBParser
    P = PDBParser(QUIET=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter(); model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    def emb(seq):
        if not seq:
            return []
        _, _, toks = bc([("p", seq)])
        with torch.no_grad():
            rep = model(toks.to(dev), repr_layers=[33])["representations"][33][0]
        return rep[1:len(seq) + 1].float().cpu().numpy()  # per residue (strip BOS)

    e0 = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e0_rows.json").read_text())}
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    out_path = Path("/tmp/e25_emb.json")
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for pdb, r0 in e0.items():
        if pdb in out or pdb not in bench or not r0.get("poc_pdb"):
            continue
        pep_seq = bench[pdb]["peptide_seq"]
        # pocket sequence from poc_pdb residue order
        poc = P.get_structure("r", r0["poc_pdb"])[0]
        poc_res = [(AA3to1.get(res.resname.upper(), "A"))
                   for ch in poc for res in ch if res.id[0] == " "]
        poc_seq = "".join(poc_res)
        if not pep_seq or not poc_seq or len(poc_seq) > 1022:
            continue
        try:
            pe = emb(pep_seq); re = emb(poc_seq)
        except Exception as ex:  # noqa: BLE001
            print(f"  {pdb} emb FAIL {type(ex).__name__}", flush=True); continue
        out[pdb] = dict(pep_seq=pep_seq, poc_seq=poc_seq,
                        pep_emb=pe.tolist(), poc_emb=re.tolist())
        out_path.write_text(json.dumps(out))
        print(f"  {pdb}: pep{len(pep_seq)} poc{len(poc_seq)} ({len(out)})", flush=True)
    print(f"done {len(out)}", flush=True)


# ---------------- STAGE 2: contact features + eval (score-env) ----------------

def evaluate():
    from Bio.PDB import NeighborSearch, PDBParser
    from scipy.stats import pearsonr
    P = PDBParser(QUIET=True)
    emb = json.loads(Path("/tmp/e25_emb.json").read_text())
    geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
    mj = json.loads(Path("/tmp/e24_contact.json").read_text())
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    POCK = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis"]
    IFACE = ["bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count"]
    GEO = POCK + IFACE

    def contact_pairs(cx_path, npep, npoc, cut=6.5):
        cx = P.get_structure("c", cx_path)[0]
        pep = [r for r in cx["P"] if r.id[0] == " "][:npep]
        rec = [r for ch in cx if ch.id != "P" for r in ch if r.id[0] == " "][:npoc]
        rec_atoms = [a for r in rec for a in r if a.element != "H"]
        ns = NeighborSearch(rec_atoms)
        ridx = {id(r): k for k, r in enumerate(rec)}
        pairs = set()
        for i, rp in enumerate(pep):
            for atom in rp:
                if atom.element == "H":
                    continue
                for b in ns.search(atom.coord, cut):
                    j = ridx.get(id(b.get_parent()))
                    if j is not None:
                        pairs.add((i, j))
        return list(pairs)

    rows = []
    for pdb, g in geo.items():
        if pdb not in emb or pdb not in mj:
            continue
        merged = Path(f"/tmp/e18v3_cx/{pdb}.pdb")
        if not merged.exists():
            continue
        pe = np.array(emb[pdb]["pep_emb"]); re = np.array(emb[pdb]["poc_emb"])
        if pe.ndim != 2 or re.ndim != 2:
            continue
        pen = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-9)
        ren = re / (np.linalg.norm(re, axis=1, keepdims=True) + 1e-9)
        pairs = contact_pairs(str(merged), len(pe), len(re))
        if not pairs:
            continue
        dots, coss = [], []
        for i, j in pairs:
            if i < len(pe) and j < len(re):
                dots.append(float(pe[i] @ re[j]))
                coss.append(float(pen[i] @ ren[j]))
        if not dots:
            continue
        rows.append(dict(g, esm_dot=float(np.sum(dots)), esm_cos=float(np.sum(coss)),
                         esm_cos_max=float(np.max(coss)), mj_contact=mj[pdb]["mj_contact"],
                         vina=bench[pdb]["vina_docked"]))
    y = np.array([r["y"] for r in rows])

    def loo(X, y):
        p = np.zeros(len(y))
        for i in range(len(y)):
            tr = [j for j in range(len(y)) if j != i]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
            w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            p[i] = np.r_[1, (X[i] - mu) / sd] @ w
        return p

    def rr(p):
        return pearsonr(p, y).statistic, float(np.sqrt(((p - y) ** 2).mean()))
    Xg = np.array([[r.get(f, 0.0) for f in GEO] for r in rows])
    print(f"n={len(rows)}  (ESM per-contact vs MJ; kcal/mol RMSE)\n")
    for nm, extra in [("geometry", []), ("+esm_dot", ["esm_dot"]), ("+esm_cos", ["esm_cos"]),
                      ("+esm_cos_max", ["esm_cos_max"]), ("+MJ", ["mj_contact"]),
                      ("+MJ+esm_cos", ["mj_contact", "esm_cos"])]:
        X = np.column_stack([Xg] + [[r[e] for r in rows] for e in extra]) if extra else Xg
        p = loo(X, y); r, rmse = rr(p)
        raw = ""
        if len(extra) == 1:
            raw = f"  (raw corr {pearsonr([rr_[extra[0]] for rr_ in rows], y).statistic:+.3f})"
        print(f"  {nm:<16} r={r:+.3f}  RMSE={rmse:.2f}{raw}")
    print("\n>> if esm beats MJ -> ESM models per-contact energy; else MJ (physics) wins.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract()
    else:
        evaluate()
