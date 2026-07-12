"""E3f — expand independent families and re-test the NIS cross-family signal.

The permutation p≈0.06 is n-limited (14-20 independent families). The bulk pool
has ~30 new Kd PDBs (most already on disk). Extract peptide+receptor, compute
nis_p_frac, merge with the 65-set, re-cluster families, and re-run the family-mean
permutation test. If more independent families push p<0.05, NIS is conclusively a
cross-family absolute-ΔG signal.

Defensive extraction: peptide = shortest protein chain with 4-35 residues; receptor
= all other chains; nis via scoring/nis.py contact definition. Failures skipped.

!! CAVEAT (2026-06-10): this shortest-chain/whole-receptor heuristic produces
DEGENERATE nis_p (~0 or ~1) for 88% of the new Kd complexes — it mis-identifies
the peptide/pocket. So the "new-Kd" replication here is an INVALID test, not a
negative result. The Kd-expanded p=0.029 line was a merge artifact; a clean
disentangle (orig-Kd −0.535 / new-Kd +0.095) shows the new extractions carry no
usable signal. Proper curation (correct chain ID + pocket crop) is required to
actually test cross-family replication. See docs/overnight_scoring_2026-06-10.md.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Bio.PDB import PDBIO, PDBParser, Select  # noqa: E402
from Bio.SeqUtils import seq1  # noqa: E402

P = PDBParser(QUIET=True)
_IO = PDBIO()


class _ChainSel(Select):
    def __init__(self, keep):
        self.keep = set(keep)

    def accept_chain(self, ch):
        return ch.id in self.keep

    def accept_residue(self, res):
        return res.id[0] == " "


def extract(pdb_path: Path, out_dir: Path):
    """Return (peptide_pdb, receptor_pdb, seq) or None."""
    s = P.get_structure("x", str(pdb_path))[0]
    chains = []
    for ch in s:
        aa = [r for r in ch if r.id[0] == " " and r.resname != "HOH"]
        if 4 <= len(aa) <= 35:
            chains.append((len(aa), ch.id, aa))
    if not chains:
        return None
    chains.sort()
    pep_len, pep_id, pep_aa = chains[0]
    other = [ch.id for ch in s if ch.id != pep_id
             and any(r.id[0] == " " for r in ch)]
    if not other:
        return None
    try:
        seq = "".join(seq1(r.resname, custom_map={}) for r in pep_aa)
    except Exception:
        seq = "".join("X" for _ in pep_aa)
    out_dir.mkdir(parents=True, exist_ok=True)
    pep_path = out_dir / f"{pdb_path.stem}_pep.pdb"
    rec_path = out_dir / f"{pdb_path.stem}_rec.pdb"
    _IO.set_structure(s)
    _IO.save(str(pep_path), _ChainSel([pep_id]))
    _IO.save(str(rec_path), _ChainSel(other))
    return pep_path, rec_path, seq


def kmer_groups(seqs, th=0.3, k=3):
    ks = [{s[i:i+k] for i in range(max(0, len(s)-k+1))} for s in seqs]
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            u = len(ks[i] | ks[j])
            D[i, j] = D[j, i] = 1.0 - (len(ks[i] & ks[j]) / u if u else 0.0)
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - th).fit_predict(D)


def resid(x, z):
    if np.std(z) == 0:
        return x - x.mean()
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def perm_p(V, Y, L, n=20000, seed=0):
    vr, yr = resid(V, L), resid(Y, L)
    r = pearsonr(vr, yr).statistic
    rng = np.random.default_rng(seed)
    cnt = sum(abs(pearsonr(vr, resid(Y[rng.permutation(len(Y))], L)).statistic) >= abs(r)
              for _ in range(n))
    return r, (cnt + 1) / (n + 1)


def main():
    from hybridock_pep.scoring.nis import compute_nis_features

    # existing 65-set
    base = json.loads(Path("/tmp/e3_features.json").read_text())
    merged = [dict(seq=r["seq"], L=r["L"], y=r["y"], aff=r["aff"],
                   nis_p=r["nis_p_frac"], src="orig", pdb=r["pdb"]) for r in base]
    cur = {r["pdb"].upper() for r in base}

    # new Kd (+ Ki) from bulk pool
    rows = list(csv.DictReader(open(ROOT / "data/rcsb_binding_affinity_bulk.csv")))
    seen = set()
    new = []
    for r in rows:
        if r["affinity_type"] not in ("Kd", "Ki"):
            continue
        pid = r["pdb_id"].upper()
        if pid in cur or pid in seen:
            continue
        try:
            pkd = float(r["experimental_pkd"])
        except (ValueError, KeyError):
            continue
        seen.add(pid)
        new.append((pid, r["affinity_type"], pkd))

    out_dir = Path("/tmp/e3f_extract")
    ok = fail = 0
    for pid, aff, pkd in new:
        raw = ROOT / f"datasets/raw_pdbs/{pid}.pdb"
        if not raw.exists():
            fail += 1
            continue
        try:
            res = extract(raw, out_dir)
            if res is None:
                fail += 1
                continue
            pep, rec, seq = res
            polar, charged = compute_nis_features(pep, rec)
            merged.append(dict(seq=seq, L=len(seq), y=-1.364 * pkd, aff=aff,
                               nis_p=polar, src="new", pdb=pid))
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
    print(f"new complexes added: {ok}  (failed/missing: {fail})")

    for label, sub in [("ALL orig", [m for m in merged if m["src"] == "orig"]),
                       ("ALL expanded", merged),
                       ("Kd expanded", [m for m in merged if m["aff"] == "Kd"])]:
        seqs = [m["seq"] for m in sub]
        g = kmer_groups(seqs, 0.3)
        # family-mean collapse
        d = {}
        for i, gi in enumerate(g):
            d.setdefault(gi, []).append(i)
        ks = sorted(d)
        Y = np.array([np.mean([sub[i]["y"] for i in d[k]]) for k in ks])
        L = np.array([np.mean([sub[i]["L"] for i in d[k]]) for k in ks])
        V = np.array([np.nanmean([sub[i]["nis_p"] for i in d[k]]) for k in ks])
        ok2 = np.isfinite(V)
        r, p = perm_p(V[ok2], Y[ok2], L[ok2])
        print(f"  {label:<14} n={len(sub):<3} families={ok2.sum():<3} "
              f"nis_p lenresid r={r:+.3f}  perm p={p:.4f}")


if __name__ == "__main__":
    main()
