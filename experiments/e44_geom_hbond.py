"""E44 — INTENSIVE geometry-weighted H-bond term (FlexPepDock's insight + our generalization).

FlexPepDock's engine is geometry-weighted H-bonds (hbond_lr_bb −0.52 on crystal-65) — but as
EXTENSIVE energies they're size-confounded and collapse on diverse data (−0.03 on the 98), like
all extensive terms. The synthesis: weight each H-bond by its GEOMETRIC QUALITY (FlexPepDock),
but aggregate INTENSIVELY (per-H-bond mean / per-residue) so it generalizes (our lesson).

Per peptide-receptor H-bond (donor/acceptor N/O pair within 3.5 Å), geometric quality:
  q = f_dist(d) · f_angle(θ)
    f_dist  = exp(−((d − 2.9)/0.6)²)        optimal D-A distance ~2.9 Å
    f_angle = cos²(θ_antecedent)            linearity: A anti to donor's antecedent heavy atom
Features:
  hb_qual_mean  = mean q over H-bonds        INTENSIVE quality (the key — should generalize)
  hb_qual_perL  = Σ q / L                     per-residue (intensive)
  hb_qual_sum   = Σ q                          extensive (for contrast — expect it to flip)
  hb_best       = max q                        best single H-bond
Tests sign-consistency cr vs 98 + whether the INTENSIVE quality (unlike the extensive sum)
generalizes and adds to geometry+entropy.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Bio.PDB import NeighborSearch, PDBParser  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

P = PDBParser(QUIET=True)


def _antecedent(atom, res):
    """A bonded heavy neighbor of `atom` within the residue (proxy for the donor antecedent)."""
    best = None; bd = 99.0
    for a in res:
        if a is atom or a.element == "H":
            continue
        d = float(np.linalg.norm(a.coord - atom.coord))
        if 1.0 < d < 1.8 and d < bd:
            bd = d; best = a
    return best


def geom_hbonds(pep_pdb, rec_pdb):
    tmp = Path(f"/tmp/_e44_{Path(pep_pdb).stem}.pdb")
    lines = []
    for src, ch in ((pep_pdb, "P"), (rec_pdb, "R")):
        for ln in Path(src).read_text().splitlines():
            if ln.startswith(("ATOM", "HETATM")) and ln[17:20] != "HOH":
                lines.append(ln[:21] + ch + ln[22:])
    tmp.write_text("\n".join(lines) + "\nEND\n")
    try:
        cx = P.get_structure("c", str(tmp))[0]
        pep = [r for r in cx["P"] if r.id[0] == " "]
        L = len(pep)
        rec_no = [a for ch in cx if ch.id != "P" for r in ch if r.id[0] == " "
                  for a in r if a.element in ("N", "O")]
        if not rec_no or not pep:
            return None
        ns = NeighborSearch(rec_no)
        quals = []
        for r in pep:
            for a in r:
                if a.element not in ("N", "O"):
                    continue
                ante = _antecedent(a, r)
                for b in ns.search(a.coord, 3.6):
                    d = float(np.linalg.norm(a.coord - b.coord))
                    if d < 2.3:
                        continue  # clash, not an H-bond
                    f_dist = np.exp(-((d - 2.9) / 0.6) ** 2)
                    # linearity: angle antecedent-A-B should be ~109-180; reward anti-arrangement
                    if ante is not None:
                        v1 = ante.coord - a.coord; v2 = b.coord - a.coord
                        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                        f_ang = ((1 - cos) / 2) ** 2  # max when antecedent anti to acceptor (cos=-1)
                    else:
                        f_ang = 0.5
                    quals.append(f_dist * f_ang)
        if not quals:
            return dict(hb_qual_mean=0.0, hb_qual_perL=0.0, hb_qual_sum=0.0, hb_best=0.0)
        quals = np.array(quals)
        return dict(hb_qual_mean=float(quals.mean()), hb_qual_perL=float(quals.sum() / L),
                    hb_qual_sum=float(quals.sum()), hb_best=float(quals.max()))
    finally:
        tmp.unlink(missing_ok=True)


def main():
    bench = {r["pdb"].upper(): r for r in json.loads((ROOT / "data/benchmark_crystal.json").read_text())}
    e28 = json.loads(Path("/tmp/e28_feats.json").read_text())
    work = Path("/tmp/ppep_work")
    cr_path, b_path = Path("/tmp/e44_cr.json"), Path("/tmp/e44_b98.json")
    cr = json.loads(cr_path.read_text()) if cr_path.exists() else {}
    b98 = json.loads(b_path.read_text()) if b_path.exists() else {}
    for k, m in bench.items():
        if k in cr:
            continue
        if (ROOT / m["peptide_pdb"]).exists():
            f = geom_hbonds(str((ROOT / m["peptide_pdb"]).resolve()), str((ROOT / m["pocket_pdb"]).resolve()))
            if f:
                cr[k] = dict(f, y=m["dg_exp"]); cr_path.write_text(json.dumps(cr))
    for k, v in e28.items():
        if k in b98:
            continue
        pep = work / f"{k}_pep.pdb"; rec = work / f"{k}_rec.pdb"
        if pep.exists() and rec.exists():
            f = geom_hbonds(str(pep), str(rec))
            if f:
                b98[k] = dict(f, y=v["y"]); b_path.write_text(json.dumps(b98))
    ycr = np.array([cr[k]["y"] for k in cr]); y98 = np.array([b98[k]["y"] for k in b98])
    FEATS = ["hb_qual_mean", "hb_qual_perL", "hb_qual_sum", "hb_best"]
    print(f"cr={len(cr)} b98={len(b98)}")
    print("=== geometry-weighted H-bond: INTENSIVE (mean) vs EXTENSIVE (sum), sign-consistency ===")
    print(f"  {'feature':<14}{'crystal-65':>12}{'the-98':>10}{'universal':>11}")
    for f in FEATS:
        vc = np.array([cr[k][f] for k in cr]); v9 = np.array([b98[k][f] for k in b98])
        rc = pearsonr(vc, ycr).statistic if vc.std() > 0 else 0
        r9 = pearsonr(v9, y98).statistic if v9.std() > 0 else 0
        u = "YES" if rc * r9 > 0 and min(abs(rc), abs(r9)) > 0.1 else ""
        print(f"  {f:<14}{rc:>+12.3f}{r9:>+10.3f}{u:>11}")
    # does intensive H-bond quality add to geometry+entropy, generalizably?
    inten = None
    e31 = Path("/tmp/e31_intensive.json")
    print("  (compare: extensive sum should flip like Rosetta's hbond; intensive mean should hold)")
    print("  >> if hb_qual_mean is UNIVERSAL, it's the geometry-weighting FlexPepDock has + our intensive fix")


if __name__ == "__main__":
    main()
