"""E121 — MD length convergence: 60 vs 200 vs 500 ps on 5 diverse peptides (pick production length).

Measures, per peptide, how s_free (mean per-residue dihedral entropy) and the per-residue TYPE signal
converge with trajectory length. If s_free plateaus by 200ps, 200 is enough; if still drifting at 500,
we need longer. The per-residue type signal = corr(per_res_entropy, residue intrinsic flexibility) — a
proxy for "is the per-residue entropy meaningful (tracks residue type) or noise?"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PEPDIR = ROOT / "data" / "sfree_peptides"
# intrinsic side-chain/backbone flexibility (higher = floppier); Gly/Ser high, Pro/Ile/Val low
FLEX = {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.50, "G": 0.54, "H": 0.32,
        "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.30, "F": 0.31, "P": 0.51, "S": 0.51, "T": 0.44, "W": 0.31,
        "Y": 0.42, "V": 0.39}


def main():
    from e18v2_md import run_free_dynamics

    index = json.loads((PEPDIR / "index.json").read_text())  # seq -> hash
    bylen = {}
    for seq, h in index.items():
        bylen.setdefault(len(seq), (seq, h))
    # 5 diverse lengths
    targets = []
    for L in sorted(bylen):
        if L in (6, 9, 12, 16, 20) and len(targets) < 5:
            targets.append(bylen[L])
    while len(targets) < 5:  # fallback: spread across available lengths
        ls = sorted(bylen)
        targets = [bylen[ls[int(i)]] for i in np.linspace(0, len(ls) - 1, 5)]
        break

    print("=== E121 MD length convergence (60/200/500 ps), 5 peptides ===\n")
    print(f"{'seq':<22}{'len':>4}{'ps':>6}{'s_free':>9}{'tot':>8}{'rmsf':>7}{'type-r':>9}{'time':>8}")
    results = {}
    for seq, h in targets:
        pep = PEPDIR / f"{h}.pdb"
        flex = np.array([FLEX.get(c, 0.45) for c in seq])
        row = {}
        for ps in (60, 200, 500):
            t = time.time()
            try:
                rmsf, ent = run_free_dynamics(pep, ps)
                ent = np.array([x if np.isfinite(x) else np.nan for x in ent])
                ok = ~np.isnan(ent)
                # align flex length to residues with entropy (phi/psi exist for interior residues)
                fl = flex[:len(ent)]
                typer = pearsonr(ent[ok], fl[ok])[0] if ok.sum() > 3 and np.std(fl[ok]) > 0 else np.nan
                sf = float(np.nanmean(ent))
                row[ps] = dict(s_free=sf, tot=float(np.nansum(ent)), rmsf=float(np.mean(rmsf)), typer=float(typer))
                dt = time.time() - t
                print(f"{seq[:22]:<22}{len(seq):>4}{ps:>6}{sf:>9.3f}{row[ps]['tot']:>8.1f}{row[ps]['rmsf']:>7.2f}{typer:>9.2f}{dt:>7.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{seq[:22]:<22}{len(seq):>4}{ps:>6}  FAIL {type(e).__name__}", flush=True)
        results[seq] = row
        print()

    # convergence summary: how much does s_free shift 60->200 and 200->500?
    print("=== CONVERGENCE (Δs_free between lengths; small 200→500 = converged at 200) ===")
    d_60_200, d_200_500, tr60, tr200, tr500 = [], [], [], [], []
    for seq, row in results.items():
        if 60 in row and 200 in row and 500 in row:
            d_60_200.append(abs(row[200]["s_free"] - row[60]["s_free"]))
            d_200_500.append(abs(row[500]["s_free"] - row[200]["s_free"]))
            tr60.append(row[60]["typer"]); tr200.append(row[200]["typer"]); tr500.append(row[500]["typer"])
    if d_60_200:
        print(f"  mean |Δs_free| 60→200 = {np.mean(d_60_200):.3f}")
        print(f"  mean |Δs_free| 200→500 = {np.mean(d_200_500):.3f}   (≪ above ⇒ 200ps already converged)")
        print(f"  per-residue TYPE signal corr: 60ps={np.nanmean(tr60):+.2f}  200ps={np.nanmean(tr200):+.2f}  500ps={np.nanmean(tr500):+.2f}")
        print("\n  reading: pick the shortest ps where (a) 200→500 shift is small AND (b) type-signal has")
        print("  plateaued. That's the production length for the full 922-peptide run.")


if __name__ == "__main__":
    main()
