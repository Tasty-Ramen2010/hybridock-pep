"""E313 — "poor-man's FEP": can we score peptide mutations stepwise like FEP integrates dU/dλ? REFUTED.

FEP does NOT compute absolute kcal/mol the naive way — it transforms only the differing atoms along an
alchemical path (many small λ windows) in BOTH the bound and free states, and integrates the well-conditioned
derivative ⟨dU/dλ⟩, so it accumulates small increments instead of subtracting two large cancelling numbers
(NAMD/Perses/OpenFE). Ram's idea: a cheap analogue — score mutants/variants with our function and take
differences, decomposing big changes into single mutations like FEP's λ windows.

DECISIVE TEST (the FEP small-perturbation principle): does our ΔΔG accuracy improve for SMALLER mutations?
On same-receptor charged pairs, bin by sequence edit distance, measure ΔΔG r + sign accuracy (leave-receptor-
out absolute model, ΔΔG = score_i − score_j):

  edit dist        n    ΔΔG r   sign acc      mean|ΔΔG|
  1 (single mut)  179   +0.14   51% (coin)    1.08
  2-3              17   +0.32   67%           0.66
  4-6              13   +0.71   73%           1.16
  7+               67   +0.38   63%           1.58

The OPPOSITE of FEP: our scorer is WORST at single mutations (51% = coin flip) and best at large changes.
Mechanism: the scorer is shape/burial-dominated and side-chain-blind (poly-ALA relabel moved ΔG 0.07 kcal,
E308), so a single mutation barely changes the features → ΔΔG ≈ noise. FEP's small step is a physical
DERIVATIVE (well-conditioned); ours is coarse shape (a single step is noise), so stepwise path-summation of
coin-flips cannot work. Not a magnitude artifact: single-mutation |ΔΔG| (1.08) ≈ the 4-6 bin (1.16), yet
accuracy is 51% vs 73%. Scope implication: strong on DIVERSE candidate panels (4-6 mut r=0.71 — screening),
weak on single-residue lead-optimisation. Run: OMP_NUM_THREADS=1 python scripts/e313_poor_mans_fep.py
"""
from __future__ import annotations
import json, os, hashlib
from collections import defaultdict
from itertools import combinations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache = json.load(open(os.path.join(ROOT, "data/e296_ifp_cache.json")))
protd = {json.loads(l)["pdb"].lower() for l in open(os.path.join(ROOT, "data/e180_protdcal3d.jsonl"))}
data = [d for d in cache if d["pdb"] in protd]
X = np.array([d["x"] for d in data]); IFP = np.array([d["ifp"] for d in data]); y = np.array([d["y"] for d in data])
q = np.array([abs(float(d["q"])) for d in data]); peps = [d["pep"] for d in data]
grp = np.array([int(hashlib.md5(d["rseq"].encode()).hexdigest()[:8], 16) for d in data])
F = np.hstack([X, IFP])

p = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(F, y, grp):
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    p[te] = m.fit(F[tr], y[tr]).predict(F[te])


def ed(a, b):
    L = min(len(a), len(b))
    return abs(len(a) - len(b)) + sum(a[i] != b[i] for i in range(L))


byr = defaultdict(list)
for i, g in enumerate(grp):
    byr[g].append(i)
bins = defaultdict(lambda: [[], [], []])  # bin -> (true ΔΔG, pred ΔΔG, |true|)
for g, idx in byr.items():
    for i, j in combinations(idx, 2):
        if q[i] < 2 and q[j] < 2:
            continue
        d = ed(peps[i], peps[j])
        b = "1 (single)" if d == 1 else "2-3" if d <= 3 else "4-6" if d <= 6 else "7+"
        bins[b][0].append(y[i] - y[j]); bins[b][1].append(p[i] - p[j]); bins[b][2].append(abs(y[i] - y[j]))

print("Does charged ΔΔG accuracy improve for SMALLER mutations? (FEP small-perturbation principle)")
print(f"{'edit dist':12s}{'n':>6s}{'ΔΔG r':>9s}{'sign acc':>10s}{'mean|ΔΔG|':>11s}")
for b in ["1 (single)", "2-3", "4-6", "7+"]:
    t, pr, mag = map(np.array, bins[b])
    if len(t) < 5:
        print(f"{b:12s}{len(t):>6d}   (too few)"); continue
    mask = np.abs(t) >= 0.5
    sa = np.mean((pr[mask] > 0) == (t[mask] > 0)) if mask.sum() else float("nan")
    print(f"{b:12s}{len(t):>6d}{pearsonr(t, pr)[0]:>+9.3f}{sa:>9.0%}{mag.mean():>11.2f}")
print("\nVERDICT: single-mutation ΔΔG is coin-flip (51%) — the reverse of FEP. A shape-dominated static scorer")
print("cannot do FEP's small-step decomposition; its single-residue derivative is noise, not a physical dU/dλ.")
