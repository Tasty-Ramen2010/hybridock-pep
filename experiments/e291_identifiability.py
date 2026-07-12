"""E291 — can the 2x2 grid solve b(R1),b(R2),c(P1),c(P2)? + the variance decomposition that decides Ram's plan.

(A) RANK demo: the additive model e(P,R)=b(R)+c(P) on a 2x2 grid is a 4x4 system of RANK 3 -> the absolute
    b's and c's are NOT identifiable (1 gauge constant); only DIFFERENCES b(R1)-b(R2), c(P1)-c(P2) and the
    sums are. Demonstrate on a real grid (two gauge fixes -> same differences, different absolutes).

(B) The decisive number for Ram's plan: if we COULD solve b(R) for every receptor, how good is absolute
    scoring? = oracle-b ceiling (y_hat = S - b(R)). Compare b(R) std vs c(P) std: if b(R) dominates, then
    'solve b(R)' IS the answer; if c(P) is comparable, knowing b(R) alone won't finish the job.
Run: OMP_NUM_THREADS=1 python experiments/e291_identifiability.py
"""
from __future__ import annotations
import json, numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr


def pf(v):
    if isinstance(v, str):
        v = v.strip(); return json.loads(v) if v.startswith("[") else float(v)
    return v


# ---------- (A) rank of the 2x2 additive design ----------
# rows = [e(P1,R1), e(P1,R2), e(P2,R1), e(P2,R2)] in unknowns [b(R1),b(R2),c(P1),c(P2)]
A = np.array([[1, 0, 1, 0],   # b(R1)+c(P1)
              [0, 1, 1, 0],   # b(R2)+c(P1)
              [1, 0, 0, 1],   # b(R1)+c(P2)
              [0, 1, 0, 1]])  # b(R2)+c(P2)
print("=== (A) 2x2 additive design rank ===")
print(f"  4 equations, 4 unknowns, but rank(A) = {np.linalg.matrix_rank(A)} -> 1 gauge freedom")
print("  null space (the unidentifiable shift):", np.round(
    np.linalg.svd(A)[2][-1], 2), "= [b1,b2,c1,c2] direction (add k to b's, subtract k from c's)")
print("  => ABSOLUTE b(R1),b(R2),c(P1),c(P2) NOT solvable; only b(R1)-b(R2), c(P1)-c(P2), and sums are.")

# ---------- load data, OOF model, residuals ----------
recs = []
for r in (json.loads(l) for l in open("data/ppikb_features.jsonl")):
    if r.get("aff_type") not in ("Kd", "Ki", "KD") or not r.get("desc3d"):
        continue
    try:
        d3 = pf(r["desc3d"]); pk = pf(r["pocket_pkf"]); y = pf(r["y"])
    except Exception:
        continue
    if isinstance(d3, list) and len(d3) == 37 and isinstance(pk, list) and np.isfinite(y):
        recs.append({"rec": r["protein_seq"], "pep": r["seq"], "y": float(y),
                     "x": d3 + pk + [pf(r["length"]), pf(r["net_charge"])]})
X = np.array([r["x"] for r in recs]); y = np.array([r["y"] for r in recs])
rec = [r["rec"] for r in recs]; pep = [r["pep"] for r in recs]
grp = np.array([hash(s) % (10**9) for s in rec])
S = np.full(len(y), np.nan)
for tr, te in GroupKFold(8).split(X, y, grp):
    S[te] = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict(X[te])
e = S - y   # model residual = b(R)+c(P)+eta
idx = {(pep[i], rec[i]): i for i in range(len(recs))}

# verify on a real grid: solve with two gauge fixes, show differences match
pep_recs = defaultdict(set)
for (p, rr) in idx:
    pep_recs[p].add(rr)
multi = [p for p, rs in pep_recs.items() if len(rs) >= 2]
demo = None
for P1, P2 in combinations(multi, 2):
    common = list(pep_recs[P1] & pep_recs[P2])
    if len(common) >= 2:
        R1, R2 = common[0], common[1]
        if all((p, r) in idx for p in (P1, P2) for r in (R1, R2)):
            demo = (P1, P2, R1, R2); break
if demo:
    P1, P2, R1, R2 = demo
    e11 = e[idx[(P1, R1)]]; e12 = e[idx[(P1, R2)]]; e21 = e[idx[(P2, R1)]]; e22 = e[idx[(P2, R2)]]
    print("\n  real grid residuals: e11=%.2f e12=%.2f e21=%.2f e22=%.2f" % (e11, e12, e21, e22))
    print("  identifiable b(R1)-b(R2) = (e11-e12+e21-e22)/2 = %.2f" % ((e11 - e12 + e21 - e22) / 2))
    print("  identifiable c(P1)-c(P2) = (e11-e21+e12-e22)/2 = %.2f" % ((e11 - e21 + e12 - e22) / 2))
    print("  coupling/inconsistency e11-e12-e21+e22 = %.2f (0 in perfect additive model)"
          % (e11 - e12 - e21 + e22))
    # gauge fix 1: c(P1)=0 -> b(R1)=e11, b(R2)=e12, c(P2)=e21-e11
    print("  gauge c(P1)=0 -> b(R1)=%.2f b(R2)=%.2f c(P2)=%.2f" % (e11, e12, e21 - e11))
    # gauge fix 2: b(R2)=0 -> c(P1)=e12, b(R1)=e11-e12, c(P2)=e22
    print("  gauge b(R2)=0 -> b(R1)=%.2f c(P1)=%.2f c(P2)=%.2f" % (e11 - e12, e12, e22))
    print("  -> absolutes differ by gauge; DIFFERENCES are invariant. (this is the whole point)")

# ---------- (B) variance decomposition + oracle-b ceiling ----------
rec_cells = defaultdict(list); pep_cells = defaultdict(list)
for i in range(len(recs)):
    rec_cells[rec[i]].append(i); pep_cells[pep[i]].append(i)
bR = {s: np.mean([e[i] for i in c]) for s, c in rec_cells.items() if len(c) >= 2}
cP = {s: np.mean([e[i] for i in c]) for s, c in pep_cells.items() if len(c) >= 2}
print("\n=== (B) variance components of the model residual e=S-y ===")
print(f"  b(R) std (receptor offset, >=2 cells, n={len(bR)}): {np.std(list(bR.values())):.2f} kcal/mol")
print(f"  c(P) std (peptide bias,    >=2 cells, n={len(cP)}): {np.std(list(cP.values())):.2f} kcal/mol")
print(f"  total residual std: {np.std(e):.2f}")
# oracle-b ceiling: if b(R) known PERFECTLY (in-sample), predict y_hat = S - b(R)
brall = {s: np.mean([e[i] for i in c]) for s, c in rec_cells.items()}
yhat_oracle = np.array([S[i] - brall[rec[i]] for i in range(len(recs))])
r_oracle = pearsonr(y, yhat_oracle)[0]; m_oracle = np.mean(np.abs(y - yhat_oracle))
r_abs = pearsonr(y, S)[0]
print(f"\n  absolute (no b correction):     r={r_abs:+.3f}")
print(f"  ORACLE-b (b(R) known perfectly): r={r_oracle:+.3f} MAE={m_oracle:.2f}  <- the ceiling if we SOLVE b(R)")
print(f"  remaining error after perfect b(R) = c(P)+eta (this is what b(R) CANNOT fix)")
json.dump(dict(rank=int(np.linalg.matrix_rank(A)), bR_std=float(np.std(list(bR.values()))),
               cP_std=float(np.std(list(cP.values()))), r_abs=float(r_abs), r_oracle_b=float(r_oracle)),
          open("data/e291_ident.json", "w"), indent=1)
print("\nVERDICT: if oracle-b r is HIGH (>0.8), solving b(R) IS the answer (network/FEP worth it).")
print("If c(P) std ~ b(R) std, knowing b(R) alone won't finish — need peptide-side info too.")
print("saved data/e291_ident.json")
