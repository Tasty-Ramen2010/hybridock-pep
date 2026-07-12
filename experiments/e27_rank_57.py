"""E27 — rank the 57 PepPC complexes by PREDICTED ΔG (our geometry+MJ scorer).

No experimental Kd exists for these, so this is the tool's PREDICTION, not validated truth.
Fit the geometry+MJ linear model on labeled crystal-65, apply to the 57 crystal poses,
rank strongest->weakest predicted binder. Sanity columns: hydrophobic burial, aromatic
contacts, MJ contact energy, hb. Flags out-of-distribution SS (sheet/long) where the
absolute number is an extrapolation.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridock_pep.scoring.ensemble import GEOMETRY_FEATURES  # noqa: E402
from hybridock_pep.scoring.geometry_features import compute_geometry_features  # noqa: E402

# --- fit geometry+MJ on labeled crystal-65 (oracle poses) ---
geo = {r["pdb"].upper(): r for r in json.loads(Path("/tmp/e19_cr.json").read_text())}
mj = json.loads(Path("/tmp/e24_contact.json").read_text())
train = [dict(geo[p], mj_contact=mj[p]["mj_contact"]) for p in geo if p in mj]
ytr = np.array([r["y"] for r in train])
Xtr = np.array([[r.get(f, 0.0) for f in GEOMETRY_FEATURES] for r in train])
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
w, *_ = np.linalg.lstsq(A, ytr, rcond=None)


def predict(feat):
    x = np.array([feat.get(f, 0.0) for f in GEOMETRY_FEATURES])
    return float(np.r_[1, (x - mu) / sd] @ w)


# --- score the 57 on their crystal poses ---
rows = list(csv.DictReader(open(ROOT / "data/gen_n500_57.csv")))
scored = []
for r in rows:
    f = compute_geometry_features(Path(r["peptide_pdb"]), Path(r["receptor"]))
    if not f:
        continue
    dg = predict(f)
    ood = (r["ss_class"] == "SHEET") or (int(r["pep_len"]) > 20)
    scored.append(dict(name=r["name"][6:], seq=r["seq"], L=int(r["pep_len"]),
                       ss=r["ss_class"], pred_dg=dg, mj=f["mj_contact"],
                       bsa_hyd=f["bsa_hyd"], arom=f["arom_cc"], hb=f["hb_count"], ood=ood))

scored.sort(key=lambda d: d["pred_dg"])  # most negative = strongest first
print(f"Ranked {len(scored)} complexes by PREDICTED ΔG (kcal/mol). "
      f"NO experimental Kd — this is the tool's prediction, not validated.\n")
print(f"{'#':>3} {'complex':<14}{'L':>3} {'SS':<8}{'predΔG':>8}{'hydΔSASA':>9}{'arom':>5}{'hb':>4}{'MJ':>8}  seq")
for i, d in enumerate(scored, 1):
    flag = " *OOD" if d["ood"] else ""
    print(f"{i:>3} {d['name']:<14}{d['L']:>3} {d['ss']:<8}{d['pred_dg']:>8.1f}"
          f"{d['bsa_hyd']:>9.1f}{int(d['arom']):>5}{int(d['hb']):>4}{d['mj']:>8.0f}  {d['seq'][:20]}{flag}")

strong = scored[:10]; weak = scored[-10:]
print(f"\nPREDICTED STRONGEST 10: mean predΔG {np.mean([d['pred_dg'] for d in strong]):.1f} "
      f"kcal/mol | mean hydΔSASA {np.mean([d['bsa_hyd'] for d in strong]):.1f} | "
      f"mean MJ {np.mean([d['mj'] for d in strong]):.0f} | aromatic in {sum(d['arom']>0 for d in strong)}/10")
print(f"PREDICTED WEAKEST 10:   mean predΔG {np.mean([d['pred_dg'] for d in weak]):.1f} "
      f"kcal/mol | mean hydΔSASA {np.mean([d['bsa_hyd'] for d in weak]):.1f} | "
      f"mean MJ {np.mean([d['mj'] for d in weak]):.0f} | aromatic in {sum(d['arom']>0 for d in weak)}/10")
n_ood = sum(d["ood"] for d in scored)
print(f"\n*OOD = sheet or >20mer ({n_ood}/{len(scored)}): absolute ΔG is an EXTRAPOLATION "
      f"(calibration = short helix/loop crystal-65). Ranking among in-distribution more reliable.")
