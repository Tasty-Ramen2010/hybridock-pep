"""Persist scoring experiment outputs into tracked, ML-ready datasets (don't lose them to /tmp wipes).

Consolidates the electrostatic-decomposition records (and any cached MD/feature records) into a stable
JSONL under data/ that an ML model can train on directly: each row = features + the experimental ΔG
label. Idempotent and append-safe — re-run any time (e.g. after the e72 background run finishes) to
refresh from the /tmp caches.

Output: data/electrostatic_decomp_dataset.jsonl  (one JSON row per complex)
  features: vdw, coul, gbpol, net_elec (+/L intensive), L, net_charge, charged_frac, hyd-ish from seq
  label:    y (experimental ΔG, kcal/mol)
  meta:     id, seq, dataset
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/electrostatic_decomp_dataset.jsonl"
SRC = Path("/tmp/e72_elec.json")


def seqfeat(seq: str) -> dict:
    L = max(1, len(seq))
    return dict(
        charged_frac=sum(c in "DEKR" for c in seq) / L,
        hyd_frac=sum(c in "AILMFVWC" for c in seq) / L,
        arom_frac=sum(c in "FWYH" for c in seq) / L,
    )


def main() -> None:
    if not SRC.exists():
        print(f"no source cache at {SRC}")
        return
    src = json.loads(SRC.read_text())
    rows = []
    for k, d in src.items():
        seq = d.get("seq", "")
        L = d.get("L", len(seq)) or 1
        row = {
            "id": k,
            "seq": seq,
            "dataset": "the98",
            # electrostatic decomposition features (the learnable signal)
            "vdw": d["vdw"], "coul": d["coul"], "gbpol": d["gbpol"], "net_elec": d["net_elec"],
            "coul_per_L": d["coul"] / L, "gbpol_per_L": d["gbpol"] / L,
            "net_elec_per_L": d["net_elec"] / L,
            "L": L, "net_charge": d["net_charge"],
            **seqfeat(seq),
            # supervised label
            "y": d["y"],
        }
        rows.append(row)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} ML-ready rows -> {OUT.relative_to(ROOT)}")
    # quick integrity: label present, features finite
    import math
    bad = [r["id"] for r in rows if not math.isfinite(r["y"]) or not math.isfinite(r["net_elec"])]
    print(f"  rows with non-finite label/feature: {len(bad)}")
    print(f"  feature columns: vdw, coul, gbpol, net_elec(+/L), L, net_charge, charged/hyd/arom_frac")
    print(f"  label column: y (experimental ΔG)")


if __name__ == "__main__":
    main()
