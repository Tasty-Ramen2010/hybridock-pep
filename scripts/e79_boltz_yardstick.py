"""E79 — Boltz-2 co-folding CONFIDENCE yardstick for protein-peptide ranking.

Boltz-2's AFFINITY head is small-molecule only (docs: binder must be a ligand chain <=56 atoms; peptides
forbidden). So we cannot get a Kd from it. The honest (b)-lever test: does Boltz-2's co-folding CONFIDENCE
(ipTM / pTM / pLDDT) RANK binding strength on our peptides? If a frontier co-folding model's confidence
tracks ΔG, it's a yardstick (and a candidate feature); if it's flat (all known binders fold confidently),
it confirms there's no off-the-shelf frontier substitute for our physics ranking.

Picks: strong/weak extremes of the98 with tractable receptors (40-320 res). Co-fold receptor (chain A) +
peptide (chain B) with --use_msa_server, parse confidence, correlate with experimental ΔG.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    P = PDBParser(QUIET=True)
    HAVE_BIO = True
except Exception:  # noqa: BLE001 - boltz-env may lack biopython; YAMLs pre-written by score-env pass
    HAVE_BIO = False

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/ppep_work")
OUTROOT = Path.home() / ".boltz_yard"
OUTROOT.mkdir(exist_ok=True)
BOLTZ = "/home/igem/miniconda3/envs/boltz-env/bin/boltz"

PICKS = ["2HFG_H_R", "2BYP_C_I", "3LK4_D_F", "3KJ0_A_B",   # strong
         "1G6R_C_Q", "2O9K_A_B", "1AWI_A_P", "1OSV_B_D"]   # weak


def chain_seq(pdb, longest=True):
    m = P.get_structure("x", str(pdb))[0]
    seqs = []
    for ch in m.get_chains():
        res = [r for r in ch.get_residues() if r.id[0] == " "]
        if res:
            seqs.append("".join(seq1(r.resname, undef_code="X") for r in res))
    seqs = [s for s in seqs if set(s) != {"X"}]
    if not seqs:
        return ""
    return max(seqs, key=len) if longest else seqs[0]


def write_yaml(cid):
    y = OUTROOT / f"{cid}.yaml"
    if y.exists():            # pre-written by the score-env pass; boltz-env reuses
        return y
    if not HAVE_BIO:
        return None
    rec = chain_seq(WORK / f"{cid}_rec.pdb")
    pep = chain_seq(WORK / f"{cid}_pep.pdb")
    if not rec or not pep:
        return None
    y = OUTROOT / f"{cid}.yaml"
    y.write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {rec}\n"
        f"  - protein:\n      id: B\n      sequence: {pep}\n"
    )
    return y


def parse_conf(cid):
    # boltz writes boltz_results_<name>/predictions/<name>/confidence_<name>_model_0.json
    base = OUTROOT / f"boltz_results_{cid}" / "predictions" / cid
    for f in base.glob("confidence_*_model_0.json"):
        d = json.loads(f.read_text())
        return dict(iptm=d.get("iptm"), ptm=d.get("ptm"),
                    plddt=d.get("complex_plddt"), conf=d.get("confidence_score"))
    return None


def main():
    e = json.load(open("/tmp/e49b_the98.json"))
    run = "--run" in sys.argv
    rows = []
    for cid in PICKS:
        y = e[cid]["y"]
        yaml = write_yaml(cid)
        if yaml is None:
            print(f"  {cid} no seq"); continue
        if run and parse_conf(cid) is None:
            print(f"=== boltz predict {cid} (y={y:+.1f}) ===", flush=True)
            subprocess.run([BOLTZ, "predict", str(yaml), "--use_msa_server",
                            "--out_dir", str(OUTROOT), "--override"],
                           cwd=str(OUTROOT))
        c = parse_conf(cid)
        if c:
            rows.append(dict(id=cid, y=y, **c))
            print(f"  {cid} y={y:+6.1f} iptm={c['iptm']} ptm={c['ptm']} plddt={c['plddt']}", flush=True)
    if len(rows) >= 5:
        import numpy as np
        from scipy.stats import spearmanr, pearsonr
        yv = np.array([r["y"] for r in rows])
        print(f"\n=== Boltz confidence vs ΔG (n={len(rows)}) ===")
        for f in ["iptm", "ptm", "plddt", "conf"]:
            x = np.array([r[f] if r[f] is not None else np.nan for r in rows])
            m = ~np.isnan(x)
            if m.sum() >= 5:
                print(f"  {f:<7} Spearman={spearmanr(x[m], yv[m]).statistic:+.3f}  "
                      f"Pearson={pearsonr(x[m], yv[m])[0]:+.3f}")
        (ROOT / "data/e79_boltz_confidence.json").write_text(json.dumps(rows, indent=2))
        print("  >> strong NEG corr (higher confidence -> lower/stronger ΔG) = Boltz ranks affinity.")
        print("     flat = no off-the-shelf frontier substitute for our physics ranking.")


if __name__ == "__main__":
    main()
