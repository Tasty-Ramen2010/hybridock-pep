"""Supplementary binding affinity fetcher from three sources:

  1. PDBe REST API  — European PDB, better coverage of IC50/Kd than RCSB
  2. ChEMBL via PDB cross-reference — matches molecules with PDB structures
  3. mmCIF REMARK parsing — embedded IC50/Kd in existing .pdb.gz files

Writes data/affinity_supplement.csv (union of all three sources, deduped).
Merges with data/rcsb_binding_affinity.csv to produce data/all_binding_affinity.csv.

Usage:
    python scripts/fetch_affinity_supplement.py
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import math
import re
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"

PDBE_AFFINITY = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{}"
CHEMBL_API    = "https://www.ebi.ac.uk/chembl/api/data"
RCSB_GRAPHQL  = "https://data.rcsb.org/graphql"

PKD_MIN, PKD_MAX = 3.0, 12.0

KNOWN_PEPTIDE_TARGETS = [
    # MDM2 / MDMX — well-characterized p53 peptide binding
    "MDM2_HUMAN", "MDMX_HUMAN",
    # BCL-2 family — BH3 peptide binding
    "BCL2_HUMAN", "BCLXL_HUMAN", "MCL1_HUMAN",
    # SH3 domains with peptide ligands
    "SRC_HUMAN", "ABL1_HUMAN", "CRK_HUMAN",
    # WW domains
    "FBP11_HUMAN", "NEDD4_HUMAN",
    # PDZ domains
    "SHANK3_HUMAN", "DLG4_HUMAN", "MAGI3_HUMAN",
    # Calmodulin
    "CALM1_HUMAN", "CALM2_HUMAN",
    # Bromodomains
    "BRD4_HUMAN", "BRD2_HUMAN", "BRD3_HUMAN",
]


# ---------------------------------------------------------------------------
# Source 1: PDBe REST API
# ---------------------------------------------------------------------------

def _fetch_pdbe_affinity(pdb_ids: list[str]) -> list[dict]:
    """Query PDBe for binding site / ligand affinity data."""
    records = []
    for pdb_id in pdb_ids:
        try:
            url = PDBE_AFFINITY.format(pdb_id.lower())
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                continue
            if r.status_code != 200:
                continue
            data = r.json()
            sites = data.get(pdb_id.lower(), [])
            for site in sites:
                for ligand in site.get("site_residues", []):
                    pass
                # Look for affinity in site details
                details = site.get("details", "")
                # Search for KD/Ki/IC50 patterns in details string
                for m in re.finditer(
                    r"(?:Kd|KD|Ki|KI|IC50|kd|ki|ic50)\s*[=:]\s*([\d.]+)\s*(nM|µM|uM|mM|pM|μM)",
                    details, re.IGNORECASE
                ):
                    val_str, unit = m.group(1), m.group(2)
                    atype = m.group(0).split("=")[0].split(":")[0].strip().upper()
                    pkd = _to_pkd(float(val_str), unit)
                    if pkd and PKD_MIN <= pkd <= PKD_MAX:
                        records.append({
                            "pdb_id": pdb_id.upper(),
                            "affinity_type": atype,
                            "value": float(val_str),
                            "unit": unit,
                            "experimental_pkd": pkd,
                            "source": "pdbe",
                        })
        except Exception as exc:
            _log.debug("PDBe query failed for %s: %s", pdb_id, exc)
        time.sleep(0.05)
    return records


# ---------------------------------------------------------------------------
# Source 2: ChEMBL PDB cross-references
# ---------------------------------------------------------------------------

def _chembl_activities_for_pdb(pdb_id: str) -> list[dict]:
    """Find ChEMBL activity records for a given PDB ID via cross-reference."""
    records = []
    # Step 1: get ChEMBL target ID from PDB
    try:
        url = f"{CHEMBL_API}/target.json?target_components__xref_src=PDB&target_components__xref_id={pdb_id}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return records
        targets = r.json().get("targets", [])
    except Exception:
        return records

    for target in targets:
        tchid = target.get("target_chembl_id")
        if not tchid:
            continue
        # Step 2: get activities for this target with Kd/Ki
        try:
            url2 = (
                f"{CHEMBL_API}/activity.json"
                f"?target_chembl_id={tchid}"
                f"&standard_type__in=Kd,Ki"
                f"&pchembl_value__isnull=false"
                f"&limit=100"
            )
            r2 = requests.get(url2, timeout=20)
            if r2.status_code != 200:
                continue
            activities = r2.json().get("activities", [])
            for act in activities:
                pchembl = act.get("pchembl_value")
                if not pchembl:
                    continue
                pkd = float(pchembl)
                if not (PKD_MIN <= pkd <= PKD_MAX):
                    continue
                # Only include if molecular weight suggests a peptide (>400 Da)
                mol_weight = act.get("molecule_properties", {})
                if isinstance(mol_weight, dict):
                    mw = mol_weight.get("mw_freebase") or mol_weight.get("full_mwt")
                else:
                    mw = None
                if mw and float(mw) < 400:
                    continue  # too small to be a peptide
                records.append({
                    "pdb_id": pdb_id.upper(),
                    "affinity_type": act.get("standard_type", "Kd"),
                    "value": act.get("standard_value"),
                    "unit": act.get("standard_units", "nM"),
                    "experimental_pkd": pkd,
                    "source": "chembl",
                    "chembl_target": tchid,
                    "molecule_chembl_id": act.get("molecule_chembl_id"),
                })
        except Exception as exc:
            _log.debug("ChEMBL activity query failed for %s: %s", tchid, exc)
        time.sleep(0.1)

    return records


# ---------------------------------------------------------------------------
# Source 3: mmCIF REMARK parsing
# ---------------------------------------------------------------------------

_AFFINITY_RE = re.compile(
    r"(?:BINDING|AFFINITY|INHIBITION|KD|KI|IC50|DISSOCIATION)"
    r".*?([\d.]+(?:E[+-]?\d+)?)\s*(NM|UM|µM|MM|PM|NANOMOLAR|MICROMOLAR|NANOMOL|MICROMOL)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_pdb_remarks(pdb_gz_path: Path, pdb_id: str) -> list[dict]:
    """Extract affinity values from REMARK 300 / 400 / 800 in a .pdb.gz file."""
    records = []
    try:
        with gzip.open(pdb_gz_path, "rb") as f:
            text = f.read().decode("latin-1")
    except Exception:
        return records

    # Look in REMARKs only
    remark_lines = [l for l in text.splitlines() if l.startswith("REMARK")]
    remark_text = "\n".join(remark_lines)

    for m in _AFFINITY_RE.finditer(remark_text):
        try:
            val = float(m.group(1))
            unit = m.group(2).upper()
            pkd = _to_pkd(val, unit)
            if pkd and PKD_MIN <= pkd <= PKD_MAX:
                records.append({
                    "pdb_id": pdb_id.upper(),
                    "affinity_type": "REMARK",
                    "value": val,
                    "unit": unit,
                    "experimental_pkd": pkd,
                    "source": "mmcif_remark",
                })
        except Exception:
            continue
    return records


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def _to_pkd(val: float, unit: str) -> float | None:
    unit = unit.strip().upper()
    multipliers = {
        "NM": 1.0, "NANOMOLAR": 1.0, "NANOMOL/L": 1.0, "NANOMOL": 1.0,
        "UM": 1000.0, "µM": 1000.0, "MICROMOLAR": 1000.0,
        "μM": 1000.0, "MICROMOL": 1000.0, "MICROMOL/L": 1000.0,
        "MM": 1e6, "MILLIMOLAR": 1e6,
        "PM": 0.001, "PICOMOLAR": 0.001,
    }
    factor = multipliers.get(unit)
    if factor is None:
        return None
    v_nM = val * factor
    if v_nM <= 0:
        return None
    return round(-math.log10(v_nM * 1e-9), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Collect all PDB IDs we have on disk
    all_ids: set[str] = set()
    for ds in ["pdb_2024_2026", "ppii_enriched", "raw_pdbs",
               "pdb_2019_2023", "pdb_2010_2018", "family_targeted",
               "ppii_extended", "training_expanded_structures"]:
        d = REPO / "datasets" / ds
        if d.is_dir():
            sdir = d / "structures"
            if sdir.is_dir():
                for f in sdir.glob("*.pdb.gz"):
                    all_ids.add(f.name.split(".")[0].upper())
            else:
                for f in d.glob("*.pdb"):
                    all_ids.add(f.stem.upper())

    # Also add known training / test IDs
    for csv_f in [DATA_DIR / "training_complexes.csv",
                  DATA_DIR / "training_complexes_expanded.csv",
                  DATA_DIR / "test_complexes.csv"]:
        if csv_f.exists():
            df = pd.read_csv(csv_f)
            if "pdb_id" in df.columns:
                all_ids.update(df["pdb_id"].str.upper().tolist())

    query_ids = sorted(all_ids)
    _log.info("Total unique PDB IDs to query: %d", len(query_ids))

    all_records: list[dict] = []

    # Source 1: PDBe
    _log.info("--- Source 1: PDBe REST API ---")
    pdbe_recs = _fetch_pdbe_affinity(query_ids)
    _log.info("PDBe: %d records", len(pdbe_recs))
    all_records.extend(pdbe_recs)

    # Source 2: ChEMBL (only for known training structures, limit API calls)
    _log.info("--- Source 2: ChEMBL (training + test structures) ---")
    chembl_ids = []
    for csv_f in [DATA_DIR / "training_complexes.csv", DATA_DIR / "test_complexes.csv"]:
        if csv_f.exists():
            chembl_ids.extend(pd.read_csv(csv_f)["pdb_id"].str.upper().tolist())
    for pid in sorted(set(chembl_ids)):
        recs = _chembl_activities_for_pdb(pid)
        if recs:
            _log.info("ChEMBL for %s: %d records", pid, len(recs))
            all_records.extend(recs)
        time.sleep(0.2)

    # Source 3: mmCIF REMARK parsing on raw_pdbs
    _log.info("--- Source 3: mmCIF REMARK parsing ---")
    raw_dir = REPO / "datasets" / "raw_pdbs"
    if raw_dir.exists():
        for pdb_f in raw_dir.glob("*.pdb"):
            pdb_id = pdb_f.stem.upper()
            recs = _parse_pdb_remarks(pdb_f, pdb_id)  # plain .pdb works too
            if recs:
                _log.info("REMARK: %s → %d records", pdb_id, len(recs))
                all_records.extend(recs)

    # Also scan ppii_enriched (which has some high-quality structures)
    ppii_struct_dir = REPO / "datasets" / "ppii_enriched" / "structures"
    if ppii_struct_dir.exists():
        for f in ppii_struct_dir.glob("*.pdb.gz"):
            pdb_id = f.name.split(".")[0].upper()
            recs = _parse_pdb_remarks(f, pdb_id)
            if recs:
                _log.info("REMARK (ppii): %s → %d records", pdb_id, len(recs))
                all_records.extend(recs)

    # Save supplementary affinity
    if all_records:
        supp = pd.DataFrame(all_records)
        supp_path = DATA_DIR / "affinity_supplement.csv"
        supp.to_csv(supp_path, index=False)
        _log.info("Saved supplement: %d records → %s", len(supp), supp_path)
    else:
        _log.info("No supplementary affinity records found")
        supp = pd.DataFrame()

    # Merge with RCSB affinity
    rcsb_path = DATA_DIR / "rcsb_binding_affinity.csv"
    if rcsb_path.exists():
        rcsb = pd.read_csv(rcsb_path)
        rcsb["source"] = "rcsb"
    else:
        rcsb = pd.DataFrame()

    combined = pd.concat([rcsb, supp], ignore_index=True)
    combined = combined.dropna(subset=["experimental_pkd"])
    combined = combined[combined["experimental_pkd"].between(PKD_MIN, PKD_MAX)]

    # Deduplicate: per pdb_id, keep best (Kd > Ki > IC50)
    priority = {"Kd": 0, "KD": 0, "Ki": 1, "KI": 1, "IC50": 2, "REMARK": 3, "REMARK_IC50": 3}
    combined["priority"] = combined["affinity_type"].map(priority).fillna(99)
    combined = combined.sort_values("priority")

    all_path = DATA_DIR / "all_binding_affinity.csv"
    combined.to_csv(all_path, index=False)
    _log.info("Final: %d total affinity records → %s", len(combined), all_path)

    # Summary
    print(f"\n=== Binding Affinity Summary ===")
    print(f"RCSB records:       {len(rcsb) if not rcsb.empty else 0}")
    print(f"PDBe records:       {len(pdbe_recs)}")
    print(f"ChEMBL records:     {len([r for r in all_records if r['source'] == 'chembl'])}")
    print(f"REMARK records:     {len([r for r in all_records if r['source'] == 'mmcif_remark'])}")
    print(f"Total (unique PDB): {combined['pdb_id'].nunique()} structures with affinity")
    print(f"pKd range:          {combined['experimental_pkd'].min():.1f} – {combined['experimental_pkd'].max():.1f}")
    print(f"Affinity types:     {combined['affinity_type'].value_counts().to_dict()}")

    # Show structures with Kd/Ki (most reliable for calibration)
    kd_ki = combined[combined["affinity_type"].str.upper().isin(["KD", "KI"])]
    print(f"\nKd/Ki-specific records ({len(kd_ki)}):")
    print(kd_ki[["pdb_id","affinity_type","value","unit","experimental_pkd","source"]].to_string())


if __name__ == "__main__":
    main()
