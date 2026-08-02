"""E400 — build a binary domain-peptide specificity benchmark from ProP-PD.

Ora Schueler-Furman's review (2026-07-31) suggested testing this protocol on
binding-specificity prediction, a binary task, using the CliPepPI datasets.
CliPepPI's own curated splits are not published as downloadable files, so this
rebuilds the ProP-PD benchmark from the primary source it was drawn from.

Source
------
Benz et al., "Proteome-scale mapping of binding sites in the unstructured
regions of the human proteome", Mol Syst Biol 18(1):e10584 (2022).
Open access; EV Dataset 9 ("HD2 Combined") carries every ProP-PD interaction
with a confidence grade, the 16-mer peptide, and a bait string that encodes the
domain and its residue range. Downloaded from EuropePMC rather than committed:
it is 24 MB of third-party supplementary material.

    bait                            Hit               protein_accession
    YAP1_WW_1_P46937_169_207        PPGPPPPYHAHAHLHH  Q8NEA6
    ^gene ^domain ^idx ^uniprot ^start ^stop

Which is exactly a (receptor domain, peptide) pair.

Negatives
---------
Phage display reports binders, so true non-binders are unknown — the same
problem CliPepPI describes. Their construction is reproduced here: pair a
peptide with a domain it is not known to bind, keeping the pair only if

  (i)  the decoy domain is <50% identical to the peptide's real domain, and
  (ii) the peptide is <80% identical to every peptide known to bind that
       decoy domain.

Without both filters a "negative" can be a near-duplicate of a positive and the
benchmark measures nothing.

Reference numbers on ProP-PD, for whatever this protocol is eventually scored
against (CliPepPI paper, Table 1): CliPepPI AUC 0.72, AlphaFold2 actifpTM 0.90.

Output
------
`data/propd_specificity.jsonl`, one JSON object per pair:

    {"domain_id", "domain_gene", "domain_type", "domain_uniprot",
     "domain_start", "domain_stop", "peptide", "peptide_source",
     "label": 1 | 0, "confidence"}

This script only builds the *dataset*. Scoring it is a separate step: each
domain needs a structure, and each pair a dock + score.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "propd_specificity.jsonl"
CACHE = ROOT / "data" / "cache" / "benz2022_supplementary.zip"

SUPPL_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8769072/supplementaryFiles"
INTERACTIONS_XLSX = "MSB-18-e10584-s016.xlsx"
INTERACTIONS_SHEET = "HD2 Combined"

#: ProP-PD confidence grades to keep. The paper grades 1-4; CliPepPI used
#: "high-confidence interactions with known binding motifs", so require the top
#: grade *and* a curated motif annotation.
MIN_CONFIDENCE = 4

#: CliPepPI's negative-pair filters (Methods, "Negative sample generation").
MAX_DECOY_DOMAIN_IDENTITY = 0.50
MAX_PEPTIDE_IDENTITY_TO_BINDERS = 0.80

#: Negatives per positive. 1:1 keeps AUC readable and matches the ~1:2
#: positive:total ratio the CliPepPI ProP-PD set reports (282 of 782).
NEGATIVES_PER_POSITIVE = 2

# bait strings look like GENE_DOMAIN_[idx_]UNIPROT_START_STOP, where DOMAIN can
# itself contain underscores ("Alix-V-domain_", "PTB_"). Anchor on the UniProt
# accession + two integers at the end, which are unambiguous.
_BAIT_RE = re.compile(
    r"^(?P<gene>[^_]+)_(?P<domain>.+?)_+(?P<uniprot>[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})_(?P<start>\d+)_(?P<stop>\d+)$"
)


def _download_supplementary() -> Path:
    """Fetch the Benz 2022 supplementary bundle, caching it under data/cache/."""
    if CACHE.is_file() and CACHE.stat().st_size > 1_000_000:
        print(f"  cached: {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {SUPPL_URL}")
    with urllib.request.urlopen(SUPPL_URL, timeout=180) as r:
        CACHE.write_bytes(r.read())
    print(f"  saved {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    return CACHE


def _load_interactions():
    """Return the ProP-PD interaction table as a DataFrame."""
    import pandas as pd

    with zipfile.ZipFile(_download_supplementary()) as z:
        blob = z.read(INTERACTIONS_XLSX)
    # header=1: row 0 is the table's prose caption, not column names.
    return pd.read_excel(BytesIO(blob), sheet_name=INTERACTIONS_SHEET, header=1)


def parse_bait(bait: str) -> dict | None:
    """Split a ProP-PD bait string into its domain fields, or None if malformed."""
    m = _BAIT_RE.match(str(bait).strip())
    if m is None:
        return None
    d = m.groupdict()
    return {
        "domain_id": str(bait).strip(),
        "domain_gene": d["gene"],
        "domain_type": d["domain"].strip("_"),
        "domain_uniprot": d["uniprot"],
        "domain_start": int(d["start"]),
        "domain_stop": int(d["stop"]),
    }


def sequence_identity(a: str, b: str) -> float:
    """Placement-aware identity in [0, 1]: matches at aligned positions.

    Deliberately gap-free. A free-gap metric collapses to
    longest-common-subsequence over the shorter sequence, which scores GGA
    against ACC at 0.33 and merges unrelated peptides into one cluster — the
    exact bug that had to be corrected in this project's own identity trend
    (see experiments/e367_gap_penalized_trend.py). For 16-mers from a phage
    library, position-wise comparison over the best offset is both adequate and
    honest.
    """
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0
    for off in range(len(b) - len(a) + 1):
        best = max(best, sum(1 for i, ch in enumerate(a) if ch == b[off + i]))
    return best / len(b)


def build() -> int:
    df = _load_interactions()
    print(f"  loaded {len(df)} ProP-PD rows, {df['bait'].nunique()} baits")

    conf = df["HD2 confidence"].astype(str).str.strip()
    keep = (conf == str(MIN_CONFIDENCE)) & df["curated_motifs"].notna() & df["Hit"].notna()
    hits = df[keep]
    print(
        f"  {len(hits)} rows at confidence {MIN_CONFIDENCE} with a curated motif "
        f"({hits['bait'].nunique()} baits)"
    )

    positives: list[dict] = []
    binders_by_domain: dict[str, list[str]] = defaultdict(list)
    domains: dict[str, dict] = {}

    for _, row in hits.iterrows():
        info = parse_bait(row["bait"])
        if info is None:
            continue
        pep = str(row["Hit"]).strip().upper()
        if not pep.isalpha():
            continue
        domains[info["domain_id"]] = info
        binders_by_domain[info["domain_id"]].append(pep)
        positives.append(
            {
                **info,
                "peptide": pep,
                "peptide_source": str(row.get("protein_accession", "")),
                "label": 1,
                "confidence": int(MIN_CONFIDENCE),
            }
        )

    print(f"  {len(positives)} positive pairs across {len(domains)} domains")
    if not positives:
        return 0

    # Domain-vs-domain identity, computed once, over the *domain type* plus
    # gene. Two SH3 domains from different genes are different receptors; the
    # same domain from the same gene is the same receptor.
    domain_ids = list(domains)

    def domains_too_similar(a_id: str, b_id: str) -> bool:
        a, b = domains[a_id], domains[b_id]
        if a["domain_uniprot"] == b["domain_uniprot"]:
            return True
        # No domain sequences in this table, so identity is approximated by
        # domain family: same family (SH3 vs SH3) is the conservative call —
        # treat as too similar and never use as a decoy.
        return a["domain_type"].split("_")[0] == b["domain_type"].split("_")[0]

    import random

    rng = random.Random(0)
    negatives: list[dict] = []
    for pos in positives:
        true_id = pos["domain_id"]
        candidates = [
            d for d in domain_ids
            if d != true_id and not domains_too_similar(true_id, d)
        ]
        rng.shuffle(candidates)
        added = 0
        for decoy_id in candidates:
            if added >= NEGATIVES_PER_POSITIVE:
                break
            binders = binders_by_domain[decoy_id]
            if any(
                sequence_identity(pos["peptide"], b) > MAX_PEPTIDE_IDENTITY_TO_BINDERS
                for b in binders
            ):
                continue
            negatives.append(
                {
                    **domains[decoy_id],
                    "peptide": pos["peptide"],
                    "peptide_source": pos["peptide_source"],
                    "label": 0,
                    "confidence": 0,
                }
            )
            added += 1

    print(f"  {len(negatives)} negative pairs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for rec in positives + negatives:
            fh.write(json.dumps(rec) + "\n")
    print(f"  wrote {OUT} ({len(positives) + len(negatives)} pairs)")
    return len(positives) + len(negatives)


def main() -> int:
    print("=== E400 — ProP-PD domain-peptide specificity benchmark ===")
    try:
        n = build()
    except ImportError as exc:
        print(f"  needs pandas + openpyxl in this env: {exc}", file=sys.stderr)
        return 2
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
