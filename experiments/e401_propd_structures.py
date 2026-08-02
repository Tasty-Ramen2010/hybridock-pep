"""E401 — receptor structures for the ProP-PD specificity benchmark (E400).

One structure per domain bait. Sources the AlphaFold DB model for the bait's
UniProt accession and crops it to the bait's residue range, which is the same
representation CliPepPI used (AlphaFold2 models, receptor truncated to the
binding domain) and the only one with uniform coverage — PDB coverage across
these 27 domains is patchy and would silently mix crystal and modelled inputs.

Cropping matters for more than tidiness. A full-length chain drags the docking
box out to the whole protein, and E400's baits are ordered domains excised from
proteins that are mostly disordered; the surrounding coil is not part of the
binding site and only adds decoy surface.

Also writes, per domain, the geometric centre and an extent-derived box size,
so E402 does not have to re-derive a site for every pair.

Outputs
-------
    data/propd_structures/<domain_id>.pdb   cropped domain
    data/propd_structures/index.json        {domain_id: {path, site, box, n_res, plddt}}

pLDDT is carried through because it is a real caveat on any result: a domain
modelled at pLDDT 70 is not the same evidence as one at 95, and low-confidence
baits should be separable when the AUC is read.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "propd_specificity.jsonl"
OUTDIR = ROOT / "data" / "propd_structures"
CACHE = ROOT / "data" / "cache" / "afdb"
INDEX = OUTDIR / "index.json"

#: Ask the API for the model URL rather than composing one. AlphaFold DB bumps
#: its file version periodically and old URLs 404 outright — every accession
#: here returned 404 on the `model_v4` pattern once the DB moved to v6. The
#: prediction endpoint returns the current `pdbUrl`, so this keeps working
#: across version bumps.
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

#: Fallback if the API is unreachable: try recent file versions, newest first.
AFDB_FILE = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
AFDB_VERSIONS = (6, 5, 4)

#: Padding added around the domain's bounding box, in Angstroms. The peptide
#: binds at the surface and extends outward, so the box has to contain more
#: than the domain itself.
BOX_PAD_AA = 12.0

#: Upper bound on box edge. driver.py caps grids at 100 A for OOM safety; going
#: above it here would just be silently clamped later.
BOX_MAX_AA = 100.0


def _afdb_urls(acc: str) -> list[str]:
    """Candidate model URLs for an accession, API answer first."""
    urls: list[str] = []
    try:
        with urllib.request.urlopen(AFDB_API.format(acc=acc), timeout=60) as r:
            payload = json.loads(r.read())
        entry = payload[0] if isinstance(payload, list) else payload
        if entry.get("pdbUrl"):
            urls.append(entry["pdbUrl"])
    except Exception:  # noqa: BLE001 - API down or shape changed; fall through
        pass
    urls += [AFDB_FILE.format(acc=acc, ver=v) for v in AFDB_VERSIONS]
    return urls


def fetch_afdb(acc: str) -> Path | None:
    """Download (and cache) the AlphaFold DB model for a UniProt accession."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"AF-{acc}-F1.pdb"
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    errors = []
    for url in _afdb_urls(acc):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{url.rsplit('/', 1)[-1]}: {exc}")
            continue
        if dest.stat().st_size > 1000:
            return dest
    print(f"    ! AFDB fetch failed for {acc}: {errors[0] if errors else 'empty file'}")
    return None


def crop(model: Path, start: int, stop: int, out: Path) -> dict | None:
    """Write residues [start, stop] of `model` to `out`.

    Returns geometry for the docking step, or None if the range is not present
    (a truncated or mis-numbered model).
    """
    kept: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    plddts: list[float] = []
    residues: set[int] = set()

    for line in model.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        if not (start <= resi <= stop):
            continue
        kept.append(line)
        residues.add(resi)
        try:
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
            # AlphaFold stores per-residue pLDDT in the B-factor column.
            plddts.append(float(line[60:66]))
        except ValueError:
            continue

    if len(residues) < 10:
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\nEND\n")

    centre = (
        round(sum(xs) / len(xs), 2),
        round(sum(ys) / len(ys), 2),
        round(sum(zs) / len(zs), 2),
    )
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    box = round(min(extent + BOX_PAD_AA, BOX_MAX_AA), 1)
    return {
        "path": str(out.relative_to(ROOT)),
        "site": list(centre),
        "box": box,
        "n_res": len(residues),
        "plddt": round(sum(plddts) / len(plddts), 1) if plddts else None,
    }


def main() -> int:
    if not PAIRS.is_file():
        print(f"missing {PAIRS} — run e400_propd_specificity_data.py first", file=sys.stderr)
        return 2

    domains: dict[str, dict] = {}
    for line in PAIRS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        domains.setdefault(r["domain_id"], r)

    print(f"=== E401 — structures for {len(domains)} ProP-PD domains ===")
    index: dict[str, dict] = {}
    failed: list[str] = []

    for did, d in sorted(domains.items()):
        model = fetch_afdb(d["domain_uniprot"])
        if model is None:
            failed.append(did)
            continue
        geo = crop(model, d["domain_start"], d["domain_stop"], OUTDIR / f"{did}.pdb")
        if geo is None:
            print(f"    ! {did}: residues {d['domain_start']}-{d['domain_stop']} not in model")
            failed.append(did)
            continue
        geo.update(
            uniprot=d["domain_uniprot"],
            start=d["domain_start"],
            stop=d["domain_stop"],
        )
        index[did] = geo
        print(
            f"  {did:42s} {geo['n_res']:4d} res  box {geo['box']:5.1f} A  "
            f"pLDDT {geo['plddt']}"
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"\n  wrote {INDEX} ({len(index)} domains)")
    if failed:
        print(f"  FAILED ({len(failed)}): {', '.join(failed)}")
    low = [k for k, v in index.items() if (v["plddt"] or 100) < 70]
    if low:
        print(f"  low-confidence models (pLDDT < 70), read their results with care: {', '.join(low)}")
    return 0 if index else 1


if __name__ == "__main__":
    raise SystemExit(main())
