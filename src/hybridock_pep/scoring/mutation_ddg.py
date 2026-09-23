"""Single-mutation ΔΔG head — the congeneric/affinity-maturation regime the absolute scorer cannot see.

The absolute scorer is architecturally blind to single-residue changes. Measured on the OppA K-X-K
series (19 tripeptides, one varying position, 3.00 kcal/mol experimental span): the median complex-level
feature moves only **14% of its dataset-wide sd** across a one-residue change, and only 5 of 59 features
move more than half. The scorer manages r=+0.183 there, and adding sequence descriptors makes it *worse*
(r=−0.056) — see experiments/e409. No refitting can fix that: the inputs barely change.

This module is the separate head for that regime, trained on **SKEMPI v2** (4956 single mutations across
319 complexes; experiments/e410). It predicts ΔΔG = ΔG(mutant) − ΔG(wild-type), sign convention
**positive = the mutation WEAKENS binding**.

Measured performance, leave-COMPLEX-out CV (all mutations of a complex held out together):

    Pearson r = 0.385 · Spearman = 0.442 · RMSE 1.617 · MAE 1.062 kcal/mol
    mean-predictor MAE 1.216 · permutation null r = −0.007 ± 0.016

Two honesty notes that must travel with those numbers:

  * **Leave-complex-out is harder than most published splits.** A random split lets mutations of the
    same complex sit on both sides, which inflates r substantially. Published figures in the r≈0.40–0.50
    band (mCSM-PPI2 0.40; a 2020 eight-method comparison peaked at 0.49) are generally *not* grouped this
    way, so ours is not directly comparable to them and should not be advertised as beating them.
  * **This is a structure-free property model**, not a physics calculation. It sees residue identities
    and an interface-location class, never the mutant structure. That is what makes it instant and
    portable to any peptide variant; it is also why it cannot resolve mechanism-specific effects.

Accuracy is strongly location-dependent, and the caller gets told which regime they are in:

    COR  r=0.294  MAE 1.444    SUP  r=0.293  MAE 1.106    RIM  r=0.200  MAE 0.826
    INT  r=0.077  MAE 0.693    SUR  r=0.088  MAE 0.346

External check on a system absent from SKEMPI (OppA K-X-K, peptide rather than protein–protein, ITC
rather than SPR): r=+0.371, but p=0.129 at n=18 — **suggestive, not significant**. Do not cite it as
validation.

What this head is NOT for: it does not improve absolute ΔG. Feeding its in-silico alanine scan to the
absolute model as features is measurably null (Δr = −0.002 against ±0.008 seed noise, with noise-column
and shuffled controls both also negative — experiments/e411). Use it where a reference peptide exists:
ranking variants of a known binder, affinity maturation, and hot-spot triage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from hybridock_pep._paths import data_file

logger = logging.getLogger(__name__)

#: ΔΔG above which a position is called a hot spot (kcal/mol).
HOT_SPOT_THRESHOLD = 1.0
#: Interface location classes SKEMPI annotates, in the model's feature order.
LOCATIONS = ("COR", "SUP", "RIM", "INT", "SUR")
#: Per-location MAE from the leave-complex-out CV (experiments/e410), for honest error bars.
LOCATION_MAE = {"COR": 1.444, "SUP": 1.106, "RIM": 0.826, "INT": 0.693, "SUR": 0.346}
_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

_VOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
        "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
        "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
       "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7,
       "W": -0.9, "Y": -1.3, "V": 4.2}
_CHG = {a: (1.0 if a in "KR" else -1.0 if a in "DE" else 0.5 if a == "H" else 0.0)
        for a in _STANDARD_AA}
_FLEX = {"A": 0.36, "R": 0.53, "N": 0.46, "D": 0.51, "C": 0.35, "Q": 0.49, "E": 0.50, "G": 0.54,
         "H": 0.32, "I": 0.46, "L": 0.37, "K": 0.47, "M": 0.30, "F": 0.31, "P": 0.51, "S": 0.51,
         "T": 0.44, "W": 0.31, "Y": 0.42, "V": 0.39}
_POLAR = {"A": 0.046, "R": 0.291, "N": 0.134, "D": 0.105, "C": 0.128, "Q": 0.180, "E": 0.151,
          "G": 0.0, "H": 0.230, "I": 0.186, "L": 0.186, "K": 0.219, "M": 0.221, "F": 0.290,
          "P": 0.131, "S": 0.062, "T": 0.108, "W": 0.409, "Y": 0.298, "V": 0.140}
_HYD = {"A": 1.99, "V": 2.00, "L": 2.28, "I": 2.15, "F": -0.76, "W": -5.88, "M": -1.48, "S": -5.06,
        "T": -4.88, "C": -1.24, "N": -9.68, "Q": -9.38, "Y": -6.11, "H": -10.27, "D": -6.70,
        "E": -6.47, "K": -4.29, "R": -10.90, "P": -2.10, "G": 0.0}
_AROM = set("FWYH")
_DON = {"R": 2, "K": 1, "W": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1, "C": 1}
_ACC = {"D": 2, "E": 2, "N": 1, "Q": 1, "S": 1, "T": 1, "Y": 1, "H": 1}


@dataclass(frozen=True)
class MutationEffect:
    """Predicted effect of one point mutation.

    Attributes:
        position: 1-based index into the peptide sequence.
        wild_type: Original one-letter residue.
        mutant: Substituted one-letter residue.
        ddg: Predicted ΔΔG in kcal/mol; positive means the mutation weakens binding.
        expected_error: Typical absolute error for this interface location (kcal/mol), from the
            leave-complex-out CV. Report it — a bare ΔΔG from this head overstates its precision.
        location: Interface location class assumed for the position.
    """

    position: int
    wild_type: str
    mutant: str
    ddg: float
    expected_error: float
    location: str

    @property
    def is_hot_spot(self) -> bool:
        """True if the predicted cost exceeds :data:`HOT_SPOT_THRESHOLD`."""
        return self.ddg > HOT_SPOT_THRESHOLD


def mutation_features(wild_type: str, mutant: str, location: str = "COR") -> list[float]:
    """Featurise a single point mutation for the ΔΔG head.

    Carries the property deltas, both endpoints (context matters — mutating a buried Trp is not the
    same event as mutating a Ser), categorical special cases (Pro/Gly, charge reversal), and a
    one-hot interface location.

    Args:
        wild_type: Original residue, one-letter.
        mutant: Substituted residue, one-letter.
        location: SKEMPI interface class; unrecognised values yield an all-zero location block,
            which the model treats as an unknown environment rather than failing.

    Returns:
        Feature vector matching the trained artifact's ``feature_names`` order.

    Raises:
        ValueError: If either residue is not one of the 20 standard amino acids.
    """
    for residue in (wild_type, mutant):
        if residue not in _VOL:
            raise ValueError(f"non-standard amino acid {residue!r}; expected one of {_STANDARD_AA}")
    d_vol = _VOL[mutant] - _VOL[wild_type]
    d_kd = _KD[mutant] - _KD[wild_type]
    feats = [
        d_vol, d_kd, _CHG[mutant] - _CHG[wild_type], _FLEX[mutant] - _FLEX[wild_type],
        _POLAR[mutant] - _POLAR[wild_type], _HYD[mutant] - _HYD[wild_type],
        float(mutant in _AROM) - float(wild_type in _AROM),
        _DON.get(mutant, 0) - _DON.get(wild_type, 0), _ACC.get(mutant, 0) - _ACC.get(wild_type, 0),
        _VOL[wild_type], _KD[wild_type], _CHG[wild_type], _FLEX[wild_type], _POLAR[wild_type],
        _HYD[wild_type], float(wild_type in _AROM),
        _VOL[mutant], _KD[mutant], _CHG[mutant], _FLEX[mutant], _POLAR[mutant], _HYD[mutant],
        float(mutant in _AROM),
        float(mutant == "A"), float(mutant == "G"), float(wild_type == "G"), float(mutant == "P"),
        float(wild_type == "P"), float(_CHG[wild_type] * _CHG[mutant] < 0),
        float(_CHG[wild_type] != 0 and _CHG[mutant] == 0),
        float(_CHG[wild_type] == 0 and _CHG[mutant] != 0),
        abs(d_vol), abs(d_kd), float(_KD[wild_type] > 0 and _KD[mutant] > 0),
    ]
    return feats + [float(location == c) for c in LOCATIONS]


@lru_cache(maxsize=1)
def _load_model(artifact: str | None = None):
    """Load the trained SKEMPI artifact, or None if it is not installed.

    Returns:
        The unpickled artifact dict, or None when the file is absent or unreadable — callers degrade
        to a no-op rather than crashing a docking run over an optional scoring extra.
    """
    path = Path(artifact) if artifact else data_file("skempi_ddg_model.joblib")
    if not path.exists():
        logger.debug("SKEMPI ΔΔG artifact not found at %s; mutation scoring disabled", path)
        return None
    try:
        import joblib

        return joblib.load(path)
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        logger.warning("could not load SKEMPI ΔΔG artifact %s: %s", path, exc)
        return None


def predict_ddg(wild_type: str, mutant: str, location: str = "COR",
                artifact: str | Path | None = None) -> float | None:
    """Predict ΔΔG for one point mutation.

    Args:
        wild_type: Original residue, one-letter.
        mutant: Substituted residue, one-letter.
        location: Interface location class (see :data:`LOCATIONS`).
        artifact: Override path to the trained model.

    Returns:
        ΔΔG in kcal/mol (positive = weaker binding), or None if the model is not installed.

    Raises:
        ValueError: If either residue is not a standard amino acid.
    """
    art = _load_model(str(artifact) if artifact else None)
    if art is None:
        return None
    x = np.array([mutation_features(wild_type, mutant, location)], float)
    return float(art["model"].predict(x)[0])


def scan_variants(reference: str, variants: list[str], location: str = "COR",
                  artifact: str | Path | None = None) -> list[MutationEffect] | None:
    """Rank single-mutation variants of a reference peptide.

    This is the deployable use of the head: the reference supplies the per-target offset that the
    absolute scorer cannot predict, and the head supplies the variant-to-variant differences that the
    absolute scorer cannot see.

    Args:
        reference: Wild-type peptide sequence.
        variants: Sequences differing from ``reference`` at exactly one position.
        location: Interface location class assumed for the mutated positions.
        artifact: Override path to the trained model.

    Returns:
        Effects sorted most-stabilising first (most negative ΔΔG), or None if the model is absent.

    Raises:
        ValueError: If a variant is not the same length as the reference or does not differ from it
            at exactly one position.
    """
    art = _load_model(str(artifact) if artifact else None)
    if art is None:
        return None
    out = []
    for variant in variants:
        if len(variant) != len(reference):
            raise ValueError(f"variant {variant!r} length {len(variant)} != reference "
                             f"{len(reference)}; this head handles single substitutions only")
        diffs = [i for i, (a, b) in enumerate(zip(reference, variant)) if a != b]
        if len(diffs) != 1:
            raise ValueError(f"variant {variant!r} differs from reference at {len(diffs)} positions; "
                             "expected exactly 1")
        i = diffs[0]
        ddg = predict_ddg(reference[i], variant[i], location, artifact)
        out.append(MutationEffect(position=i + 1, wild_type=reference[i], mutant=variant[i],
                                  ddg=float(ddg), location=location,
                                  expected_error=LOCATION_MAE.get(location, 1.062)))
    return sorted(out, key=lambda e: e.ddg)


def alanine_scan(sequence: str, location: str = "COR",
                 artifact: str | Path | None = None) -> list[MutationEffect] | None:
    """In-silico alanine scan: predicted cost of mutating each residue to alanine.

    Useful as a hot-spot triage map (which residues the interface likely depends on). Note this is a
    *prior*, not a measurement — as features for absolute ΔΔG prediction it measured null
    (experiments/e411), so treat the output as a ranking aid rather than an energy.

    Args:
        sequence: Peptide sequence.
        location: Interface location class assumed for every position.
        artifact: Override path to the trained model.

    Returns:
        Effects in sequence order for every non-alanine standard residue, or None if the model is
        absent. Alanine positions and non-standard residues are skipped.
    """
    art = _load_model(str(artifact) if artifact else None)
    if art is None:
        return None
    rows, meta = [], []
    for i, aa in enumerate(sequence):
        if aa == "A" or aa not in _STANDARD_AA:
            continue
        rows.append(mutation_features(aa, "A", location))
        meta.append((i + 1, aa))
    if not rows:
        return []
    ddgs = art["model"].predict(np.array(rows, float))
    err = LOCATION_MAE.get(location, 1.062)
    return [MutationEffect(position=p, wild_type=aa, mutant="A", ddg=float(d),
                           expected_error=err, location=location)
            for (p, aa), d in zip(meta, ddgs)]
