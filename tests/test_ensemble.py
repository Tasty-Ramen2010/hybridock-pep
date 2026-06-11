"""Unit tests for the geometry+Vina ensemble scorer (SCORE-ENS)."""
from __future__ import annotations

import numpy as np

from hybridock_pep.scoring.ensemble import (
    GEOMETRY_FEATURES,
    EnsembleCalibration,
    fit_ensemble_calibration,
    score,
)


def _toy_records(n: int = 40, seed: int = 0) -> list[dict]:
    """Synthetic complexes where ΔG depends on geometry + vina with independent noise."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        feats = {f: float(rng.normal()) for f in GEOMETRY_FEATURES}
        geo_signal = feats["bsa_hyd"] - 0.5 * feats["poc_eis"]
        vina = float(rng.normal())
        y = -8.0 + 1.5 * geo_signal + 1.2 * vina + rng.normal(scale=0.3)
        recs.append({**feats, "vina": vina, "vina_inter": vina - 0.2, "y": y})
    return recs


def test_fit_returns_valid_calibration() -> None:
    cal = fit_ensemble_calibration(_toy_records(), blend=0.5, vina_mode="total")
    assert cal.feature_names == GEOMETRY_FEATURES
    assert len(cal.geo_weights) == len(GEOMETRY_FEATURES)
    assert cal.vina_mode == "total"
    assert 0.0 <= cal.blend <= 1.0
    assert cal.y_std > 0


def test_score_is_finite_and_in_range() -> None:
    recs = _toy_records()
    cal = fit_ensemble_calibration(recs, blend=0.5)
    s = score(recs[0], recs[0]["vina"], cal)
    assert np.isfinite(s)
    # blended estimate should land within the ΔG envelope, not wildly outside
    assert cal.y_mean - 6 * cal.y_std < s < cal.y_mean + 6 * cal.y_std


def test_ensemble_correlates_with_truth() -> None:
    recs = _toy_records(n=60)
    cal = fit_ensemble_calibration(recs, blend=0.5)
    preds = np.array([score(r, r["vina"], cal) for r in recs])
    y = np.array([r["y"] for r in recs])
    r = np.corrcoef(preds, y)[0, 1]
    assert r > 0.7  # in-sample fit on a well-specified model should be strong


def test_inter_mode_requires_vina_inter() -> None:
    cal = fit_ensemble_calibration(_toy_records(), vina_mode="inter")
    # supplying vina_inter works
    s = score(_toy_records()[0], 0.0, cal, vina_inter=-1.0)
    assert np.isfinite(s)
    # omitting it raises
    try:
        score(_toy_records()[0], 0.0, cal)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when vina_inter missing in inter mode")


def test_calibration_roundtrip(tmp_path) -> None:
    cal = fit_ensemble_calibration(_toy_records(), blend=0.6)
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = EnsembleCalibration.load(p)
    assert loaded.blend == cal.blend
    assert loaded.geo_weights == cal.geo_weights
    r0 = _toy_records()[0]
    assert score(r0, r0["vina"], cal) == score(r0, r0["vina"], loaded)
