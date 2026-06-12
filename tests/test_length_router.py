"""Unit tests for the length-conditional scoring router."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybridock_pep.scoring.ensemble import EnsembleCalibration
from hybridock_pep.scoring.length_router import (
    SHORT_FEATURES,
    SHORT_MAX_LEN,
    LengthRouterCalibration,
    fit_length_router,
    route_score,
)

_CAL_PATH = Path(__file__).resolve().parents[1] / "data" / "calibration_length_router.json"


def _toy_records(n: int = 20) -> list[dict]:
    """Synthetic short + long records with a clean bsa_hyd->y relationship."""
    recs = []
    for i in range(n):
        bsa = float(i)
        recs.append({"length": 6, "y": -4.0 - 0.3 * bsa,
                     "bsa_hyd": bsa, "mj_contact": -10.0 - i, "strength_bur": 0.5 + 0.01 * i})
    for i in range(n):
        recs.append({"length": 14, "y": -9.0, "bsa_hyd": float(i),
                     "mj_contact": -50.0, "strength_bur": 1.0})
    return recs


def test_fit_requires_enough_short_records():
    with pytest.raises(ValueError):
        fit_length_router([{"length": 5, "y": -5.0, "bsa_hyd": 1.0,
                            "mj_contact": -1.0, "strength_bur": 0.1}])


def test_fit_recovers_monotonic_short_signal():
    cal = fit_length_router(_toy_records())
    assert cal.short_max_len == SHORT_MAX_LEN
    assert cal.feature_names == SHORT_FEATURES
    # bsa_hyd drives y negative in the toy data -> its standardized weight must be negative
    assert cal.weights[cal.feature_names.index("bsa_hyd")] < 0
    # predictions for stronger-burial short peptides should be more negative
    strong = route_score({"bsa_hyd": 19.0, "mj_contact": -29.0, "strength_bur": 0.69},
                         vina_total=-5.0, peptide_length=6, router_cal=cal,
                         ensemble_cal=_dummy_ensemble())
    weak = route_score({"bsa_hyd": 0.0, "mj_contact": -10.0, "strength_bur": 0.5},
                       vina_total=-5.0, peptide_length=6, router_cal=cal,
                       ensemble_cal=_dummy_ensemble())
    assert strong < weak


def _dummy_ensemble() -> EnsembleCalibration:
    feats = ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis",
             "bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count", "mj_contact", "strength_bur"]
    return EnsembleCalibration(
        feature_names=feats, geo_intercept=-9.0, geo_weights=[0.0] * len(feats),
        geo_mean=[0.0] * len(feats), geo_std=[1.0] * len(feats),
        geo_pred_mean=-9.0, geo_pred_std=1.0, vina_mean=-5.0, vina_std=1.0,
        blend=1.0, vina_mode="total", y_mean=-9.0, y_std=1.0)


def test_router_dispatches_by_length():
    cal = fit_length_router(_toy_records())
    ens = _dummy_ensemble()
    feats = {f: 1.0 for f in ["poc_n", "poc_f_hyd", "poc_f_arom", "poc_net", "poc_eis",
                              "bsa_hyd", "sasa_hb", "sasa_sb", "arom_cc", "hb_count",
                              "mj_contact", "strength_bur"]}
    # long peptide -> ensemble path (dummy ensemble returns y_mean = -9.0 with zero weights)
    long_dg = route_score(feats, vina_total=-5.0, peptide_length=14, router_cal=cal, ensemble_cal=ens)
    assert math.isclose(long_dg, -9.0, abs_tol=1e-6)
    # short peptide -> sub-model path (not the ensemble constant)
    short_dg = route_score(feats, vina_total=-5.0, peptide_length=6, router_cal=cal, ensemble_cal=ens)
    assert not math.isclose(short_dg, -9.0, abs_tol=1e-6)


def test_boundary_length_routes_short():
    cal = fit_length_router(_toy_records())
    ens = _dummy_ensemble()
    feats = {f: 1.0 for f in SHORT_FEATURES}
    feats.update({f: 0.0 for f in ens.feature_names})
    at = route_score(feats, -5.0, SHORT_MAX_LEN, cal, ens)
    above = route_score(feats, -5.0, SHORT_MAX_LEN + 1, cal, ens)
    assert at != above  # length 8 uses short model, 9 uses ensemble


@pytest.mark.skipif(not _CAL_PATH.exists(), reason="production router calibration not present")
def test_production_calibration_loads():
    cal = LengthRouterCalibration.load(_CAL_PATH)
    assert cal.short_max_len == 8
    assert cal.feature_names == SHORT_FEATURES
    assert len(cal.weights) == len(SHORT_FEATURES)
