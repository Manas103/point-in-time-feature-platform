import pytest

from ptf.signal_spec import ASSET_CLASSES, HALF_SPREAD_BY_ASSET_CLASS, SignalSpec


def test_to_feature_definition_uses_mean_not_sum():
    spec = SignalSpec(
        name="sig", asset_class="FX", source_field="ret", group_key="instrument_id",
        order_key="event_time_ns", window_events=10, risk_budget=1.0,
    )
    fd = spec.to_feature_definition()
    assert fd.agg == "mean"
    assert fd.name == "sig"
    assert fd.window_events == 10


def test_unknown_asset_class_rejected():
    with pytest.raises(ValueError):
        SignalSpec(
            name="sig", asset_class="CRYPTO", source_field="ret", group_key="instrument_id",
            order_key="event_time_ns", window_events=10, risk_budget=1.0,
        )


def test_every_asset_class_has_a_half_spread():
    for asset_class in ASSET_CLASSES:
        assert asset_class in HALF_SPREAD_BY_ASSET_CLASS
        assert HALF_SPREAD_BY_ASSET_CLASS[asset_class] > 0.0


def test_half_spread_matches_lookup():
    spec = SignalSpec(
        name="sig", asset_class="CREDIT", source_field="ret", group_key="instrument_id",
        order_key="event_time_ns", window_events=10, risk_budget=1.0,
    )
    assert spec.half_spread() == HALF_SPREAD_BY_ASSET_CLASS["CREDIT"]
