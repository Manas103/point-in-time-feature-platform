import pytest

from ptf.feature_spec import FeatureDefinition


def test_version_is_stable_for_identical_definitions():
    a = FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 20)
    b = FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 20)
    assert a.version() == b.version()


def test_version_changes_with_any_field():
    base = FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 20)
    variants = [
        FeatureDefinition("g", "qty", "instrument_id", "event_time_ns", "sum", 20),
        FeatureDefinition("f", "price", "instrument_id", "event_time_ns", "sum", 20),
        FeatureDefinition("f", "qty", "venue_id", "event_time_ns", "sum", 20),
        FeatureDefinition("f", "qty", "instrument_id", "arrival_ns", "sum", 20),
        FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "count", 20),
        FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 21),
    ]
    versions = {base.version()} | {v.version() for v in variants}
    assert len(versions) == len(variants) + 1


def test_rejects_unsupported_agg():
    with pytest.raises(ValueError):
        FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "median", 20)


def test_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 0)


def test_frozen():
    spec = FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 20)
    with pytest.raises(Exception):
        spec.window_events = 5
