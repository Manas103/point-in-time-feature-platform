import pandas as pd

from ptf.asof import asof_join, naive_asof_join_buggy
from ptf.feature_spec import FeatureDefinition

SPEC = FeatureDefinition("f", "qty", "instrument_id", "event_time_ns", "sum", 1)


def _frame():
    return pd.DataFrame({
        "instrument_id": [1, 1, 1],
        "event_time_ns": [1000, 2000, 3000],
        "qty": [10, 20, 30],
    })


def test_exact_match_included_backward():
    queries = pd.DataFrame({"instrument_id": [1], "query_time_ns": [2000]})
    out = asof_join(_frame(), queries, SPEC)
    assert out.loc[0, "event_time_ns"] == 2000


def test_query_before_any_event_has_no_match():
    queries = pd.DataFrame({"instrument_id": [1], "query_time_ns": [500]})
    out = asof_join(_frame(), queries, SPEC)
    assert out.loc[0, "event_time_ns"] != out.loc[0, "event_time_ns"]  # NaN


def test_naive_nearest_can_pick_a_future_event():
    queries = pd.DataFrame({"instrument_id": [1], "query_time_ns": [1999]})
    correct = asof_join(_frame(), queries, SPEC)
    naive = naive_asof_join_buggy(_frame(), queries, SPEC)
    assert correct.loc[0, "event_time_ns"] == 1000   # backward: last event at/before 1999
    assert naive.loc[0, "event_time_ns"] == 2000      # nearest: 1ns away, but in the future
