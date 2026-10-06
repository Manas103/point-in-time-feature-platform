"""Event-time as-of joins: for each (instrument, query_time) row in
`queries`, attach the feature value from the latest event at or before
query_time for that instrument. No row in the result may carry information
from an event whose event_time is after that row's query_time -- that is
the lookahead-leak property ptf.leak_check checks for, and the property a
training set depends on for being honest about what was knowable at the
time.

Two implementations are exported on purpose. `asof_join` is correct.
`naive_asof_join_buggy` exists only so ptf.leak_check's 40 seeded scenarios
have something that actually leaks to catch -- a leak checker that has never
been shown to flag a real leak is unproven, see docs/leak_check_output.txt.
"""
from __future__ import annotations

import pandas as pd

from ptf.feature_spec import FeatureDefinition


def asof_join(
    feature_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    spec: FeatureDefinition,
    query_time_col: str = "query_time_ns",
) -> pd.DataFrame:
    """Correct point-in-time join: pandas merge_asof, direction='backward'
    (only events at or before the query time are eligible), grouped by
    instrument via the `by` parameter. Both sides must be sorted by their
    respective time columns per pandas' own requirement for merge_asof,
    which this function enforces rather than assumes.
    """
    # merge_asof requires the "on" column sorted ascending across the WHOLE
    # frame, not just within each `by` group -- sorting only within group
    # (e.g. [group_key, time]) leaves the overall time column non-monotonic
    # whenever groups interleave, and merge_asof raises on that.
    left = queries_df.sort_values(query_time_col, kind="mergesort").reset_index(drop=True)
    right = feature_df.sort_values(spec.order_key, kind="mergesort").reset_index(drop=True)
    return pd.merge_asof(
        left,
        right,
        left_on=query_time_col,
        right_on=spec.order_key,
        by=spec.group_key,
        direction="backward",
        allow_exact_matches=True,
    )


def naive_asof_join_buggy(
    feature_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    spec: FeatureDefinition,
    query_time_col: str = "query_time_ns",
) -> pd.DataFrame:
    """A plausible-looking but wrong join: nearest-in-time instead of
    backward-only. This is a realistic mistake, not a strawman -- merge_asof
    defaults to direction='backward' already, but a direction='nearest' call
    (reaching for "closest value" without thinking about which side of now
    that value is on) is exactly the kind of lookahead bug a leak check
    exists to catch, and it is one line away from the correct call above.
    """
    left = queries_df.sort_values(query_time_col, kind="mergesort").reset_index(drop=True)
    right = feature_df.sort_values(spec.order_key, kind="mergesort").reset_index(drop=True)
    return pd.merge_asof(
        left,
        right,
        left_on=query_time_col,
        right_on=spec.order_key,
        by=spec.group_key,
        direction="nearest",
    )
