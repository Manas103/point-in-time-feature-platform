"""Lookahead-leak detector: given the output of an as-of join, flags every
row whose attached event_time is strictly after that row's own query_time.
One row flagged is one proven leak; it does not matter how small the gap is.
"""
from __future__ import annotations

import pandas as pd

from ptf.feature_spec import FeatureDefinition


def find_lookahead_violations(
    joined_df: pd.DataFrame,
    spec: FeatureDefinition,
    query_time_col: str = "query_time_ns",
) -> pd.DataFrame:
    """Returns the subset of rows where the joined feature's event_time is
    after the query time -- i.e. the join used information from the future
    relative to the point it claims to describe. Rows with no matched event
    at all (NaN order_key, a query before any event existed) are not leaks
    and are excluded.
    """
    has_match = joined_df[spec.order_key].notna()
    violation = has_match & (joined_df[spec.order_key] > joined_df[query_time_col])
    return joined_df.loc[violation]
