"""Batch backfill compiler backend: FeatureDefinition -> pandas rolling computation.

Reads the same fields compile_streaming.to_flink_sql() reads (group_key,
order_key, source_field, agg, window_events) and builds the equivalent
trailing-window aggregate with pandas, sorted within each group by the
event-time column. For the sum/count aggregations this project uses, this is
exact integer arithmetic over the identical per-partition event order a
correctly-windowed Flink OVER clause uses, which is what makes a row-for-row
diff against the streaming output meaningful rather than approximate.
"""
from __future__ import annotations

import pandas as pd

from ptf.feature_spec import FeatureDefinition


def compute_batch_feature(events: pd.DataFrame, spec: FeatureDefinition) -> pd.DataFrame:
    """events must have columns [spec.group_key, spec.order_key, spec.source_field].

    Returns a DataFrame with columns [group_key, order_key, spec.name], one
    row per input event, sorted the same way the input was received.
    """
    sorted_events = events.sort_values([spec.group_key, spec.order_key], kind="mergesort")
    grouped = sorted_events.groupby(spec.group_key, sort=False)[spec.source_field]

    if spec.agg == "sum":
        feature = grouped.rolling(spec.window_events, min_periods=1).sum()
    elif spec.agg == "count":
        feature = grouped.rolling(spec.window_events, min_periods=1).count()
    elif spec.agg == "mean":
        feature = grouped.rolling(spec.window_events, min_periods=1).mean()
    else:  # pragma: no cover - guarded in FeatureDefinition.__post_init__
        raise ValueError(f"unsupported agg {spec.agg!r}")

    feature = feature.reset_index(level=0, drop=True)
    out = sorted_events[[spec.group_key, spec.order_key]].copy()
    out[spec.name] = feature.values
    if spec.agg in ("sum", "count"):
        out[spec.name] = out[spec.name].astype("int64")
    return out.reset_index(drop=True)
