"""The one feature definition both compiler backends read from.

A FeatureDefinition names a windowed aggregation (which column, grouped by
which key, ordered by which event-time column, over how many trailing
events, aggregated how) and nothing else. It contains no Flink code and no
pandas code. compile_streaming.to_flink_sql() and compile_batch.to_pandas()
each read these same five fields to build their own execution plan, so there
is exactly one place a feature's definition lives, not one copy per engine.

version() hashes the definition's own fields, not the data. Two
FeatureDefinition instances with identical fields always hash to the same
version string, and a definition is never mutated in place (the dataclass is
frozen) -- changing a window size is a new FeatureDefinition with a new
version, not an edit, which is what "every training set resolves to a pinned
definition version" needs to be true.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

_ALLOWED_AGG = {"sum", "count", "mean"}


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    source_field: str      # column being aggregated, e.g. "qty"
    group_key: str          # partition key, e.g. "instrument_id"
    order_key: str          # event-time column, e.g. "event_time_ns"
    agg: str                # one of _ALLOWED_AGG
    window_events: int       # trailing window size, in events (ROWS BETWEEN window_events-1 PRECEDING)

    def __post_init__(self) -> None:
        if self.agg not in _ALLOWED_AGG:
            raise ValueError(f"unsupported agg {self.agg!r}, must be one of {_ALLOWED_AGG}")
        if self.window_events < 1:
            raise ValueError("window_events must be >= 1")

    def version(self) -> str:
        spec = (
            f"{self.name}|{self.source_field}|{self.group_key}|"
            f"{self.order_key}|{self.agg}|{self.window_events}"
        )
        return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


# The single feature definition this project compiles to both paths.
TRAILING_QTY_SUM = FeatureDefinition(
    name="trailing_qty_sum_20",
    source_field="qty",
    group_key="instrument_id",
    order_key="event_time_ns",
    agg="sum",
    window_events=20,
)
