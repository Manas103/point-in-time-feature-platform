"""A SignalSpec is a FeatureDefinition plus the three things a trading
signal needs that a feature does not: which asset class it trades (so a
realistic, asset-class-specific half-spread cost can be charged), a risk
budget (so the spec produces a position, not a bare number), and a
rebalance-lag convention.

``to_feature_definition()`` is the whole point: it hands the SAME
FeatureDefinition object the rest of this project already compiles to a
PyFlink streaming job and a pandas-over-Iceberg batch backfill (see
feature_spec.py, streaming_flink.py, batch_pandas.py). A SignalSpec adds no
new compiler backend of its own; "one spec, research and live" means the
signal's underlying trailing-sum feature is compiled by the exact code path
already proven to match row for row across 12,000,000 values (see
README "Extension: multi-asset signal implementation" for why that proof is
cited here rather than re-run at signal scale).
"""
from __future__ import annotations

from dataclasses import dataclass

from ptf.feature_spec import FeatureDefinition

# Asset-class-specific half-spread assumptions (fraction of notional per
# rebalance), roughly ordered by real-world liquidity: FX majors trade
# tightest, government rates next, equities wider, credit widest.
HALF_SPREAD_BY_ASSET_CLASS = {
    "FX": 0.00002,
    "RATE": 0.00005,
    "EQUITY": 0.0005,
    "CREDIT": 0.001,
}

ASSET_CLASSES = tuple(HALF_SPREAD_BY_ASSET_CLASS.keys())


@dataclass(frozen=True)
class SignalSpec:
    name: str
    asset_class: str          # one of ASSET_CLASSES
    source_field: str          # column being aggregated, e.g. "ret"
    group_key: str             # partition key, e.g. "instrument_id"
    order_key: str             # event-time column, e.g. "event_time_ns"
    window_events: int          # trailing window size, in events
    risk_budget: float           # target position notional per unit signal strength
    implementation_lag_events: int = 1  # one-bar lag by default

    def __post_init__(self) -> None:
        if self.asset_class not in HALF_SPREAD_BY_ASSET_CLASS:
            raise ValueError(f"unknown asset_class {self.asset_class!r}")
        if self.window_events < 1:
            raise ValueError("window_events must be >= 1")
        if self.implementation_lag_events < 0:
            raise ValueError("implementation_lag_events must be >= 0")

    def half_spread(self) -> float:
        return HALF_SPREAD_BY_ASSET_CLASS[self.asset_class]

    def to_feature_definition(self) -> FeatureDefinition:
        """The exact same FeatureDefinition machinery the rest of this
        project compiles to both engines, reused unchanged for the
        signal's trailing-mean-return feature. "mean", not "sum": returns
        are small fractional numbers, and compute_batch_feature casts a
        "sum"/"count" result to int64 (correct for this project's original
        integer-qty feature), which would silently truncate a fractional
        return signal to zero. "mean" is also the more standard momentum
        construction (average return over the window, not its sum)."""
        return FeatureDefinition(
            name=self.name, source_field=self.source_field, group_key=self.group_key,
            order_key=self.order_key, agg="mean", window_events=self.window_events,
        )
