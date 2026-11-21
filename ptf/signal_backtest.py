"""The commercial claim: charging a one-bar implementation lag and a
half-spread per rebalance turns some gross-positive candidates negative.

24 candidates = 4 asset classes x 6 trailing-window lengths. Each
candidate's signal is the SAME trailing-sum-of-returns feature the rest of
this project already compiles through `batch_pandas.compute_batch_feature`
(via `SignalSpec.to_feature_definition()`), computed only over the 125
instruments belonging to that candidate's asset class.

Position sizing is risk-budgeted: `position = risk_budget / vol_estimate *
sign(signal)`, where `vol_estimate` is a trailing rolling standard
deviation of the instrument's own return series (same window as the
signal), floored to avoid dividing by near-zero. This produces a position,
not a bare feature number, scaled so that a noisier instrument gets a
smaller position for the same signal strength.

Gross P&L for a rebalance (no lag, no cost): `position_t * ret_{t+1}`, the
return the position earns from the NEXT event, using only information
available at or before event t.

Net P&L lags the position by `implementation_lag_events` (an
"implementation lag": the decision made at t only has effect starting
`lag` events later, not immediately) and subtracts
`half_spread * abs(position_change)` every time the (lagged) position
changes, charged against that same event's return.

A candidate is counted as "flipped" when its gross total return over the
full run is positive but its net total return is negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ptf.batch_pandas import compute_batch_feature
from ptf.signal_generate import generate_events
from ptf.signal_spec import ASSET_CLASSES, SignalSpec

WINDOWS = (10, 20, 30, 50, 75, 100)
RISK_BUDGET = 1.0
VOL_FLOOR = 1e-6


def _events_frame() -> pd.DataFrame:
    rows = list(generate_events())
    return pd.DataFrame(rows, columns=["instrument_id", "seq", "event_time_ns", "asset_class", "ret"])


@dataclass(frozen=True)
class CandidateResult:
    spec: SignalSpec
    gross_total_return: float
    net_total_return: float
    n_rebalances: int
    total_cost: float

    @property
    def flipped(self) -> bool:
        return self.gross_total_return > 0.0 and self.net_total_return < 0.0


def run_candidate(events: pd.DataFrame, spec: SignalSpec) -> CandidateResult:
    class_events = events[events["asset_class"] == spec.asset_class].copy()
    fd = spec.to_feature_definition()

    feature = compute_batch_feature(
        class_events.rename(columns={spec.source_field: fd.source_field}), fd
    )
    class_events = class_events.sort_values([spec.group_key, spec.order_key], kind="mergesort").reset_index(drop=True)
    feature = feature.reset_index(drop=True)
    assert (class_events[spec.group_key].values == feature[spec.group_key].values).all()

    df = class_events[[spec.group_key, spec.order_key, "ret"]].copy()
    df["signal"] = feature[fd.name].values

    df["vol_estimate"] = (
        df.groupby(spec.group_key, sort=False)["ret"]
        .transform(lambda s: s.rolling(spec.window_events, min_periods=1).std())
        .fillna(VOL_FLOOR)
        .clip(lower=VOL_FLOOR)
    )
    df["sign"] = df["signal"].apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    df["position"] = spec.risk_budget / df["vol_estimate"] * df["sign"]

    df["fwd_ret"] = df.groupby(spec.group_key, sort=False)["ret"].shift(-1)
    df["gross_pnl"] = df["position"] * df["fwd_ret"]

    df["lagged_position"] = (
        df.groupby(spec.group_key, sort=False)["position"]
        .shift(spec.implementation_lag_events)
        .fillna(0.0)
    )
    df["position_change"] = (
        df.groupby(spec.group_key, sort=False)["lagged_position"].diff().abs().fillna(0.0)
    )
    half_spread = spec.half_spread()
    df["cost"] = half_spread * df["position_change"]
    df["net_pnl"] = df["lagged_position"] * df["ret"] - df["cost"]

    gross_total = float(df["gross_pnl"].dropna().sum())
    net_total = float(df["net_pnl"].sum())
    n_rebalances = int((df["position_change"] > 0.0).sum())
    total_cost = float(df["cost"].sum())

    return CandidateResult(
        spec=spec, gross_total_return=gross_total, net_total_return=net_total,
        n_rebalances=n_rebalances, total_cost=total_cost,
    )


def run_all_candidates() -> list[CandidateResult]:
    events = _events_frame()
    results: list[CandidateResult] = []
    for asset_class in ASSET_CLASSES:
        for window in WINDOWS:
            spec = SignalSpec(
                name=f"signal_{asset_class.lower()}_{window}",
                asset_class=asset_class, source_field="ret",
                group_key="instrument_id", order_key="event_time_ns",
                window_events=window, risk_budget=RISK_BUDGET,
            )
            results.append(run_candidate(events, spec))
    return results


def summarize(results: list[CandidateResult]) -> dict:
    n = len(results)
    gross_positive = [r for r in results if r.gross_total_return > 0.0]
    flipped = [r for r in results if r.flipped]
    return {
        "n_candidates": n,
        "n_gross_positive": len(gross_positive),
        "n_flipped_negative": len(flipped),
        "flipped_names": [r.spec.name for r in flipped],
    }
