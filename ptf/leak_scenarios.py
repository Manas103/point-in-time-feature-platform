"""40 seeded lookahead-leak scenarios, plus 40 clean (unambiguous) scenarios
for the false-positive check, both deterministic.

Layout: 4 instruments x 15 events, event_time = base + seq * 1000 (ns).

Adversarial scenarios (40 = 4 instruments x 10): query_time = event_time(seq) - 1,
for seq in 1..10. The correct event is seq-1 (999ns in the past); the event
at seq is only 1ns in the future. A direction='nearest' join picks the
*nearer* candidate by raw distance and has no notion of past vs future, so it
picks event(seq) -- 1ns ahead of the query, a textbook lookahead leak. A
correct backward-only join picks event(seq-1), which is non-adjacent in time
but legal. These are "seeded leaks" in the sense that they are constructed
specifically to make a plausible-but-wrong join leak; whether a given join
implementation actually leaks on them is the thing being measured.

Clean scenarios (40 = 4 instruments x 10): query_time = event_time(seq) + 100,
for seq in 1..10. event(seq) is unambiguously the nearest *and* the latest
not-after candidate (100ns in the past, versus event(seq+1) at 900ns in the
future), so every reasonable join agrees on the answer and no implementation
should flag a leak here.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from ptf.feature_spec import FeatureDefinition

NUM_INSTR = 4
NUM_EVENTS = 15
BASE_TIME_NS = 1_000_000
SPACING_NS = 1_000

LEAK_SPEC = FeatureDefinition(
    name="trailing_qty_sum_3",
    source_field="qty",
    group_key="instrument_id",
    order_key="event_time_ns",
    agg="sum",
    window_events=3,
)


def _event_time(seq: int) -> int:
    return BASE_TIME_NS + seq * SPACING_NS


def build_feature_df() -> pd.DataFrame:
    rows = []
    for instr in range(NUM_INSTR):
        for seq in range(NUM_EVENTS):
            rows.append({
                "instrument_id": instr,
                "seq": seq,
                "event_time_ns": _event_time(seq),
                "qty": 10 + seq,
            })
    return pd.DataFrame(rows)


def build_adversarial_queries() -> pd.DataFrame:
    rows = []
    for instr in range(NUM_INSTR):
        for seq in range(1, 11):
            rows.append({
                "instrument_id": instr,
                "query_time_ns": _event_time(seq) - 1,
                "expected_event_seq": seq - 1,
            })
    assert len(rows) == 40
    return pd.DataFrame(rows)


def build_clean_queries() -> pd.DataFrame:
    rows = []
    for instr in range(NUM_INSTR):
        for seq in range(1, 11):
            rows.append({
                "instrument_id": instr,
                "query_time_ns": _event_time(seq) + 100,
                "expected_event_seq": seq,
            })
    assert len(rows) == 40
    return pd.DataFrame(rows)


def build_scenarios() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_feature_df(), build_adversarial_queries(), build_clean_queries()
