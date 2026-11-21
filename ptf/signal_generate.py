"""Deterministic synthetic per-event RETURN series for the signal
implementation harness, at a smaller scale than generate.py's 12,000,000
feature-value parity run (see README for why this is a separate, smaller
dataset rather than a reuse of that one).

500 instruments, partitioned into 4 equal asset-class groups of 125
(EQUITY, RATE, FX, CREDIT). Each instrument's return series is a
deterministic trend-plus-noise signal, a pure function of
(instrument_id, seq); no library RNG, so two runs produce byte-identical
input, same discipline as generate.py.
"""
from __future__ import annotations

import math
from typing import Iterator, Tuple

from ptf.signal_spec import ASSET_CLASSES

NUM_INSTRUMENTS = 500
EVENTS_PER_INSTRUMENT = 1_000
GROUP_SIZE = NUM_INSTRUMENTS // len(ASSET_CLASSES)  # 125

BASE_EVENT_TIME_NS = 1_750_000_000_000_000_000
EVENT_SPACING_NS = 1_000_000


def asset_class_for(instrument_id: int) -> str:
    return ASSET_CLASSES[instrument_id // GROUP_SIZE]


def event_time_ns_for(instrument_id: int, seq: int) -> int:
    return BASE_EVENT_TIME_NS + instrument_id * EVENTS_PER_INSTRUMENT * EVENT_SPACING_NS + seq * EVENT_SPACING_NS


def ret_for(instrument_id: int, seq: int) -> float:
    """A deterministic trend-plus-noise return: a slow sinusoid (period
    depends on instrument_id, so different instruments trend on different
    cycles) plus a smaller deterministic pseudo-noise term derived from a
    fixed integer hash, with no library RNG."""
    period = 40 + (instrument_id % 37)  # events per half-cycle, varies by instrument
    trend = 0.0008 * math.sin(2.0 * math.pi * seq / period)

    h = (instrument_id * 2654435761 + seq * 40503) & 0xFFFFFFFF
    noise_unit = (h % 10_000) / 10_000.0 - 0.5  # in [-0.5, 0.5), deterministic
    noise = 0.0012 * noise_unit

    return trend + noise


def generate_events(
    num_instruments: int = NUM_INSTRUMENTS,
    events_per_instrument: int = EVENTS_PER_INSTRUMENT,
) -> Iterator[Tuple[int, int, int, str, float]]:
    """Yields (instrument_id, seq, event_time_ns, asset_class, ret)."""
    for instrument_id in range(num_instruments):
        asset_class = asset_class_for(instrument_id)
        for seq in range(events_per_instrument):
            yield (
                instrument_id,
                seq,
                event_time_ns_for(instrument_id, seq),
                asset_class,
                ret_for(instrument_id, seq),
            )
