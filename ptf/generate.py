"""Deterministic synthetic event generator.

500 simulated instruments, 24,000 events each, 12,000,000 events total. Each
instrument's own event_time_ns strictly increases with seq, so there are no
same-instrument timestamp ties for the streaming/batch windows to disagree
about (cross-instrument ties are irrelevant -- the window is partitioned).
Everything is a pure function of (instrument_id, seq); no library RNG is
used, for the same reason the sibling C++ repos avoid std::uniform_int_distribution:
a value that must reproduce exactly should not depend on which standard
library draws it.
"""
from __future__ import annotations

from typing import Iterator, Tuple

NUM_INSTRUMENTS = 500
EVENTS_PER_INSTRUMENT = 24_000
TOTAL_EVENTS = NUM_INSTRUMENTS * EVENTS_PER_INSTRUMENT  # 12,000,000

BASE_EVENT_TIME_NS = 1_750_000_000_000_000_000  # arbitrary fixed epoch
EVENT_SPACING_NS = 1_000_000  # 1ms between an instrument's own consecutive events


def qty_for(instrument_id: int, seq: int) -> int:
    return 1 + ((instrument_id * 7919 + seq * 104729) % 500)


def event_time_ns_for(instrument_id: int, seq: int) -> int:
    # Each instrument gets its own disjoint time band so no cross-instrument
    # collision is even possible, and within a band, time strictly increases
    # with seq.
    return BASE_EVENT_TIME_NS + instrument_id * EVENTS_PER_INSTRUMENT * EVENT_SPACING_NS + seq * EVENT_SPACING_NS


def generate_events(
    num_instruments: int = NUM_INSTRUMENTS,
    events_per_instrument: int = EVENTS_PER_INSTRUMENT,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yields (instrument_id, seq, event_time_ns, qty)."""
    for instrument_id in range(num_instruments):
        for seq in range(events_per_instrument):
            yield (
                instrument_id,
                seq,
                event_time_ns_for(instrument_id, seq),
                qty_for(instrument_id, seq),
            )
