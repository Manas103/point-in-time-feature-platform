"""Generates the 12,000,000-event synthetic session, writes it to the
Iceberg events table (the single source of truth both paths read from so
"same transformation code, same data" is actually true), and exports the
identical rows to a CSV for the Flink streaming job to replay.

Usage: python scripts/generate_and_load.py <warehouse_dir> <events_csv_out>
"""
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
from ptf.generate import generate_events, NUM_INSTRUMENTS, EVENTS_PER_INSTRUMENT, TOTAL_EVENTS  # noqa: E402
from ptf.iceberg_store import append_events_df, ensure_events_table, open_catalog  # noqa: E402


def main():
    warehouse_dir = sys.argv[1]
    csv_out = sys.argv[2]

    t0 = time.time()
    df = pd.DataFrame(
        generate_events(),
        columns=["instrument_id", "seq", "event_time_ns", "qty"],
    )
    t1 = time.time()
    assert len(df) == TOTAL_EVENTS

    catalog = open_catalog(warehouse_dir)
    table = ensure_events_table(catalog)
    snapshot_id = append_events_df(table, df)
    t2 = time.time()

    df.to_csv(csv_out, header=False, index=False)
    t3 = time.time()

    print(f"instruments={NUM_INSTRUMENTS} events_per_instrument={EVENTS_PER_INSTRUMENT} "
          f"total_events={len(df)}")
    print(f"generate: {t1 - t0:.2f}s")
    print(f"iceberg append: {t2 - t1:.2f}s  snapshot_id={snapshot_id}")
    print(f"csv export: {t3 - t2:.2f}s -> {csv_out}")
    print(f"SNAPSHOT_ID={snapshot_id}")


if __name__ == "__main__":
    main()
