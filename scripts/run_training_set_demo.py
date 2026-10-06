"""Demonstrates pinned-snapshot reproducibility: builds a training set
against snapshot N, appends more events (snapshot N+1), and rebuilds
against the pinned snapshot N id, showing the values are identical even
though the table has moved on. Writes its own warehouse under
<warehouse_dir>/training_set_demo so it never touches the 12M-event
warehouse used for the parity check.

Usage: python scripts/run_training_set_demo.py <warehouse_dir>
"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from ptf.feature_spec import FeatureDefinition  # noqa: E402
from ptf.iceberg_store import append_events_df, ensure_events_table, open_catalog  # noqa: E402
from ptf.training_set import build_training_set  # noqa: E402

SPEC = FeatureDefinition("trailing_qty_sum_3", "qty", "instrument_id", "event_time_ns", "sum", 3)


def main():
    warehouse_dir = sys.argv[1] + "/training_set_demo"
    catalog = open_catalog(warehouse_dir)
    table = ensure_events_table(catalog, namespace="demo")

    batch_1 = pd.DataFrame({
        "instrument_id": [7, 7, 7, 7],
        "seq": [0, 1, 2, 3],
        "event_time_ns": [1000, 2000, 3000, 4000],
        "qty": [10, 20, 30, 40],
    })
    snap_1 = append_events_df(table, batch_1)

    queries = pd.DataFrame({"instrument_id": [7], "query_time_ns": [4000]})
    result_1, manifest_1 = build_training_set(table, SPEC, queries, snapshot_id=snap_1)
    value_1 = int(result_1.loc[0, SPEC.name])
    print(f"snapshot_1={snap_1}  feature_version={manifest_1.feature_version}  value={value_1}")

    batch_2 = pd.DataFrame({
        "instrument_id": [7],
        "seq": [4],
        "event_time_ns": [5000],
        "qty": [9999],
    })
    snap_2 = append_events_df(table, batch_2)
    print(f"appended batch_2 -> snapshot_2={snap_2}")

    result_1_again, manifest_1_again = build_training_set(table, SPEC, queries, snapshot_id=snap_1)
    value_1_again = int(result_1_again.loc[0, SPEC.name])
    print(f"rebuilt against pinned snapshot_1={snap_1}: value={value_1_again}")

    result_2, manifest_2 = build_training_set(table, SPEC, queries, snapshot_id=snap_2)
    value_2 = int(result_2.loc[0, SPEC.name])
    print(f"rebuilt against new snapshot_2={snap_2}: value={value_2}")

    pinned_ok = (value_1 == value_1_again) and (manifest_1.snapshot_id == manifest_1_again.snapshot_id == snap_1)
    unaffected_by_future_data = (value_2 == value_1)  # query_time=4000 still can't see seq=4 (event_time=5000)
    print()
    print(f"pinned snapshot reproduces identical value after later appends: {pinned_ok}")
    print(f"new snapshot still respects point-in-time correctness for this query: {unaffected_by_future_data}")
    ok = pinned_ok and unaffected_by_future_data
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
