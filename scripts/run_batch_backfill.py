"""Iceberg batch backfill: scans the pinned snapshot, computes the feature
with ptf.batch_pandas (the batch half of the same FeatureDefinition the
streaming job in scripts/run_streaming.py compiles), writes one row per
event to a CSV for scripts/run_parity_check.py to diff against the
streaming output.

Usage: python scripts/run_batch_backfill.py <warehouse_dir> <snapshot_id> <out_csv>
"""
import sys
import time

sys.path.insert(0, ".")
from ptf.batch_pandas import compute_batch_feature  # noqa: E402
from ptf.feature_spec import TRAILING_QTY_SUM  # noqa: E402
from ptf.iceberg_store import ensure_events_table, open_catalog, scan_events  # noqa: E402


def main():
    warehouse_dir = sys.argv[1]
    snapshot_id = int(sys.argv[2])
    out_csv = sys.argv[3]

    catalog = open_catalog(warehouse_dir)
    table = ensure_events_table(catalog)

    t0 = time.time()
    events = scan_events(table, snapshot_id=snapshot_id)
    t1 = time.time()

    result = compute_batch_feature(events, TRAILING_QTY_SUM)
    t2 = time.time()

    result.to_csv(out_csv, index=False)
    t3 = time.time()

    print(f"rows={len(result)} feature_version={TRAILING_QTY_SUM.version()} snapshot_id={snapshot_id}")
    print(f"iceberg scan: {t1 - t0:.2f}s")
    print(f"batch compute: {t2 - t1:.2f}s")
    print(f"csv write: {t3 - t2:.2f}s -> {out_csv}")


if __name__ == "__main__":
    main()
