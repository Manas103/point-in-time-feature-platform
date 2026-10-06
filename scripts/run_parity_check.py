"""Row-for-row diff between the Flink streaming output and the Iceberg
batch-backfill output, both computed from the same FeatureDefinition over
the same 12,000,000-event session.

Usage: python scripts/run_parity_check.py <stream_sink_dir> <batch_csv>
"""
import glob
import sys

import pandas as pd

sys.path.insert(0, ".")
from ptf.feature_spec import TRAILING_QTY_SUM  # noqa: E402

SPEC = TRAILING_QTY_SUM


def main():
    stream_dir = sys.argv[1]
    batch_csv = sys.argv[2]

    stream_files = sorted(glob.glob(f"{stream_dir}/*"))
    stream_df = pd.concat(
        (pd.read_csv(f, header=None, names=[SPEC.group_key, "seq", SPEC.order_key, SPEC.name]) for f in stream_files),
        ignore_index=True,
    )
    batch_df = pd.read_csv(batch_csv)

    print(f"stream rows: {len(stream_df)}  batch rows: {len(batch_df)}")
    assert len(stream_df) == len(batch_df), "row count mismatch before any value comparison"

    merged = stream_df.merge(
        batch_df,
        on=[SPEC.group_key, SPEC.order_key],
        suffixes=("_stream", "_batch"),
        how="outer",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    mismatched_value = merged[
        (merged["_merge"] == "both") & (merged[f"{SPEC.name}_stream"] != merged[f"{SPEC.name}_batch"])
    ]

    total = len(merged)
    bad = len(unmatched) + len(mismatched_value)
    print(f"total compared keys: {total}")
    print(f"unmatched keys (present in only one side): {len(unmatched)}")
    print(f"matched keys with different feature values: {len(mismatched_value)}")
    print(f"exact matches: {total - bad} / {total}")

    if bad == 0:
        print(f"PASS: streaming and batch outputs matched row for row across {total} feature values.")
    else:
        print("FAIL")
        print(mismatched_value.head(20))

    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
