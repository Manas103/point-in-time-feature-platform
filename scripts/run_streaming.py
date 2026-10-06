"""Runs the PyFlink streaming Table API job for TRAILING_QTY_SUM against the
event CSV exported by generate_and_load.py.

Usage: python scripts/run_streaming.py <events_csv> <sink_dir>
"""
import shutil
import sys
import time

sys.path.insert(0, ".")
from ptf.feature_spec import TRAILING_QTY_SUM  # noqa: E402
from ptf.streaming_flink import run_streaming_job  # noqa: E402


def main():
    events_csv = sys.argv[1]
    sink_dir = sys.argv[2]
    shutil.rmtree(sink_dir, ignore_errors=True)

    t0 = time.time()
    result = run_streaming_job(TRAILING_QTY_SUM, events_csv, sink_dir, parallelism=4)
    result.wait()
    t1 = time.time()

    print(f"feature_version={TRAILING_QTY_SUM.version()}")
    print(f"flink streaming job elapsed: {t1 - t0:.2f}s -> {sink_dir}")


if __name__ == "__main__":
    main()
