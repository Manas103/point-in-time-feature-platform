import shutil
import tempfile

import pandas as pd
import pytest

pyiceberg = pytest.importorskip("pyiceberg", reason="training-set pinning test needs the batch venv")

from ptf.feature_spec import FeatureDefinition  # noqa: E402
from ptf.iceberg_store import append_events_df, ensure_events_table, open_catalog  # noqa: E402
from ptf.training_set import build_training_set  # noqa: E402

SPEC = FeatureDefinition("trailing_sum_2", "qty", "instrument_id", "event_time_ns", "sum", 2)


@pytest.fixture
def warehouse_dir():
    d = tempfile.mkdtemp(prefix="ptf_test_warehouse_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pinned_snapshot_survives_later_appends(warehouse_dir):
    catalog = open_catalog(warehouse_dir)
    table = ensure_events_table(catalog, namespace="ns1")

    first_batch = pd.DataFrame({
        "instrument_id": [1, 1, 1],
        "seq": [0, 1, 2],
        "event_time_ns": [1000, 2000, 3000],
        "qty": [10, 20, 30],
    })
    snap1 = append_events_df(table, first_batch)

    queries = pd.DataFrame({"instrument_id": [1], "query_time_ns": [3000]})
    result_v1, manifest_v1 = build_training_set(table, SPEC, queries, snapshot_id=snap1)
    assert manifest_v1.snapshot_id == snap1
    assert manifest_v1.feature_version == SPEC.version()
    value_v1 = result_v1.loc[0, SPEC.name]
    assert value_v1 == 50  # 20 + 30, window=2

    # Append more data (a later vintage) and build again, pinned to the OLD
    # snapshot -- must reproduce exactly the same value.
    second_batch = pd.DataFrame({
        "instrument_id": [1],
        "seq": [3],
        "event_time_ns": [4000],
        "qty": [999],
    })
    snap2 = append_events_df(table, second_batch)
    assert snap2 != snap1

    result_v1_again, manifest_v1_again = build_training_set(table, SPEC, queries, snapshot_id=snap1)
    assert result_v1_again.loc[0, SPEC.name] == value_v1
    assert manifest_v1_again.snapshot_id == snap1

    # Building against the new snapshot sees the new data.
    result_v2, manifest_v2 = build_training_set(table, SPEC, queries, snapshot_id=snap2)
    assert manifest_v2.snapshot_id == snap2
    assert result_v2.loc[0, SPEC.name] == value_v1  # query_time=3000 still can't see seq=3 (event_time=4000)
