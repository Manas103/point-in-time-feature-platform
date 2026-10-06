from ptf.feature_spec import FeatureDefinition
from ptf.streaming_flink import to_flink_sql

SPEC = FeatureDefinition("trailing_qty_sum_20", "qty", "instrument_id", "event_time_ns", "sum", 20)


def test_to_flink_sql_reads_spec_fields_not_hardcoded():
    sql = to_flink_sql(SPEC)
    assert "PARTITION BY instrument_id" in sql
    assert "ORDER BY event_time" in sql
    assert "19 PRECEDING" in sql  # window_events - 1
    assert "SUM(qty)" in sql
    assert "trailing_qty_sum_20" in sql


def test_to_flink_sql_changes_with_spec():
    other = FeatureDefinition("other_feat", "price", "venue_id", "event_time_ns", "mean", 5)
    sql = to_flink_sql(other)
    assert "PARTITION BY venue_id" in sql
    assert "AVG(price)" in sql
    assert "4 PRECEDING" in sql
