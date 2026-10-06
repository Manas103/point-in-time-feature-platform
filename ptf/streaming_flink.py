"""Streaming compiler backend: FeatureDefinition -> PyFlink Table API job.

Runs in streaming execution mode (EnvironmentSettings.in_streaming_mode()),
not batch -- the OVER window below is evaluated against a declared
event-time attribute with a watermark, which is what makes this the
streaming runtime's incremental, event-time-ordered evaluation path rather
than the batch optimizer's. The source here is a bounded CSV replay of a
recorded event log rather than a live Kafka topic, which is the honest
"simulated" framing this project states throughout: the point being proved
is that one feature definition drives Flink's actual streaming execution
engine, not that a live broker was stood up.

Must be run under the streaming venv (apache-flink installed; see README).
Writing the result to a filesystem sink, not collecting rows in Python, is
not a style choice -- collect() pulls one row at a time over the Python/JVM
bridge and was measured at roughly 127 rows/sec in this environment (see
docs/flink_collect_vs_sink_finding.txt), which would take about a day for
12M rows. Writing straight to a sink stays inside the JVM end to end.
"""
from __future__ import annotations

from ptf.feature_spec import FeatureDefinition

_AGG_SQL = {"sum": "SUM", "count": "COUNT", "mean": "AVG"}


def to_flink_sql(spec: FeatureDefinition, source_table: str = "events") -> str:
    agg_fn = _AGG_SQL[spec.agg]
    return f"""
        SELECT {spec.group_key}, seq, {spec.order_key},
               CAST({agg_fn}({spec.source_field}) OVER (
                   PARTITION BY {spec.group_key}
                   ORDER BY event_time
                   ROWS BETWEEN {spec.window_events - 1} PRECEDING AND CURRENT ROW
               ) AS BIGINT) AS {spec.name}
        FROM {source_table}
    """


def run_streaming_job(spec: FeatureDefinition, input_csv_path: str, output_dir: str, parallelism: int = 1):
    """Reads input_csv_path, computes spec via Flink's streaming Table API,
    writes one-row-per-event results (partitioned across `parallelism` sink
    files) to output_dir. Returns the TableResult for the caller to .wait() on.
    """
    import sys

    from pyflink.table import EnvironmentSettings, TableEnvironment

    settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(settings)
    t_env.get_config().set("parallelism.default", str(parallelism))
    t_env.get_config().set("python.executable", sys.executable)
    t_env.get_config().set("python.client.executable", sys.executable)

    # Columns must match generate.generate_events()'s (instrument_id, seq,
    # event_time_ns, qty) order exactly -- the CSV connector is positional.
    t_env.execute_sql(f"""
        CREATE TABLE events (
            {spec.group_key} BIGINT,
            seq BIGINT,
            {spec.order_key} BIGINT,
            {spec.source_field} BIGINT,
            event_time AS TO_TIMESTAMP_LTZ({spec.order_key} / 1000000, 3),
            WATERMARK FOR event_time AS event_time - INTERVAL '0' SECOND
        ) WITH (
            'connector' = 'filesystem',
            'path' = 'file://{input_csv_path}',
            'format' = 'csv'
        )
    """)

    t_env.execute_sql(f"""
        CREATE TABLE sink (
            {spec.group_key} BIGINT,
            seq BIGINT,
            {spec.order_key} BIGINT,
            {spec.name} BIGINT
        ) WITH (
            'connector' = 'filesystem',
            'path' = 'file://{output_dir}',
            'format' = 'csv'
        )
    """)

    result_table = t_env.sql_query(to_flink_sql(spec))
    stmt_set = t_env.create_statement_set()
    stmt_set.add_insert("sink", result_table)
    return stmt_set.execute()
