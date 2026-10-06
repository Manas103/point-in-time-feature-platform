"""Point-in-time event store on Apache Iceberg (via pyiceberg), local SQL
catalog + filesystem warehouse (no Hive metastore, no cluster -- see README
honest framing). Every append creates a new snapshot; batch backfills pin a
snapshot id so a training set stays reproducible even after later data
arrives, which is the mechanism behind "every training set resolves to a
pinned definition version and data vintage".

Must be run under the batch venv (pyiceberg installed; see README). pyiceberg
0.12.0 requires pyarrow>=19, which conflicts with apache-flink's pinned
pyarrow<21,>=5 dependency on apache-beam<17 in the same environment -- see
docs/dependency_conflict_finding.txt. That is why this project ships two
venvs rather than one.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import pyarrow as pa

EVENTS_SCHEMA = pa.schema([
    ("instrument_id", pa.int64()),
    ("seq", pa.int64()),
    ("event_time_ns", pa.int64()),
    ("qty", pa.int64()),
])


def open_catalog(warehouse_dir: str):
    from pyiceberg.catalog.sql import SqlCatalog

    os.makedirs(warehouse_dir, exist_ok=True)
    catalog_db = os.path.join(warehouse_dir, "catalog.db")
    return SqlCatalog(
        "ptf",
        **{
            "uri": f"sqlite:///{catalog_db}",
            "warehouse": f"file://{warehouse_dir}/data",
        },
    )


def ensure_events_table(catalog, namespace: str = "ptf", table_name: str = "events"):
    from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError

    try:
        catalog.create_namespace(namespace)
    except NamespaceAlreadyExistsError:
        pass

    identifier = f"{namespace}.{table_name}"
    try:
        return catalog.load_table(identifier)
    except NoSuchTableError:
        return catalog.create_table(identifier, schema=EVENTS_SCHEMA)


def append_events_df(table, events_df: pd.DataFrame) -> int:
    """Appends events_df (columns matching EVENTS_SCHEMA) as a new snapshot.
    Returns the new snapshot id."""
    arrow_table = pa.Table.from_pandas(
        events_df[["instrument_id", "seq", "event_time_ns", "qty"]],
        schema=EVENTS_SCHEMA,
        preserve_index=False,
    )
    table.append(arrow_table)
    table.refresh()
    return table.current_snapshot().snapshot_id


def scan_events(table, snapshot_id: Optional[int] = None) -> pd.DataFrame:
    """Reads the events table as of `snapshot_id` (None = latest), which is
    the time-travel mechanism a pinned training set relies on."""
    scan = table.scan(snapshot_id=snapshot_id) if snapshot_id is not None else table.scan()
    return scan.to_arrow().to_pandas()
