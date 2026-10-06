"""Builds a training set that resolves to a pinned feature-definition
version and a pinned Iceberg data vintage (snapshot id), so re-running the
same build later -- after more events have landed in the table -- still
reproduces exactly the values a model was trained on.

Must be run under the batch venv (uses ptf.iceberg_store / pyiceberg).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ptf.asof import asof_join
from ptf.batch_pandas import compute_batch_feature
from ptf.feature_spec import FeatureDefinition
from ptf.iceberg_store import scan_events


@dataclass(frozen=True)
class TrainingSetManifest:
    feature_name: str
    feature_version: str
    snapshot_id: int


def build_training_set(
    table,
    spec: FeatureDefinition,
    queries_df: pd.DataFrame,
    snapshot_id: int,
    query_time_col: str = "query_time_ns",
) -> tuple[pd.DataFrame, TrainingSetManifest]:
    """snapshot_id must be passed explicitly and pinned by the caller (not
    defaulted to "latest") -- that is what makes the manifest meaningful
    rather than a label on data that could have silently moved under it.
    """
    events = scan_events(table, snapshot_id=snapshot_id)
    feature_df = compute_batch_feature(events, spec)
    joined = asof_join(feature_df, queries_df, spec, query_time_col=query_time_col)
    manifest = TrainingSetManifest(
        feature_name=spec.name,
        feature_version=spec.version(),
        snapshot_id=snapshot_id,
    )
    return joined, manifest
