"""
Nexvia Fraud Detection Platform (AIML-01)
Feast Feature Definitions: feature_repo/features.py

Defines the transaction entity, Parquet offline batch source, and the
transaction_velocity_features FeatureView for real-time online retrieval.
"""

from datetime import timedelta
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Resolves path relative to feature_repo
REPO_DIR = Path(__file__).resolve().parent
PARQUET_PATH = str(REPO_DIR / "data" / "transactions.parquet")

# -----------------------------------------------------------------------------
# 1. Primary Entity: transaction_id
# -----------------------------------------------------------------------------
# Based purely on the authentic transaction sequence in creditcard.csv without
# fabricating non-existent user_id or account_id identifiers.
transaction_entity = Entity(
    name="transaction_id",
    value_type=ValueType.INT64,
    join_keys=["transaction_id"],
    description="Deterministic transaction identifier corresponding to chronological ledger index"
)

# -----------------------------------------------------------------------------
# 2. Batch Offline Source (Parquet File)
# -----------------------------------------------------------------------------
transaction_source = FileSource(
    name="transaction_parquet_source",
    path=PARQUET_PATH,
    timestamp_field="datetime",
    created_timestamp_column="created"
)

# -----------------------------------------------------------------------------
# 3. Feature View: Velocity and Rolling Aggregates
# -----------------------------------------------------------------------------
transaction_features_view = FeatureView(
    name="transaction_velocity_features",
    entities=[transaction_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(
            name="Amount",
            dtype=Float32,
            description="Transaction monetary amount ($)"
        ),
        Field(
            name="time_delta_prev",
            dtype=Float32,
            description="Seconds elapsed since the immediately preceding transaction in the ledger"
        ),
        Field(
            name="rolling_amount_mean_1h",
            dtype=Float32,
            description="1-hour rolling average transaction amount"
        ),
        Field(
            name="rolling_tx_count_1h",
            dtype=Int64,
            description="1-hour rolling transaction frequency count"
        ),
        Field(
            name="amount_to_window_mean_ratio",
            dtype=Float32,
            description="Ratio of current amount to 1-hour average (anomalous spike indicator)"
        ),
        Field(
            name="pca_energy_norm",
            dtype=Float32,
            description="Euclidean L2 norm of the 28 PCA anonymized components"
        )
    ],
    online=True,
    source=transaction_source,
    description="Real-time transaction velocity, rolling statistical aggregations, and PCA energy metrics"
)
