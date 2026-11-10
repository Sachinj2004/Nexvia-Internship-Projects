"""
Nexvia Fraud Detection Platform (AIML-01)
Feature Store Sample Data Generator: feature_repo/generate_sample_data.py
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    from feature_repo.features import transaction_entity, transaction_features_view
except ImportError:
    from features import transaction_entity, transaction_features_view

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = REPO_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "transactions.parquet"

# Exact mapping from Feast PrimitiveFeastType enum name to numpy dtype
FEAST_DTYPE_MAP = {
    "FLOAT32": np.dtype("float32"),
    "FLOAT64": np.dtype("float64"),
    "INT64": np.dtype("int64"),
    "INT32": np.dtype("int32"),
    "BOOL": np.dtype("bool"),
}


def validate_feast_schema(df: pd.DataFrame) -> None:
    """Validate DataFrame against actual Feast definitions."""
    # Fixed: Feast 0.67.0 uses join_key (singular)
    entity_key = getattr(transaction_entity, "join_key", transaction_entity.name)
    timestamp_col = transaction_features_view.batch_source.timestamp_field
    created_col = transaction_features_view.batch_source.created_timestamp_column
    feature_fields = {field.name: field.dtype for field in transaction_features_view.schema}

    # 1. Missing columns check
    required_cols = [entity_key, timestamp_col, created_col] + list(feature_fields.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feast schema missing required columns: {missing}")

    # 2. Entity key exact int64 validation
    if df[entity_key].dtype != np.dtype("int64"):
        raise TypeError(f"Entity key '{entity_key}' must be int64, found {df[entity_key].dtype}")

    # 3. Timezone-aware UTC timestamp validation and NaT rejection
    for col_name in [timestamp_col, created_col]:
        series = df[col_name]
        if not isinstance(series.dtype, pd.DatetimeTZDtype):
            raise TypeError(f"Timestamp '{col_name}' must be DatetimeTZDtype, found {series.dtype}")
        if str(series.dtype.tz) not in ["UTC", "datetime.timezone.utc", "tzutc()"]:
            raise ValueError(f"Timestamp '{col_name}' must be UTC, found {series.dtype.tz}")
        if int(series.isna().sum()) > 0:
            raise ValueError(f"Timestamp '{col_name}' contains NaT / missing values")

    # 4. Exact Feast dtype validation
    for field_name, feast_dtype in feature_fields.items():
        series = df[field_name]
        dtype_name = getattr(feast_dtype, "name", str(feast_dtype)).upper()

        if dtype_name in FEAST_DTYPE_MAP:
            expected = FEAST_DTYPE_MAP[dtype_name]
            if series.dtype != expected:
                raise TypeError(
                    f"Column '{field_name}' dtype mismatch: Feast declared {dtype_name} "
                    f"(requires {expected}), found {series.dtype}"
                )
        elif dtype_name == "STRING":
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                raise TypeError(f"Column '{field_name}' cannot be numeric/bool for Feast String")
            if not (isinstance(series.dtype, pd.StringDtype) or series.dtype == object):
                raise TypeError(f"Column '{field_name}' must be StringDtype or object")
            non_null = series.dropna()
            if not non_null.empty and not all(isinstance(v, str) for v in non_null):
                raise TypeError(f"Column '{field_name}' contains non-string values")
        else:
            raise TypeError(f"Unsupported Feast dtype: {feast_dtype}")

    # 5. Null and infinity checks
    if df[required_cols].isnull().any().any():
        raise ValueError("Null values detected in Feast required columns")
    float_cols = [c for c, dt in feature_fields.items() if getattr(dt, "name", "").upper() in ("FLOAT32", "FLOAT64")]
    if any(np.isinf(df[c]).any() for c in float_cols):
        raise ValueError("Infinite values detected in float feature columns")


def generate_parquet_data(overwrite: bool = False) -> Path:
    """Generate sample Parquet data with closed='left' rolling window."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Overwrite protection
    if OUTPUT_FILE.exists() and not overwrite:
        raise FileExistsError(
            f"Parquet file already exists: {OUTPUT_FILE}\n"
            "Execution halted. Use --overwrite to replace it."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source credit card dataset missing at {DATA_PATH}")

    print(f"Reading sample (2,000 transactions) from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, nrows=2000)

    # 1. Deterministic primary key
    df["transaction_id"] = np.arange(len(df), dtype=np.int64)

    # 2. Event timestamps (timezone-aware UTC)
    base_epoch = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    df["datetime"] = base_epoch + pd.to_timedelta(df["Time"], unit="s")
    df["created"] = pd.Timestamp.now(tz="UTC")

    # 3. Features with closed='left' (no feature leakage)
    df["time_delta_prev"] = df["Time"].diff().fillna(0.0).astype(np.float32)

    df_temp = df.set_index("datetime")
    rolling_1h = df_temp["Amount"].rolling("3600s", closed="left")
    df["rolling_tx_count_1h"] = rolling_1h.count().fillna(0).values.astype(np.int64)
    df["rolling_amount_mean_1h"] = rolling_1h.mean().fillna(0.0).values.astype(np.float32)

    # Safe ratio without NaN or inf
    amount_arr = df["Amount"].values.astype(np.float32)
    safe_ratio = np.where(
        df["rolling_tx_count_1h"].values == 0,
        1.0,
        amount_arr / np.maximum(df["rolling_amount_mean_1h"].values, 1e-4)
    )
    df["amount_to_window_mean_ratio"] = np.nan_to_num(safe_ratio, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
    df["pca_energy_norm"] = np.sqrt(np.square(df[pca_cols]).sum(axis=1)).astype(np.float32)
    df["Amount"] = df["Amount"].astype(np.float32)

    keep_cols = [
        "transaction_id", "datetime", "created", "Time", "Amount",
        "time_delta_prev", "rolling_amount_mean_1h", "rolling_tx_count_1h",
        "amount_to_window_mean_ratio", "pca_energy_norm", "Class"
    ]
    missing_source = [c for c in keep_cols if c not in df.columns]
    if missing_source:
        raise ValueError(f"Missing columns in source: {missing_source}")

    sample_df = df[keep_cols]
    validate_feast_schema(sample_df)

    sample_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Successfully generated {len(sample_df)} records in {OUTPUT_FILE}")
    print(f"Parquet file size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    return OUTPUT_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample Parquet data for Feast")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite if file exists")
    args = parser.parse_args()

    try:
        generate_parquet_data(overwrite=args.overwrite)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
