"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Evidently AI Drift Runner: src/drift_runner.py

Standalone runner script executed inside the isolated `nexvia-monitoring` environment.
Loads the baseline reference dataset and current transaction data, evaluates statistical
data drift across the 30 numerical features using Evidently's DataDriftPreset, and saves
both an interactive HTML report and a machine-readable JSON metrics summary.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from Parquet, CSV, or JSON lines format."""
    path_str = str(file_path)
    if path_str.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif path_str.endswith(".csv"):
        return pd.read_csv(file_path)
    elif path_str.endswith(".json"):
        try:
            return pd.read_json(file_path, orient="records")
        except ValueError:
            return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format for drift data: {file_path}")


def extract_drift_summary(snapshot: Any, avail_cols: List[str]) -> Dict[str, Any]:
    """Extract core drift KPIs directly from the Evidently Snapshot object."""
    summary: Dict[str, Any] = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "number_of_drifted_columns": 0,
        "number_of_columns": len(avail_cols),
        "drift_by_columns": {},
        "stat_test": "Kolmogorov-Smirnov / Wasserstein"
    }

    try:
        snap_dict = snapshot.dict()
        metrics_list = snap_dict.get("metrics", [])

        for item in metrics_list:
            if not isinstance(item, dict):
                continue

            config = item.get("config", {})
            cfg_type = config.get("type", "")
            val = item.get("value")

            # 1. Overall dataset drift count & share
            if "DriftedColumnsCount" in cfg_type or "DriftedColumnsCount" in item.get("metric_name", ""):
                if isinstance(val, dict):
                    count = int(val.get("count", 0))
                    share = round(float(val.get("share", 0.0)), 4)
                    summary["number_of_drifted_columns"] = count
                    summary["drift_share"] = share
                    summary["dataset_drift"] = count > 0

            # 2. Column-level drift score and test result
            elif "ValueDrift" in cfg_type or "ValueDrift" in item.get("metric_name", ""):
                col_name = config.get("column")
                threshold = float(config.get("threshold", 0.05))
                method = config.get("method", "K-S p_value")
                
                if col_name and isinstance(val, (int, float)):
                    # For p-value tests (K-S, chi2, etc.), p_val < threshold means statistically significant drift.
                    # For distance / norm metrics (Wasserstein, PSI, etc.), distance > threshold means drift.
                    is_p_val = "p_value" in method.lower() or "p-value" in method.lower()
                    if is_p_val:
                        drift_detected = float(val) < threshold
                    else:
                        drift_detected = float(val) > threshold

                    summary["drift_by_columns"][col_name] = {
                        "drift_detected": drift_detected,
                        "drift_score": round(float(val), 6),
                        "threshold": threshold,
                        "stat_test": method
                    }

        # Synchronize overall drift count if ValueDrift detected drift but DriftedColumnsCount missed it
        if summary["drift_by_columns"]:
            flagged = [c for c, info in summary["drift_by_columns"].items() if info.get("drift_detected")]
            if len(flagged) > summary["number_of_drifted_columns"]:
                summary["number_of_drifted_columns"] = len(flagged)
                summary["drift_share"] = round(len(flagged) / len(summary["drift_by_columns"]), 4)
                summary["dataset_drift"] = len(flagged) > 0

    except Exception as exc:
        summary["parse_warning"] = str(exc)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evidently Data Drift Worker Runner")
    parser.add_argument("--reference", type=str, default=str(PROJECT_ROOT / "data" / "reference_data.parquet"),
                        help="Path to baseline reference dataset (Parquet or CSV)")
    parser.add_argument("--current", type=str, required=True,
                        help="Path to current evaluation dataset (Parquet, CSV, or JSON)")
    parser.add_argument("--output-html", type=str, default=str(PROJECT_ROOT / "outputs" / "drift_report.html"),
                        help="Path to save interactive HTML report")
    parser.add_argument("--output-json", type=str, default=str(PROJECT_ROOT / "outputs" / "drift_metrics.json"),
                        help="Path to save extracted JSON drift summary")
    parser.add_argument("--sample-size", type=int, default=15000,
                        help="Maximum sample rows to use from reference dataset for performance")

    args = parser.parse_args()

    ref_path = Path(args.reference)
    cur_path = Path(args.current)
    html_out = Path(args.output_html)
    json_out = Path(args.output_json)

    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    if not ref_path.exists():
        sys.stderr.write(f"ERROR: Reference dataset not found: {ref_path}\n")
        sys.exit(1)
    if not cur_path.exists():
        sys.stderr.write(f"ERROR: Current evaluation dataset not found: {cur_path}\n")
        sys.exit(1)

    # 1. Load data
    ref_df = load_dataset(ref_path)
    cur_df = load_dataset(cur_path)

    # 2. Select common features strictly matching EXPECTED_FEATURES
    avail_cols = [c for c in EXPECTED_FEATURES if c in ref_df.columns and c in cur_df.columns]
    if len(avail_cols) < 5:
        sys.stderr.write(f"ERROR: Insufficient feature overlap ({len(avail_cols)} features found)\n")
        sys.exit(1)

    ref_clean = ref_df[avail_cols].copy()
    cur_clean = cur_df[avail_cols].copy()

    # Apply sampling if reference dataset is large
    if args.sample_size and len(ref_clean) > args.sample_size:
        ref_clean = ref_clean.sample(n=args.sample_size, random_state=42)

    # 3. Configure and run Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_clean, current_data=cur_clean)

    # 4. Save HTML report
    snapshot.save_html(str(html_out))

    # 5. Save Raw Snapshot JSON
    raw_snapshot_path = html_out.with_suffix(".snapshot.json")
    snapshot.save_json(str(raw_snapshot_path))

    # 6. Extract clean summary KPIs
    summary = extract_drift_summary(snapshot, avail_cols)
    summary["evaluated_features"] = len(avail_cols)
    summary["reference_rows"] = len(ref_clean)
    summary["current_rows"] = len(cur_clean)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["html_report_path"] = str(html_out)

    # Save summary JSON
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Clean up intermediate snapshot JSON
    if raw_snapshot_path.exists():
        try:
            raw_snapshot_path.unlink()
        except OSError:
            pass

    # Print summary JSON to stdout for parent process
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
