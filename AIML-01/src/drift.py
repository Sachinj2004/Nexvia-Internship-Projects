"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Evidently AI Drift Monitoring Bridge: src/drift.py

Provides thread-safe in-memory transaction buffering and bridges to the isolated
`nexvia-monitoring` environment to execute Evidently AI data-drift reports without
disturbing the main inference pipeline or incurring per-prediction latency overhead.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("nexvia.drift")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONITORING_PYTHON = Path("/opt/anaconda3/envs/nexvia-monitoring/bin/python")
DEFAULT_RUNNER_SCRIPT = PROJECT_ROOT / "src" / "drift_runner.py"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_data.parquet"
DEFAULT_OUTPUT_HTML = PROJECT_ROOT / "outputs" / "drift_report.html"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "drift_metrics.json"

EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


class FraudDriftMonitor:
    """
    Thread-safe drift monitoring manager.
    Buffers live scoring transactions and triggers asynchronous or on-demand
    Evidently drift analysis via the isolated nexvia-monitoring environment.
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        reference_path: Optional[Path] = None,
        monitoring_python: Optional[Path] = None,
        runner_script: Optional[Path] = None,
        output_html: Optional[Path] = None,
        output_json: Optional[Path] = None,
    ):
        self.buffer_size = buffer_size
        self.reference_path = reference_path or DEFAULT_REFERENCE_PATH
        self.monitoring_python = monitoring_python or DEFAULT_MONITORING_PYTHON
        self.runner_script = runner_script or DEFAULT_RUNNER_SCRIPT
        self.output_html = output_html or DEFAULT_OUTPUT_HTML
        self.output_json = output_json or DEFAULT_OUTPUT_JSON

        self._buffer: deque = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._last_metrics: Optional[Dict[str, Any]] = None

        # Load previously saved metrics if present on disk
        if self.output_json.exists():
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    self._last_metrics = json.load(f)
            except Exception:
                self._last_metrics = None

    def record_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Record an incoming transaction in the sliding window buffer.
        Thread-safe, non-blocking, sub-microsecond execution.
        """
        if not isinstance(transaction, dict):
            return

        clean_record = {}
        for col in EXPECTED_FEATURES:
            val = transaction.get(col)
            if val is not None:
                try:
                    clean_record[col] = float(val)
                except (ValueError, TypeError):
                    pass

        if len(clean_record) == len(EXPECTED_FEATURES):
            with self._lock:
                self._buffer.append(clean_record)

    def record_batch(self, transactions: Union[List[Dict[str, Any]], pd.DataFrame]) -> None:
        """Batch record multiple transactions into the sliding window."""
        if isinstance(transactions, pd.DataFrame):
            records = transactions.to_dict(orient="records")
        elif isinstance(transactions, list):
            records = transactions
        else:
            return

        for record in records:
            self.record_transaction(record)

    def get_buffer_dataframe(self) -> pd.DataFrame:
        """Retrieve a snapshot DataFrame of currently buffered transactions."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame(columns=EXPECTED_FEATURES)
            return pd.DataFrame(list(self._buffer))

    def get_buffer_size(self) -> int:
        """Get current count of transactions in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear_buffer(self) -> None:
        """Clear all transactions from the buffer."""
        with self._lock:
            self._buffer.clear()

    def evaluate_drift(
        self,
        current_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        sample_size: int = 15000,
        timeout: float = 90.0
    ) -> Dict[str, Any]:
        """
        Run statistical drift evaluation comparing current data against the baseline.

        Args:
            current_data: Optional DataFrame or list of dicts. If None, uses buffered data.
            sample_size: Max sample rows from reference dataset.
            timeout: Subprocess timeout in seconds.

        Returns:
            Dict containing dataset_drift flag, drift_share, and column breakdown.
        """
        # 1. Resolve current dataset
        if current_data is not None:
            if isinstance(current_data, list):
                cur_df = pd.DataFrame(current_data)
            elif isinstance(current_data, pd.DataFrame):
                cur_df = current_data.copy()
            else:
                return {
                    "status": "error",
                    "error": f"Invalid current_data type: {type(current_data).__name__}",
                    "dataset_drift": False
                }
        else:
            cur_df = self.get_buffer_dataframe()

        # Minimum sample size validation (at least 10 rows required for reliable statistics)
        if len(cur_df) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Buffer contains {len(cur_df)} samples. Minimum 10 samples required for drift evaluation.",
                "dataset_drift": False,
                "current_rows": len(cur_df),
                "required_rows": 10,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # 2. Check prerequisites
        if not self.reference_path.exists():
            return {
                "status": "error",
                "error": f"Reference dataset missing at {self.reference_path}",
                "dataset_drift": False
            }

        if not self.monitoring_python.exists():
            return {
                "status": "error",
                "error": f"Monitoring Python environment missing at {self.monitoring_python}",
                "dataset_drift": False
            }

        if not self.runner_script.exists():
            return {
                "status": "error",
                "error": f"Drift runner script missing at {self.runner_script}",
                "dataset_drift": False
            }

        # 3. Write current data to temporary Parquet file
        temp_dir = Path(tempfile.gettempdir())
        temp_cur_file = temp_dir / f"nexvia_cur_{os.getpid()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.parquet"

        try:
            cur_df[EXPECTED_FEATURES].to_parquet(temp_cur_file, index=False)

            # 4. Invoke drift runner in isolated environment
            cmd = [
                str(self.monitoring_python),
                str(self.runner_script),
                "--reference", str(self.reference_path),
                "--current", str(temp_cur_file),
                "--output-html", str(self.output_html),
                "--output-json", str(self.output_json),
                "--sample-size", str(sample_size)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"Drift runner failed with code {result.returncode}: {result.stderr}")
                return {
                    "status": "error",
                    "error": f"Drift execution failed: {result.stderr.strip()}",
                    "dataset_drift": False
                }

            # 5. Parse output JSON
            try:
                metrics = json.loads(result.stdout.strip().splitlines()[-1])
                metrics["status"] = "success"
                self._last_metrics = metrics
                return metrics
            except Exception as parse_err:
                logger.warning(f"Could not parse stdout, falling back to output JSON file: {parse_err}")
                if self.output_json.exists():
                    with open(self.output_json, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                        metrics["status"] = "success"
                        self._last_metrics = metrics
                        return metrics
                return {
                    "status": "error",
                    "error": f"Failed to parse drift output: {parse_err}",
                    "dataset_drift": False
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Drift evaluation exceeded timeout ({timeout}s)",
                "dataset_drift": False
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "dataset_drift": False
            }
        finally:
            if temp_cur_file.exists():
                try:
                    temp_cur_file.unlink()
                except OSError:
                    pass

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Return the most recently computed drift metrics or a default status."""
        if self._last_metrics is not None:
            res = dict(self._last_metrics)
            if "status" not in res:
                res["status"] = "success"
            res["buffered_transactions"] = self.get_buffer_size()
            res["report_available"] = self.output_html.exists()
            return res

        return {
            "status": "not_evaluated",
            "message": "No drift evaluation has been executed yet.",
            "dataset_drift": False,
            "drift_share": 0.0,
            "number_of_drifted_columns": 0,
            "number_of_columns": len(EXPECTED_FEATURES),
            "buffered_transactions": self.get_buffer_size(),
            "report_available": self.output_html.exists(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Singleton drift monitor instance for application-wide sharing
drift_monitor = FraudDriftMonitor()


def evaluate_drift_and_trigger_retraining(
    current_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    sample_size: int = 15000,
    auto_promote: bool = False,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Combined workflow:
    1. Evaluates drift on current data using Evidently AI.
    2. Checks if drift metrics breach retraining thresholds.
    3. Triggers candidate retraining if thresholds are breached.
    """
    from src.retrain import retraining_controller

    drift_res = drift_monitor.evaluate_drift(current_data=current_data, sample_size=sample_size)
    should_retrain, trigger_reason = retraining_controller.should_trigger(drift_res)

    retrain_res = None
    if should_retrain or force_retrain:
        retrain_res = retraining_controller.run_retraining(
            auto_promote=auto_promote,
            force=force_retrain,
            trigger_reason=f"Evidently Drift Alert: {trigger_reason}" if should_retrain else "Forced retrain"
        )

    return {
        "drift_evaluation": drift_res,
        "retraining_triggered": should_retrain or force_retrain,
        "trigger_reason": trigger_reason,
        "retraining_result": retrain_res
    }

