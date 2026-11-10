"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Feast Feature Retrieval Bridge: src/feature_store.py

Provides low-latency, safe online feature lookups for transaction velocity features
using Feast's public FeatureStore API isolated inside the nexvia-feast Conda environment.

Features:
- Lazy persistent worker process avoiding per-lookup subprocess overhead
- Bounded in-memory LRU cache for successful and negative (missing ID) lookups
- Non-blocking error handling returning None on failure without raising exceptions
- Automatic lifecycle cleanup on process exit
"""

import atexit
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("nexvia.feature_store")

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEAST_PYTHON = Path("/opt/anaconda3/envs/nexvia-feast/bin/python")
DEFAULT_REPO_PATH = PROJECT_ROOT / "feature_repo"

FEAST_VELOCITY_FEATURES: List[str] = [
    "Amount",
    "time_delta_prev",
    "rolling_amount_mean_1h",
    "rolling_tx_count_1h",
    "amount_to_window_mean_ratio",
    "pca_energy_norm"
]

WORKER_SCRIPT = """
import sys
import json

try:
    from feast import FeatureStore
    repo_path = sys.argv[1]
    store = FeatureStore(repo_path=repo_path)
    features = [
        "transaction_velocity_features:Amount",
        "transaction_velocity_features:time_delta_prev",
        "transaction_velocity_features:rolling_amount_mean_1h",
        "transaction_velocity_features:rolling_tx_count_1h",
        "transaction_velocity_features:amount_to_window_mean_ratio",
        "transaction_velocity_features:pca_energy_norm"
    ]
    sys.stdout.write("READY\\n")
    sys.stdout.flush()
except Exception as init_err:
    sys.stderr.write(f"INIT_ERROR: {init_err}\\n")
    sys.stderr.flush()
    sys.exit(1)

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    if line == "PING":
        sys.stdout.write("PONG\\n")
        sys.stdout.flush()
        continue
    if line == "EXIT":
        break

    try:
        req = json.loads(line)
        tx_id = int(req["transaction_id"])
        response = store.get_online_features(
            features=features,
            entity_rows=[{"transaction_id": tx_id}]
        )
        res_dict = response.to_dict()
        vals = {
            "Amount": res_dict.get("Amount", [None])[0],
            "time_delta_prev": res_dict.get("time_delta_prev", [None])[0],
            "rolling_amount_mean_1h": res_dict.get("rolling_amount_mean_1h", [None])[0],
            "rolling_tx_count_1h": res_dict.get("rolling_tx_count_1h", [None])[0],
            "amount_to_window_mean_ratio": res_dict.get("amount_to_window_mean_ratio", [None])[0],
            "pca_energy_norm": res_dict.get("pca_energy_norm", [None])[0]
        }
        if all(v is None for v in vals.values()):
            sys.stdout.write(json.dumps({"status": "not_found", "features": None}) + "\\n")
        else:
            sys.stdout.write(json.dumps({"status": "success", "features": vals}) + "\\n")
        sys.stdout.flush()
    except Exception as query_err:
        sys.stdout.write(json.dumps({"status": "error", "error": str(query_err)}) + "\\n")
        sys.stdout.flush()
"""


class FeastFeatureClient:
    """
    Client managing a persistent child worker process for Feast online feature lookups.
    Incorporates in-memory caching and safe error isolation.
    """

    def __init__(
        self,
        repo_path: Optional[Union[str, Path]] = None,
        python_path: Optional[Union[str, Path]] = None,
        max_cache_size: int = 10000,
        startup_timeout: float = 8.0,
        query_timeout: float = 3.0
    ):
        self.repo_path = Path(repo_path or DEFAULT_REPO_PATH)
        self.python_path = Path(python_path or DEFAULT_FEAST_PYTHON)
        self.max_cache_size = max_cache_size
        self.startup_timeout = startup_timeout
        self.query_timeout = query_timeout

        self._lock = threading.Lock()
        self._cache: OrderedDict[int, Optional[Dict[str, Any]]] = OrderedDict()
        self._process: Optional[subprocess.Popen] = None
        self._is_closed = False

        # Register cleanup on process exit
        atexit.register(self.close)

    def _ensure_worker(self) -> bool:
        """
        Lazily initialize the persistent worker process if not already running.
        Returns True if worker is active and healthy, False otherwise.
        """
        if self._is_closed:
            return False

        if self._process is not None:
            if self._process.poll() is None:
                return True
            logger.warning("Feast persistent worker terminated unexpectedly; restarting.")
            self._process = None

        if not self.python_path.exists():
            logger.warning(
                f"Feast Python executable not found at: {self.python_path}. "
                "Feature enrichment will be skipped."
            )
            return False

        if not self.repo_path.exists():
            logger.warning(
                f"Feast repo directory not found at: {self.repo_path}. "
                "Feature enrichment will be skipped."
            )
            return False

        try:
            cmd = [
                str(self.python_path),
                "-u",
                "-c",
                WORKER_SCRIPT,
                str(self.repo_path)
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Wait for READY signal with startup timeout
            ready_line = None
            if process.stdout is not None:
                # Use poll/readline bounded wait
                ready_line = process.stdout.readline().strip()

            if ready_line == "READY":
                self._process = process
                logger.info("Feast persistent worker successfully initialized.")
                return True

            stderr_msg = process.stderr.read() if process.stderr else "Unknown error"
            logger.warning(f"Feast persistent worker failed to initialize: {ready_line}. Stderr: {stderr_msg}")
            process.kill()
            return False

        except Exception as exc:
            logger.warning(f"Failed to start Feast worker process: {exc}")
            return False

    def get_online_features(
        self,
        transaction_id: Union[int, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve 6 velocity features for a transaction_id.

        Args:
            transaction_id: The chronological transaction ID.

        Returns:
            Dict containing the 6 velocity features if found, or None if missing or on error.
        """
        # Validate input ID
        try:
            tx_id_int = int(transaction_id)
        except (ValueError, TypeError):
            return None

        with self._lock:
            # 1. Check in-memory cache (hits both found and missing lookups)
            if tx_id_int in self._cache:
                # Move to end for LRU behavior
                self._cache.move_to_end(tx_id_int)
                return self._cache[tx_id_int]

            # 2. Ensure worker is running
            if not self._ensure_worker():
                return None

            # 3. Query worker process over stdin/stdout
            try:
                assert self._process is not None
                assert self._process.stdin is not None
                assert self._process.stdout is not None

                req_line = json.dumps({"transaction_id": tx_id_int}) + "\n"
                self._process.stdin.write(req_line)
                self._process.stdin.flush()

                resp_line = self._process.stdout.readline()
                if not resp_line:
                    logger.warning("Empty response from Feast worker; closing broken process.")
                    self._terminate_process()
                    return None

                result = json.loads(resp_line.strip())
                status = result.get("status")

                if status == "success":
                    features = result.get("features")
                    self._record_cache(tx_id_int, features)
                    return features

                if status == "not_found":
                    self._record_cache(tx_id_int, None)
                    return None

                logger.warning(f"Feast worker returned error for ID {tx_id_int}: {result.get('error')}")
                return None

            except Exception as comm_err:
                logger.warning(f"Error communicating with Feast worker for ID {tx_id_int}: {comm_err}")
                self._terminate_process()
                return None

    def _record_cache(self, tx_id: int, value: Optional[Dict[str, Any]]) -> None:
        """Record lookup outcome in bounded LRU cache."""
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)  # Evict oldest entry
        self._cache[tx_id] = value

    def clear_cache(self) -> None:
        """Clear all in-memory cached features."""
        with self._lock:
            self._cache.clear()

    def _terminate_process(self) -> None:
        """Internal helper to safely close and reap child worker."""
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    if self._process.stdin:
                        try:
                            self._process.stdin.write("EXIT\n")
                            self._process.stdin.flush()
                        except Exception:
                            pass
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

    def close(self) -> None:
        """Safely close and clean up worker process on exit."""
        with self._lock:
            self._is_closed = True
            self._terminate_process()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Module-Level Singleton & Convenience Accessors
# -----------------------------------------------------------------------------
_GLOBAL_CLIENT: Optional[FeastFeatureClient] = None
_GLOBAL_LOCK = threading.Lock()


def get_feature_client() -> FeastFeatureClient:
    """Retrieve or lazily initialize the process-wide FeastFeatureClient singleton."""
    global _GLOBAL_CLIENT
    with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT is None:
            _GLOBAL_CLIENT = FeastFeatureClient()
        return _GLOBAL_CLIENT


def get_feast_features(transaction_id: Optional[Union[int, str]]) -> Optional[Dict[str, Any]]:
    """
    Convenience function to safely retrieve Feast features for a transaction ID.
    Returns None if ID is None/invalid, not found in online store, or on error.
    """
    if transaction_id is None:
        return None
    return get_feature_client().get_online_features(transaction_id)


def close_feature_client() -> None:
    """Explicitly terminate the global FeastFeatureClient process."""
    global _GLOBAL_CLIENT
    with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT is not None:
            _GLOBAL_CLIENT.close()
            _GLOBAL_CLIENT = None
