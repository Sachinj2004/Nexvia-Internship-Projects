"""
Automated Test Suite for Feast Feature Store Bridge (src/feature_store.py)
Tests:
- Valid transaction ID lookups (IDs 0 and 1) returning all 6 features
- Missing transaction ID lookup (ID 999999) returning None
- In-memory caching for found and missing records
- Graceful error handling and mock worker failure recovery
- Diagnostic latency measurements
"""

from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import (
    FEAST_VELOCITY_FEATURES,
    FeastFeatureClient,
    close_feature_client,
    get_feast_features,
    get_feature_client
)


@pytest.fixture(autouse=True)
def cleanup_clients():
    """Ensure feature clients are closed and cache cleared after each test."""
    yield
    close_feature_client()


# -----------------------------------------------------------------------------
# 1. Real Worker Feature Retrieval Tests
# -----------------------------------------------------------------------------
def test_valid_transaction_id_0_retrieval():
    """Verify that valid transaction ID 0 returns all 6 Feast velocity features."""
    client = get_feature_client()
    features = client.get_online_features(0)

    assert features is not None, "Feast features should be returned for transaction_id 0"
    for expected_col in FEAST_VELOCITY_FEATURES:
        assert expected_col in features, f"Missing expected feature {expected_col}"

    assert isinstance(features["Amount"], (float, int))
    assert features["rolling_tx_count_1h"] == 1
    assert features["time_delta_prev"] == 0.0
    assert features["pca_energy_norm"] > 0.0


def test_valid_transaction_id_1_retrieval():
    """Verify that valid transaction ID 1 returns expected features."""
    client = get_feature_client()
    features = client.get_online_features(1)

    assert features is not None, "Feast features should be returned for transaction_id 1"
    assert features["rolling_tx_count_1h"] == 2
    assert features["amount_to_window_mean_ratio"] > 0.0


def test_missing_transaction_id_returns_none():
    """Verify that an unmaterialized transaction ID (999999) safely returns None."""
    client = get_feature_client()
    features = client.get_online_features(999999)

    assert features is None, "Missing transaction ID should return None"


def test_convenience_function_with_none_and_invalid_id():
    """Verify get_feast_features safely handles None and unparseable input."""
    assert get_feast_features(None) is None
    assert get_feast_features("not_an_int") is None


# -----------------------------------------------------------------------------
# 2. Caching Verification Tests
# -----------------------------------------------------------------------------
def test_cache_hits_for_valid_and_missing_records():
    """Verify that lookups populate the cache and subsequent queries hit the cache."""
    client = FeastFeatureClient()
    try:
        # First query: populates cache
        res_0 = client.get_online_features(0)
        assert res_0 is not None
        assert 0 in client._cache

        # Second query: should hit cache directly
        res_0_cached = client.get_online_features(0)
        assert res_0_cached == res_0

        # Query missing ID: should cache None
        res_missing = client.get_online_features(999999)
        assert res_missing is None
        assert 999999 in client._cache
        assert client._cache[999999] is None

        # Subsequent missing ID query returns from cache
        assert client.get_online_features(999999) is None
    finally:
        client.close()


def test_cache_size_eviction():
    """Verify that bounded cache evicts oldest entries when capacity is exceeded."""
    client = FeastFeatureClient(max_cache_size=3)
    try:
        client._record_cache(1, {"Amount": 10.0})
        client._record_cache(2, {"Amount": 20.0})
        client._record_cache(3, {"Amount": 30.0})
        assert len(client._cache) == 3

        # Add 4th entry: entry 1 should be evicted
        client._record_cache(4, {"Amount": 40.0})
        assert len(client._cache) == 3
        assert 1 not in client._cache
        assert 4 in client._cache
    finally:
        client.close()


# -----------------------------------------------------------------------------
# 3. Fault Tolerance & Failure Handling Tests
# -----------------------------------------------------------------------------
def test_worker_startup_failure_graceful_handling():
    """Verify that if the worker cannot be started, lookup returns None without crashing."""
    client = FeastFeatureClient(python_path="/non/existent/python/binary")
    try:
        features = client.get_online_features(0)
        assert features is None, "Should gracefully return None on startup failure"
    finally:
        client.close()


def test_worker_communication_error_graceful_handling():
    """Verify that communication/pipe failures safely return None without crashing."""
    client = FeastFeatureClient()
    try:
        # Mock stdin.write to raise an IOError
        client._ensure_worker()
        assert client._process is not None
        mock_stdin = MagicMock()
        mock_stdin.write.side_effect = BrokenPipeError("Simulated broken pipe")
        client._process.stdin = mock_stdin

        features = client.get_online_features(42)
        assert features is None, "Should gracefully return None on pipe failure"
    finally:
        client.close()


# -----------------------------------------------------------------------------
# 4. Diagnostic Latency Measurement
# -----------------------------------------------------------------------------
def test_diagnostic_latency_measurement():
    """
    Measure and log cold-start vs persistent worker vs in-memory cache lookup timings.
    Does not enforce rigid thresholds to avoid test brittleness.
    """
    client = FeastFeatureClient()
    try:
        # 1. First call (starts worker + performs IPC lookup)
        t0 = time.perf_counter()
        f0 = client.get_online_features(0)
        first_call_ms = (time.perf_counter() - t0) * 1000.0
        assert f0 is not None

        # 2. Second call for different ID (persistent worker IPC without startup)
        t1 = time.perf_counter()
        f1 = client.get_online_features(1)
        ipc_call_ms = (time.perf_counter() - t1) * 1000.0
        assert f1 is not None

        # 3. Third call for cached ID (in-memory LRU cache hit)
        t2 = time.perf_counter()
        f0_cached = client.get_online_features(0)
        cached_call_ms = (time.perf_counter() - t2) * 1000.0
        assert f0_cached is not None

        print(f"\n[Diagnostic Latency Report]")
        print(f"  • Worker initialization + First query: {first_call_ms:.2f} ms")
        print(f"  • Persistent worker IPC query:          {ipc_call_ms:.2f} ms")
        print(f"  • In-memory cache hit query:            {cached_call_ms:.4f} ms")

        # Sanity check: cached lookup is significantly faster than first call
        assert cached_call_ms < first_call_ms
    finally:
        client.close()
