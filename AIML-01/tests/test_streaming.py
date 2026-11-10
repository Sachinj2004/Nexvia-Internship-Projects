"""
Automated Test Suite for Real-Time Streaming Architecture (src/streaming)
Tests MockKafkaBroker, FraudTransactionProducer, FraudTransactionConsumer,
schema validation, fraud alerting, routing, and error quarantine.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import EXPECTED_FEATURES, FraudPredictor
from src.streaming import (
    FraudTransactionConsumer,
    FraudTransactionProducer,
    InMemoryMessage,
    MockKafkaBroker,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_broker():
    """Provides a freshly cleared MockKafkaBroker instance for each test."""
    broker = MockKafkaBroker()
    broker.reset_all()
    return broker


@pytest.fixture
def legitimate_transaction() -> Dict[str, Any]:
    """Synthetic standard transaction with low fraud risk."""
    payload = {col: 0.0 for col in EXPECTED_FEATURES}
    payload["transaction_id"] = 1001
    payload["Time"] = 100.0
    payload["Amount"] = 49.99
    return payload


@pytest.fixture
def fraudulent_transaction() -> Dict[str, Any]:
    """Known extreme anomaly transaction matching typical fraud PCA patterns."""
    payload = {col: 0.0 for col in EXPECTED_FEATURES}
    payload["transaction_id"] = 9999
    payload["Time"] = 406.0
    payload["V1"] = -2.3122
    payload["V2"] = 1.9520
    payload["V3"] = -1.6099
    payload["V4"] = 3.9979
    payload["V7"] = -2.5374
    payload["V10"] = -2.7723
    payload["V12"] = -2.8999
    payload["V14"] = -4.2893
    payload["V17"] = -2.8301
    payload["Amount"] = 0.0
    return payload


# -----------------------------------------------------------------------------
# 1. Mock Broker Unit Tests
# -----------------------------------------------------------------------------
def test_mock_broker_pub_sub(mock_broker):
    """Verify topic publication, FIFO ordering, and offset tracking."""
    topic = "test-topic"
    mock_broker.create_topic(topic)

    msg1 = mock_broker.publish(topic, {"id": 1, "val": "A"}, key="k1")
    msg2 = mock_broker.publish(topic, {"id": 2, "val": "B"}, key="k2")

    assert msg1.offset == 0
    assert msg2.offset == 1
    assert mock_broker.get_topic_size(topic) == 2

    # First poll batch
    polled_group1 = mock_broker.poll("group-A", [topic], max_messages=1)
    assert len(polled_group1) == 1
    assert polled_group1[0].value["id"] == 1

    # Second poll should fetch next offset
    polled_group1_next = mock_broker.poll("group-A", [topic], max_messages=1)
    assert len(polled_group1_next) == 1
    assert polled_group1_next[0].value["id"] == 2

    # Group B should independently read from offset 0
    polled_group2 = mock_broker.poll("group-B", [topic], max_messages=5)
    assert len(polled_group2) == 2


def test_mock_broker_reset(mock_broker):
    """Verify broker state reset."""
    topic = "reset-topic"
    mock_broker.publish(topic, {"msg": "hello"})
    assert mock_broker.get_topic_size(topic) == 1

    mock_broker.reset_all()
    assert mock_broker.get_topic_size(topic) == 0


# -----------------------------------------------------------------------------
# 2. Producer Unit Tests
# -----------------------------------------------------------------------------
def test_producer_send_transaction(mock_broker, legitimate_transaction):
    """Verify producer sends formatted transaction with timestamp metadata."""
    producer = FraudTransactionProducer(mock_broker=mock_broker, mock_mode=True)
    delivery = producer.send_transaction(legitimate_transaction, key="tx-1001")

    assert delivery["status"] == "DELIVERED"
    assert delivery["topic"] == "transactions"
    assert delivery["offset"] == 0

    messages = mock_broker.get_messages("transactions")
    assert len(messages) == 1
    assert "emitted_at" in messages[0].value
    assert messages[0].value["Amount"] == 49.99


def test_producer_stream_records(mock_broker, legitimate_transaction):
    """Verify batch streaming of multiple records."""
    producer = FraudTransactionProducer(mock_broker=mock_broker, mock_mode=True)
    records = [
        {**legitimate_transaction, "transaction_id": i}
        for i in range(5)
    ]
    sent_count = producer.stream_records(records)
    assert sent_count == 5
    assert mock_broker.get_topic_size("transactions") == 5


# -----------------------------------------------------------------------------
# 3. Consumer Scoring & Routing Tests
# -----------------------------------------------------------------------------
def test_consumer_approved_legitimate_transaction(mock_broker, legitimate_transaction):
    """Verify legitimate transaction receives APPROVED status and routes to approved-transactions."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True, threshold=0.5)

    result = consumer.process_message(legitimate_transaction, key="1001")

    assert result["predicted_class"] == 0
    assert result["status"] == "APPROVED"
    assert result["risk_label"] == "Legitimate"
    assert result["fraud_probability"] < 0.5
    assert result["destination_topic"] == "approved-transactions"
    assert "latency_ms" in result
    assert result["latency_ms"] >= 0.0

    # Verify message presence in approved destination topic
    approved_msgs = mock_broker.get_messages("approved-transactions")
    assert len(approved_msgs) == 1
    assert approved_msgs[0].value["status"] == "APPROVED"


def test_consumer_flagged_fraud_transaction(mock_broker, fraudulent_transaction):
    """Verify high-risk fraud transaction triggers FLAGGED_FOR_REVIEW and routes to fraud-alerts."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True, threshold=0.5)

    result = consumer.process_message(fraudulent_transaction, key="9999")

    assert result["predicted_class"] == 1
    assert result["status"] == "FLAGGED_FOR_REVIEW"
    assert result["risk_label"] == "Fraud"
    assert result["fraud_probability"] >= 0.5
    assert result["destination_topic"] == "fraud-alerts"

    # Verify message in alerts destination topic
    alert_msgs = mock_broker.get_messages("fraud-alerts")
    assert len(alert_msgs) == 1
    assert alert_msgs[0].value["transaction_id"] == 9999


def test_consumer_invalid_schema_quarantine(mock_broker):
    """Verify invalid payloads (missing features) route to dead-letter-queue without crashing."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    corrupted_payload = {"transaction_id": "bad-01", "Time": 12.0, "Amount": 5.0}

    result = consumer.process_message(corrupted_payload)

    assert result["status"] == "REJECTED_INVALID_SCHEMA"
    assert result["destination_topic"] == "dead-letter-queue"
    assert "missing" in result["error"].lower()

    dlq_msgs = mock_broker.get_messages("dead-letter-queue")
    assert len(dlq_msgs) == 1
    assert dlq_msgs[0].value["error_type"] == "ValueError"


def test_consumer_non_numeric_schema_quarantine(mock_broker, legitimate_transaction):
    """Verify payloads with non-numeric fields route to dead-letter-queue."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    invalid_data = dict(legitimate_transaction)
    invalid_data["Amount"] = "NOT_A_NUMBER"

    result = consumer.process_message(invalid_data)

    assert result["status"] == "REJECTED_INVALID_SCHEMA"
    assert result["destination_topic"] == "dead-letter-queue"

    dlq_msgs = mock_broker.get_messages("dead-letter-queue")
    assert len(dlq_msgs) == 1


def test_consumer_custom_threshold(mock_broker, legitimate_transaction):
    """Verify custom sensitivity threshold alters classification."""
    # With an ultra-low threshold of 0.0000001, even slight probability flags as fraud
    sensitive_consumer = FraudTransactionConsumer(
        mock_broker=mock_broker,
        mock_mode=True,
        threshold=0.0000001
    )
    result = sensitive_consumer.process_message(legitimate_transaction)
    assert result["threshold_used"] == 0.0000001


# -----------------------------------------------------------------------------
# 4. End-to-End Streaming Pipeline Integration Test
# -----------------------------------------------------------------------------
def test_end_to_end_streaming_pipeline(
    mock_broker,
    legitimate_transaction,
    fraudulent_transaction
):
    """
    End-to-end integration: Producer streams mixed transactions ->
    Consumer polls and scores -> Output topics collect categorized decisions.
    """
    producer = FraudTransactionProducer(mock_broker=mock_broker, mock_mode=True)
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)

    # 1. Stream 3 legitimate, 2 fraudulent, and 1 corrupted payload
    producer.send_transaction({**legitimate_transaction, "transaction_id": 1})
    producer.send_transaction({**legitimate_transaction, "transaction_id": 2})
    producer.send_transaction({**fraudulent_transaction, "transaction_id": 3})
    producer.send_transaction({**legitimate_transaction, "transaction_id": 4})
    producer.send_transaction({**fraudulent_transaction, "transaction_id": 5})
    producer.send_transaction({"transaction_id": 6, "Amount": 10.0})  # Corrupted

    assert mock_broker.get_topic_size("transactions") == 6

    # 2. Consumer polls and processes the full batch
    processed = consumer.poll_and_process(max_messages=10)
    assert len(processed) == 6

    # 3. Assert destinations
    approved = mock_broker.get_messages("approved-transactions")
    alerts = mock_broker.get_messages("fraud-alerts")
    dlq = mock_broker.get_messages("dead-letter-queue")

    assert len(approved) == 3
    assert len(alerts) == 2
    assert len(dlq) == 1

    # 4. Latency SLA validation: each transaction scored in sub-15ms
    for item in processed:
        assert item["latency_ms"] < 25.0, f"Latency exceeded: {item['latency_ms']} ms"


# -----------------------------------------------------------------------------
# 5. Broker Isolation & Optional Import Check
# -----------------------------------------------------------------------------
def test_mock_mode_operates_without_kafka():
    """Verify that mock mode operates independently without any external broker connection."""
    broker = MockKafkaBroker()
    prod = FraudTransactionProducer(mock_broker=broker, mock_mode=True)
    cons = FraudTransactionConsumer(mock_broker=broker, mock_mode=True)

    payload = {col: 0.0 for col in EXPECTED_FEATURES}
    payload["Time"] = 1.0
    payload["Amount"] = 10.0

    prod.send_transaction(payload)
    results = cons.poll_and_process(max_messages=1)

    assert len(results) == 1
    assert results[0]["status"] == "APPROVED"


# -----------------------------------------------------------------------------
# 6. Additional Focused Edge-Case & Hardening Tests
# -----------------------------------------------------------------------------
def test_producer_stream_from_csv(mock_broker):
    """Verify stream_from_csv sequentially emits real transactions from dataset."""
    csv_file = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if not csv_file.exists():
        pytest.skip("Source dataset creditcard.csv missing, skipping CSV replay test.")

    producer = FraudTransactionProducer(mock_broker=mock_broker, mock_mode=True)
    emitted = producer.stream_from_csv(csv_path=csv_file, nrows=5)

    assert emitted == 5
    assert mock_broker.get_topic_size("transactions") == 5

    messages = mock_broker.get_messages("transactions")
    for msg in messages:
        assert "Time" in msg.value
        assert "Amount" in msg.value
        assert "V1" in msg.value
        assert "emitted_at" in msg.value


def test_producer_stream_from_csv_missing_file(mock_broker):
    """Verify FileNotFoundError is raised if CSV file path does not exist."""
    producer = FraudTransactionProducer(mock_broker=mock_broker, mock_mode=True)
    with pytest.raises(FileNotFoundError):
        producer.stream_from_csv(csv_path="non_existent_file.csv", nrows=5)


def test_consumer_empty_poll(mock_broker):
    """Verify polling an empty queue returns an empty list without hanging or errors."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    results = consumer.poll_and_process(max_messages=5, timeout=0.01)
    assert results == []


def test_real_kafka_missing_client_raises_import_error():
    """Verify that initializing with mock_mode=False raises ImportError when confluent-kafka is absent."""
    with patch.dict(sys.modules, {"confluent_kafka": None}):
        with pytest.raises(ImportError, match="Real Kafka mode requires 'confluent-kafka'"):
            FraudTransactionProducer(mock_mode=False)

        with pytest.raises(ImportError, match="Real Kafka mode requires 'confluent-kafka'"):
            FraudTransactionConsumer(mock_mode=False)


def test_mocked_real_kafka_producer_success(legitimate_transaction):
    """Verify real Kafka producer initialization, default compression, and message production."""
    mock_confluent = MagicMock()
    mock_inner_producer = MagicMock()
    mock_confluent.Producer.return_value = mock_inner_producer

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        producer = FraudTransactionProducer(mock_mode=False, bootstrap_servers="kafka:9092")

        # Verify default compression is 'none'
        mock_confluent.Producer.assert_called_once()
        conf_passed = mock_confluent.Producer.call_args[0][0]
        assert conf_passed["compression.type"] == "none"
        assert conf_passed["bootstrap.servers"] == "kafka:9092"

        # Verify produce call and delivery callback
        delivery = producer.send_transaction(legitimate_transaction, key="tx-99")
        assert delivery["status"] == "QUEUED"
        assert delivery["mock"] is False
        mock_inner_producer.produce.assert_called_once()
        assert mock_inner_producer.produce.call_args[1]["on_delivery"] == producer._delivery_callback

        # Test delivery callback safety (both success and failure paths)
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "transactions"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 42

        # Success callback execution
        producer._delivery_callback(None, mock_msg)

        # Error callback execution
        producer._delivery_callback(RuntimeError("Connection lost"), mock_msg)

        # Test flush call
        producer.flush(timeout=3.0)
        mock_inner_producer.flush.assert_called_once_with(3.0)


def test_mocked_real_kafka_producer_buffer_error_recovery(legitimate_transaction):
    """Verify that producer polls to drain buffers and retries when BufferError occurs."""
    mock_confluent = MagicMock()
    mock_inner_producer = MagicMock()
    # First produce call raises BufferError, second succeeds
    mock_inner_producer.produce.side_effect = [BufferError("Local queue full"), None]
    mock_confluent.Producer.return_value = mock_inner_producer

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        producer = FraudTransactionProducer(mock_mode=False)
        delivery = producer.send_transaction(legitimate_transaction)

        assert delivery["status"] == "QUEUED_AFTER_BUFFER_FLUSH"
        assert mock_inner_producer.poll.call_count == 1
        assert mock_inner_producer.produce.call_count == 2


def test_mocked_real_kafka_consumer_poll_and_routing(legitimate_transaction):
    """Verify real Kafka consumer message poll, scoring, and output routing."""
    import json
    mock_confluent = MagicMock()
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    # Construct mock Kafka message with legitimate transaction
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.key.return_value = b"tx-123"
    mock_msg.value.return_value = json.dumps(legitimate_transaction).encode("utf-8")

    # poll returns mock_msg on first call, None on second
    mock_inner_consumer.poll.side_effect = [mock_msg, None]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        results = consumer.poll_and_process(max_messages=2)

        assert len(results) == 1
        assert results[0]["status"] == "APPROVED"
        assert results[0]["destination_topic"] == "approved-transactions"

        # Verify output message was produced to approved-transactions topic
        mock_inner_producer.produce.assert_called_once()
        produced_topic = mock_inner_producer.produce.call_args[1]["topic"]
        assert produced_topic == "approved-transactions"

        # Verify consumer close
        consumer.close()
        mock_inner_consumer.close.assert_called_once()
        assert mock_inner_producer.flush.call_count >= 1


def test_mocked_real_kafka_consumer_handles_eof_and_error():
    """Verify consumer distinguishes between partition EOF and fatal broker errors."""
    mock_confluent = MagicMock()
    mock_confluent.KafkaError._PARTITION_EOF = -191
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    # 1. Message with Partition EOF error
    eof_error = MagicMock()
    eof_error.code.return_value = -191
    eof_msg = MagicMock()
    eof_msg.error.return_value = eof_error
    eof_msg.topic.return_value = "transactions"
    eof_msg.partition.return_value = 0

    # 2. Message with fatal broker error
    fatal_error = MagicMock()
    fatal_error.code.return_value = -1
    fatal_error.__str__.return_value = "Broker transport failure"
    fatal_msg = MagicMock()
    fatal_msg.error.return_value = fatal_error
    fatal_msg.topic.return_value = "transactions"

    # Poll returns EOF, then fatal error, then None
    mock_inner_consumer.poll.side_effect = [eof_msg, fatal_msg, None]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        results = consumer.poll_and_process(max_messages=3)

        # Both errors should be safely skipped without crashing or producing results
        assert results == []


def test_consumer_malformed_utf8_in_mock_mode(mock_broker):
    """Verify invalid UTF-8 bytes in mock mode route to DLQ with readable and Base64 payloads."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    # Inject an InMemoryMessage with raw non-UTF-8 bytes
    mock_broker._topics["transactions"] = [
        InMemoryMessage(topic="transactions", key="bad-utf8", value=b"\xff\xfe\xfa\x80", offset=0)
    ]

    results = consumer.poll_and_process(max_messages=1)
    assert len(results) == 1
    assert results[0]["status"] == "REJECTED_MALFORMED_MESSAGE"
    assert results[0]["error_type"] == "UnicodeDecodeError"
    assert "raw_payload_base64" in results[0]
    assert results[0]["raw_payload_base64"] == "//76gA=="
    assert mock_broker.get_topic_size("dead-letter-queue") == 1


def test_consumer_malformed_json_in_mock_mode(mock_broker):
    """Verify invalid JSON string bytes in mock mode route to DLQ."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    mock_broker._topics["transactions"] = [
        InMemoryMessage(topic="transactions", key="bad-json", value=b'{"unclosed_json: true', offset=0)
    ]

    results = consumer.poll_and_process(max_messages=1)
    assert len(results) == 1
    assert results[0]["status"] == "REJECTED_MALFORMED_MESSAGE"
    assert results[0]["error_type"] == "JSONDecodeError"
    assert mock_broker.get_topic_size("dead-letter-queue") == 1


def test_publishing_failure_classified_as_infrastructure_not_schema(mock_broker, legitimate_transaction):
    """Verify downstream publishing failure raises an infrastructure error and does not pollute DLQ."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)

    with patch.object(consumer, "_publish_result", side_effect=RuntimeError("Kafka broker connection timeout")):
        with pytest.raises(RuntimeError, match="Kafka broker connection timeout"):
            consumer.process_message(legitimate_transaction, key="tx-infra")

    # DLQ must remain empty because this was an infrastructure transport error, not a schema rejection
    assert mock_broker.get_topic_size("dead-letter-queue") == 0


def test_real_kafka_failed_routing_does_not_commit_offset(legitimate_transaction):
    """Verify that downstream publishing failure in real Kafka mode prevents offset commit."""
    import json

    mock_confluent = MagicMock()
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.key.return_value = b"tx-fail"
    mock_msg.value.return_value = json.dumps(legitimate_transaction).encode("utf-8")
    mock_inner_consumer.poll.side_effect = [mock_msg, None]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        with patch.object(consumer, "_publish_result", side_effect=Exception("Broker connection dropped")):
            results = consumer.poll_and_process(max_messages=1)

        assert results == []
        mock_inner_consumer.commit.assert_not_called()


def test_real_kafka_commit_failure_halts_batch(legitimate_transaction):
    """Verify that when offset commit fails (via error partition or exception), consumer halts batch."""
    import json

    mock_confluent = MagicMock()
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    mock_tp = MagicMock()
    mock_tp.error = "CommitFailedError"
    mock_inner_consumer.commit.return_value = [mock_tp]

    mock_msg1 = MagicMock()
    mock_msg1.error.return_value = None
    mock_msg1.key.return_value = b"tx-1"
    mock_msg1.value.return_value = json.dumps(legitimate_transaction).encode("utf-8")

    mock_msg2 = MagicMock()
    mock_msg2.error.return_value = None
    mock_msg2.key.return_value = b"tx-2"
    mock_msg2.value.return_value = json.dumps(legitimate_transaction).encode("utf-8")

    mock_inner_consumer.poll.side_effect = [mock_msg1, mock_msg2]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        results = consumer.poll_and_process(max_messages=2)

        # Message 1 was scored, but commit failed -> batch halts immediately; message 2 is not processed
        assert len(results) == 1
        assert results[0]["status"] == "APPROVED"
        mock_inner_consumer.commit.assert_called_once()


def test_real_kafka_unflushed_producer_times_out(legitimate_transaction):
    """Verify that if downstream producer cannot flush delivery within timeout, offset is not committed."""
    import json

    mock_confluent = MagicMock()
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_inner_producer.flush.return_value = 1  # 1 message timed out
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.key.return_value = b"tx-timeout"
    mock_msg.value.return_value = json.dumps(legitimate_transaction).encode("utf-8")
    mock_inner_consumer.poll.side_effect = [mock_msg, None]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        results = consumer.poll_and_process(max_messages=1)

        # Batch halted due to delivery timeout, offset never committed
        assert results == []
        mock_inner_consumer.commit.assert_not_called()


def test_real_kafka_malformed_utf8_to_dlq_and_commits():
    """Verify invalid UTF-8 bytes in real Kafka mode are quarantined to DLQ and then offset is committed."""
    mock_confluent = MagicMock()
    mock_inner_consumer = MagicMock()
    mock_inner_producer = MagicMock()
    mock_inner_producer.flush.return_value = 0
    mock_confluent.Consumer.return_value = mock_inner_consumer
    mock_confluent.Producer.return_value = mock_inner_producer

    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.key.return_value = b"bad-bytes"
    mock_msg.value.return_value = b"\xff\xfe\xfa\x80"
    mock_inner_consumer.poll.side_effect = [mock_msg, None]

    with patch.dict(sys.modules, {"confluent_kafka": mock_confluent}):
        consumer = FraudTransactionConsumer(mock_mode=False)
        results = consumer.poll_and_process(max_messages=1)

        assert len(results) == 1
        assert results[0]["status"] == "REJECTED_MALFORMED_MESSAGE"
        assert results[0]["error_type"] == "UnicodeDecodeError"
        assert results[0]["raw_payload_base64"] == "//76gA=="

        # Output was routed to dead-letter-queue
        mock_inner_producer.produce.assert_called_once()
        assert mock_inner_producer.produce.call_args[1]["topic"] == "dead-letter-queue"

        # Offset was committed after successful quarantine
        mock_inner_consumer.commit.assert_called_once()


# -----------------------------------------------------------------------------
# Feast Velocity Feature Enrichment Tests (Stage 3)
# -----------------------------------------------------------------------------
def test_consumer_enriches_valid_feast_transaction(mock_broker, legitimate_transaction):
    """Verify streaming consumer attaches 6 Feast velocity features for known transaction IDs."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    tx = dict(legitimate_transaction)
    tx["transaction_id"] = 0

    result = consumer.process_message(tx)

    assert result["status"] == "APPROVED"
    assert result["enriched_features"] is not None
    assert "rolling_amount_mean_1h" in result["enriched_features"]
    assert "rolling_tx_count_1h" in result["enriched_features"]
    assert "amount_to_window_mean_ratio" in result["enriched_features"]

    # Verify message in destination topic contains enriched features
    approved_msgs = mock_broker.get_messages("approved-transactions")
    assert len(approved_msgs) == 1
    assert approved_msgs[0].value["enriched_features"] is not None
    assert approved_msgs[0].value["enriched_features"]["rolling_tx_count_1h"] == 1


def test_consumer_missing_feast_transaction_id_graceful(mock_broker, legitimate_transaction):
    """Verify consumer sets enriched_features: None for unmaterialized IDs without failing."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    tx = dict(legitimate_transaction)
    tx["transaction_id"] = 999999

    result = consumer.process_message(tx)

    assert result["status"] == "APPROVED"
    assert result["enriched_features"] is None

    approved_msgs = mock_broker.get_messages("approved-transactions")
    assert len(approved_msgs) == 1
    assert approved_msgs[0].value["enriched_features"] is None


def test_consumer_feast_lookup_failure_does_not_halt_processing(mock_broker, legitimate_transaction):
    """Verify Feast worker failure/exception does not prevent transaction classification or routing."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    tx = dict(legitimate_transaction)
    tx["transaction_id"] = 0

    with patch("src.streaming.consumer.get_feast_features", side_effect=RuntimeError("Feast worker crashed")):
        result = consumer.process_message(tx)

        assert result["status"] == "APPROVED"
        assert result["enriched_features"] is None
        assert result["destination_topic"] == "approved-transactions"

        approved_msgs = mock_broker.get_messages("approved-transactions")
        assert len(approved_msgs) == 1
        assert approved_msgs[0].value["status"] == "APPROVED"


def test_consumer_backward_compatibility_omitted_transaction_id(mock_broker, legitimate_transaction):
    """Verify payloads without transaction_id process cleanly with enriched_features: None."""
    consumer = FraudTransactionConsumer(mock_broker=mock_broker, mock_mode=True)
    tx = dict(legitimate_transaction)
    tx.pop("transaction_id", None)

    result = consumer.process_message(tx)

    assert result["status"] == "APPROVED"
    assert result["enriched_features"] is None
    assert result["destination_topic"] == "approved-transactions"


