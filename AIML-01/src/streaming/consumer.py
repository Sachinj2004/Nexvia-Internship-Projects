"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Streaming Consumer: src/streaming/consumer.py

Consumes transactions from the `transactions` stream topic, executes real-time
inference using FraudPredictor (XGBoost pipeline), records latency, and routes
decisions to downstream topics (`fraud-alerts`, `approved-transactions`, or `dead-letter-queue`).
"""

from datetime import datetime, timezone
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.predict import EXPECTED_FEATURES, FraudPredictor
from src.streaming.mock_broker import InMemoryMessage, MockKafkaBroker
from src.feature_store import get_feast_features

logger = logging.getLogger("nexvia.streaming.consumer")


class FraudTransactionConsumer:
    """
    Real-time transaction consumer and scoring engine.
    Integrates directly with the pre-trained XGBoost pipeline via FraudPredictor.
    """

    def __init__(
        self,
        input_topics: Optional[List[str]] = None,
        group_id: str = "nexvia-fraud-consumer-group",
        alerts_topic: str = "fraud-alerts",
        approved_topic: str = "approved-transactions",
        dead_letter_topic: str = "dead-letter-queue",
        threshold: float = 0.5,
        mock_mode: bool = True,
        bootstrap_servers: Optional[str] = None,
        mock_broker: Optional[MockKafkaBroker] = None,
        predictor: Optional[FraudPredictor] = None
    ):
        """
        Initialize the streaming consumer.

        Args:
            input_topics: Topics to subscribe to (defaults to ['transactions']).
            group_id: Consumer group identifier for offset tracking.
            alerts_topic: Destination topic for transactions flagged as fraud.
            approved_topic: Destination topic for legitimate transactions.
            dead_letter_topic: Destination topic for malformed payloads.
            threshold: Fraud decision threshold (0.0 to 1.0).
            mock_mode: Whether to run in mock mode (no broker required).
            bootstrap_servers: Kafka broker connection string (used if mock_mode=False).
            mock_broker: Optional shared instance of MockKafkaBroker.
            predictor: Optional pre-initialized FraudPredictor instance.
        """
        self.input_topics = input_topics or ["transactions"]
        self.group_id = group_id
        self.alerts_topic = alerts_topic
        self.approved_topic = approved_topic
        self.dead_letter_topic = dead_letter_topic
        self.threshold = threshold
        self.mock_mode = mock_mode
        self.bootstrap_servers = bootstrap_servers or "localhost:9092"

        # Re-use existing inference engine
        self.predictor = predictor or FraudPredictor(default_threshold=self.threshold)

        self._kafka_consumer = None
        self._kafka_producer = None

        if self.mock_mode:
            self.mock_broker = mock_broker or MockKafkaBroker.get_shared_instance()
            for topic in self.input_topics + [self.alerts_topic, self.approved_topic, self.dead_letter_topic]:
                self.mock_broker.create_topic(topic)
        else:
            try:
                import confluent_kafka
                self._kafka_consumer = confluent_kafka.Consumer({
                    "bootstrap.servers": self.bootstrap_servers,
                    "group.id": self.group_id,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False  # Disabled: manual commit after confirmed delivery
                })
                self._kafka_consumer.subscribe(self.input_topics)
                self._kafka_producer = confluent_kafka.Producer({
                    "bootstrap.servers": self.bootstrap_servers
                })
            except ImportError as err:
                raise ImportError(
                    "Real Kafka mode requires 'confluent-kafka'. "
                    "Please install confluent-kafka or initialize with mock_mode=True."
                ) from err

    def _commit_message(self, msg: Any) -> bool:
        """
        Explicitly commit offset for a processed message in real Kafka mode (at-least-once).
        Checks both exceptions and returned error results. Returns True if committed or in mock mode;
        logs and returns False on commit failure.
        """
        if self.mock_mode or self._kafka_consumer is None:
            return True

        try:
            result = self._kafka_consumer.commit(message=msg, asynchronous=False)
            if isinstance(result, list):
                for tp in result:
                    err = getattr(tp, "error", None)
                    if err is not None and isinstance(err, (str, Exception)):
                        logger.error(f"Synchronous commit partition error: {err}")
                        return False
            elif result is not None and isinstance(getattr(result, "error", None), (str, Exception)):
                logger.error(f"Synchronous commit returned error: {result.error}")
                return False
            return True
        except Exception as commit_err:
            logger.error(
                f"Failed to commit offset for message on topic {getattr(msg, 'topic', lambda: 'unknown')()} "
                f"[{getattr(msg, 'partition', lambda: 0)()}] at offset {getattr(msg, 'offset', lambda: 'unknown')()}: {commit_err}"
            )
            return False

    def _quarantine_malformed_raw(
        self,
        raw_data: Any,
        key: Optional[str],
        error_reason: str,
        error_type: str
    ) -> Dict[str, Any]:
        """Build and route a dead-letter event for undecodable or malformed raw messages."""
        if isinstance(raw_data, (bytes, bytearray)):
            raw_b64 = base64.b64encode(raw_data).decode("ascii")
            raw_text = raw_data.decode("utf-8", errors="replace")
        elif raw_data is None:
            raw_b64 = None
            raw_text = None
        else:
            raw_bytes = str(raw_data).encode("utf-8", errors="replace")
            raw_b64 = base64.b64encode(raw_bytes).decode("ascii")
            raw_text = str(raw_data)

        error_event = {
            "transaction_id": key or "unknown",
            "status": "REJECTED_MALFORMED_MESSAGE",
            "error": error_reason,
            "error_type": error_type,
            "raw_payload": raw_text,
            "raw_payload_base64": raw_b64,
            "quarantined_at": datetime.now(timezone.utc).isoformat()
        }
        self._publish_result(topic=self.dead_letter_topic, key=str(key or "unknown"), value=error_event)
        error_event["destination_topic"] = self.dead_letter_topic
        return error_event

    def process_message(
        self,
        payload: Dict[str, Any],
        key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single transaction dictionary through the fraud scoring pipeline.

        Args:
            payload: Input transaction payload.
            key: Optional message partition key.

        Returns:
            Dict[str, Any]: Enriched prediction result and routing outcome.
        """
        start_time = time.perf_counter()
        tx_id = payload.get("transaction_id", key or "unknown") if isinstance(payload, dict) else (key or "unknown")

        # 1. Validation & Model Scoring
        try:
            if not isinstance(payload, dict):
                raise ValueError(f"Payload must be a dictionary, received: {type(payload).__name__}")

            # Extract 30 features expected by FraudPredictor
            features = {col: payload[col] for col in EXPECTED_FEATURES if col in payload}

            # Invoke pre-trained XGBoost pipeline via FraudPredictor
            prediction = self.predictor.predict(features, threshold=self.threshold)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

            # Optional Feast feature store enrichment (does not block or fail message processing)
            enriched_features = None
            if isinstance(payload, dict) and "transaction_id" in payload:
                try:
                    enriched_features = get_feast_features(payload["transaction_id"])
                except Exception as feat_err:
                    logger.warning(f"Feast lookup failed for streaming transaction {tx_id}: {feat_err}")
                    enriched_features = None

            # Construct enriched result payload
            result = {
                "transaction_id": tx_id,
                "Time": payload.get("Time"),
                "Amount": payload.get("Amount"),
                "fraud_probability": prediction["fraud_probability"],
                "predicted_class": prediction["predicted_class"],
                "risk_label": prediction["risk_label"],
                "status": prediction["status"],
                "threshold_used": self.threshold,
                "latency_ms": latency_ms,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "enriched_features": enriched_features
            }
            target_topic = self.alerts_topic if prediction["status"] == "FLAGGED_FOR_REVIEW" else self.approved_topic

        except (ValueError, TypeError, KeyError) as schema_exc:
            # Domain schema / data validation error -> quarantine to dead-letter-queue
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            error_event = {
                "transaction_id": tx_id,
                "status": "REJECTED_INVALID_SCHEMA",
                "error": str(schema_exc),
                "error_type": type(schema_exc).__name__,
                "latency_ms": latency_ms,
                "raw_payload": payload,
                "quarantined_at": datetime.now(timezone.utc).isoformat()
            }
            result = error_event
            target_topic = self.dead_letter_topic

        # 2. Downstream Output Routing (Infrastructure)
        # Publishing failures raise directly and are never misclassified as schema rejections
        try:
            self._publish_result(topic=target_topic, key=str(tx_id), value=result)
            result["destination_topic"] = target_topic
            return result
        except Exception as pub_exc:
            logger.error(f"Infrastructure error: Failed to publish transaction {tx_id} to destination topic {target_topic}: {pub_exc}")
            raise

    def _publish_result(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        """Helper to route processed decision to destination topic."""
        if self.mock_mode:
            self.mock_broker.publish(topic=topic, value=value, key=key)
        else:
            delivery_error: List[Any] = []

            def _ack_callback(err: Any, msg: Any) -> None:
                if err is not None:
                    delivery_error.append(err)

            try:
                self._kafka_producer.produce(
                    topic=topic,
                    key=key.encode("utf-8") if key else None,
                    value=json.dumps(value).encode("utf-8"),
                    on_delivery=_ack_callback
                )
            except Exception as err:
                logger.error(f"Failed to publish result to Kafka topic {topic}: {err}")
                raise

            # Bounded delivery confirmation: flush producer buffer and verify broker acknowledgment
            unflushed = self._kafka_producer.flush(timeout=5.0)
            if isinstance(unflushed, (int, float)) and unflushed > 0:
                raise TimeoutError(f"Delivery timed out: {int(unflushed)} message(s) unacknowledged after 5.0s")
            if delivery_error:
                raise RuntimeError(f"Downstream delivery failed: {delivery_error[0]}")

    def poll_and_process(
        self,
        max_messages: int = 10,
        timeout: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Poll incoming messages from subscribed topics and process them sequentially.

        In real Kafka mode, offset commits occur strictly after confirmed delivery
        of the output message (or DLQ quarantine record). If downstream output delivery
        fails, the offset is not committed to enable safe retry and prevent message loss.

        Args:
            max_messages: Maximum batch size to consume.
            timeout: Maximum wait time in seconds.

        Returns:
            List[Dict[str, Any]]: Scored decision payloads.
        """
        results: List[Dict[str, Any]] = []

        if self.mock_mode:
            messages = self.mock_broker.poll(
                group_id=self.group_id,
                topics=self.input_topics,
                max_messages=max_messages,
                timeout=timeout
            )
            for msg in messages:
                raw_val = msg.value
                key = msg.key

                # Mock-mode: preserve dict support and handle malformed byte payloads
                if isinstance(raw_val, (bytes, bytearray)):
                    try:
                        decoded = raw_val.decode("utf-8")
                        payload = json.loads(decoded)
                        if not isinstance(payload, dict):
                            raise ValueError(f"JSON root must be an object/dict, got {type(payload).__name__}")
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as dec_err:
                        dlq_event = self._quarantine_malformed_raw(
                            raw_data=raw_val,
                            key=key,
                            error_reason=str(dec_err),
                            error_type=type(dec_err).__name__
                        )
                        results.append(dlq_event)
                        continue
                elif isinstance(raw_val, dict):
                    payload = raw_val
                else:
                    dlq_event = self._quarantine_malformed_raw(
                        raw_data=raw_val,
                        key=key,
                        error_reason=f"Payload must be a dict or valid JSON bytes, got {type(raw_val).__name__}",
                        error_type="TypeError"
                    )
                    results.append(dlq_event)
                    continue

                res = self.process_message(payload=payload, key=msg.key)
                results.append(res)
        else:
            # Real Kafka consumer poll
            for _ in range(max_messages):
                msg = self._kafka_consumer.poll(timeout)
                if msg is None:
                    break
                if msg.error():
                    # Distinguish partition EOF from fatal Kafka broker errors
                    is_eof = False
                    try:
                        import confluent_kafka
                        is_eof = (msg.error().code() == confluent_kafka.KafkaError._PARTITION_EOF)
                    except Exception:
                        pass

                    if is_eof:
                        logger.debug(f"Reached end of partition for {msg.topic()} [{msg.partition()}]")
                    else:
                        logger.error(f"Kafka consumer encountered error on topic {msg.topic()}: {msg.error()}")
                    continue

                raw_bytes = msg.value()
                key = msg.key().decode("utf-8", errors="replace") if msg.key() else None

                # 1. Decode UTF-8 & parse JSON (preserve malformed raw bytes in Base64 & text to DLQ)
                try:
                    if raw_bytes is None:
                        raise ValueError("Empty message payload (None)")
                    decoded = raw_bytes.decode("utf-8")
                    payload = json.loads(decoded)
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSON root must be an object/dict, got {type(payload).__name__}")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as decode_err:
                    try:
                        dlq_event = self._quarantine_malformed_raw(
                            raw_data=raw_bytes,
                            key=key,
                            error_reason=str(decode_err),
                            error_type=type(decode_err).__name__
                        )
                        results.append(dlq_event)
                    except Exception as dlq_pub_err:
                        logger.error(f"Failed to publish malformed message to DLQ: {dlq_pub_err}. Offset not committed.")
                        break

                    # Commit only after DLQ quarantine output succeeds; halt batch if commit fails
                    if not self._commit_message(msg):
                        logger.error(f"Halting poll batch: offset commit failed for quarantined message {key}")
                        break
                    continue

                # 2. Process transaction and route output
                try:
                    res = self.process_message(payload=payload, key=key)
                    results.append(res)
                except Exception as infra_err:
                    logger.error(
                        f"Infrastructure error routing transaction {key}: {infra_err}. Offset not committed."
                    )
                    break

                # 3. Explicit commit strictly after confirmed output delivery
                if not self._commit_message(msg):
                    logger.error(f"Halting poll batch: offset commit failed for transaction {key}")
                    break

        return results

    def close(self) -> None:
        """Close Kafka connections gracefully."""
        if not self.mock_mode:
            if self._kafka_consumer:
                self._kafka_consumer.close()
            if self._kafka_producer:
                self._kafka_producer.flush(5.0)
