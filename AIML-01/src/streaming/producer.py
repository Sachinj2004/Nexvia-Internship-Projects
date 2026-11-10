"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Streaming Producer: src/streaming/producer.py

Streams real-time credit card transactions to the `transactions` topic.
Supports both in-memory mock mode (zero broker required) and optional
production Kafka brokers via confluent-kafka when configured.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from src.streaming.mock_broker import MockKafkaBroker

logger = logging.getLogger("nexvia.streaming.producer")


class FraudTransactionProducer:
    """
    Modular transaction stream producer supporting both mock in-memory
    routing and real Apache Kafka brokers.
    """

    def __init__(
        self,
        default_topic: str = "transactions",
        mock_mode: bool = True,
        bootstrap_servers: Optional[str] = None,
        compression_type: str = "none",
        mock_broker: Optional[MockKafkaBroker] = None
    ):
        """
        Initialize the streaming producer.

        Args:
            default_topic: Default destination topic for incoming transactions.
            mock_mode: If True, uses the lightweight in-memory mock broker.
            bootstrap_servers: Kafka broker connection string (used only if mock_mode=False).
            compression_type: Compression algorithm ('none', 'gzip', 'snappy', 'lz4', 'zstd'). Defaults to 'none'.
            mock_broker: Optional shared instance of MockKafkaBroker.
        """
        self.default_topic = default_topic
        self.mock_mode = mock_mode
        self.bootstrap_servers = bootstrap_servers or "localhost:9092"
        self.compression_type = compression_type
        self._kafka_producer = None

        if self.mock_mode:
            self.mock_broker = mock_broker or MockKafkaBroker.get_shared_instance()
            self.mock_broker.create_topic(self.default_topic)
        else:
            try:
                import confluent_kafka
                self._kafka_producer = confluent_kafka.Producer({
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": "nexvia-fraud-producer",
                    "compression.type": self.compression_type
                })
            except ImportError as err:
                raise ImportError(
                    "Real Kafka mode requires 'confluent-kafka'. "
                    "Please install confluent-kafka or initialize with mock_mode=True."
                ) from err

    def _delivery_callback(self, err: Any, msg: Any) -> None:
        """Internal callback invoked by librdkafka on delivery success or failure."""
        if err is not None:
            logger.error(f"Kafka message delivery failed: {err}")
        else:
            logger.debug(
                f"Kafka message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

    def send_transaction(
        self,
        transaction: Dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[Union[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Send a single transaction payload to the streaming topic.

        Args:
            transaction: Dictionary containing transaction features.
            topic: Target topic (defaults to self.default_topic).
            key: Optional message key (e.g. transaction_id).

        Returns:
            Dict[str, Any]: Delivery confirmation containing topic, offset, and status.
        """
        target_topic = topic or self.default_topic
        payload = dict(transaction)

        # Inject streaming metadata if absent
        if "emitted_at" not in payload:
            payload["emitted_at"] = datetime.now(timezone.utc).isoformat()

        msg_key = str(key) if key is not None else str(payload.get("transaction_id", ""))

        if self.mock_mode:
            msg = self.mock_broker.publish(topic=target_topic, value=payload, key=msg_key)
            return {
                "topic": target_topic,
                "key": msg_key,
                "offset": msg.offset,
                "status": "DELIVERED",
                "timestamp": msg.timestamp,
                "mock": True
            }
        else:
            # Real Kafka delivery with buffer error protection and callback
            try:
                self._kafka_producer.produce(
                    topic=target_topic,
                    key=msg_key.encode("utf-8") if msg_key else None,
                    value=json.dumps(payload).encode("utf-8"),
                    on_delivery=self._delivery_callback
                )
                self._kafka_producer.poll(0)
                return {
                    "topic": target_topic,
                    "key": msg_key,
                    "status": "QUEUED",
                    "mock": False
                }
            except BufferError as b_err:
                logger.warning(f"Kafka producer buffer full, polling before retry: {b_err}")
                self._kafka_producer.poll(0.5)
                self._kafka_producer.produce(
                    topic=target_topic,
                    key=msg_key.encode("utf-8") if msg_key else None,
                    value=json.dumps(payload).encode("utf-8"),
                    on_delivery=self._delivery_callback
                )
                return {
                    "topic": target_topic,
                    "key": msg_key,
                    "status": "QUEUED_AFTER_BUFFER_FLUSH",
                    "mock": False
                }
            except Exception as exc:
                logger.error(f"Failed to produce message to Kafka topic {target_topic}: {exc}")
                raise

    def stream_records(
        self,
        records: Iterable[Dict[str, Any]],
        topic: Optional[str] = None,
        delay_seconds: float = 0.0
    ) -> int:
        """
        Stream a sequence of transaction records.

        Args:
            records: Iterable of transaction dictionaries.
            topic: Target topic.
            delay_seconds: Artificial throttle delay between emissions.

        Returns:
            int: Total count of messages delivered.
        """
        count = 0
        for record in records:
            self.send_transaction(transaction=record, topic=topic)
            count += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        self.flush()
        return count

    def stream_from_csv(
        self,
        csv_path: Union[str, Path],
        nrows: Optional[int] = None,
        topic: Optional[str] = None,
        delay_seconds: float = 0.0
    ) -> int:
        """
        Read transactions from a CSV file and stream them sequentially.

        Args:
            csv_path: Path to the credit card CSV dataset.
            nrows: Optional limit on rows to stream.
            topic: Target topic name.
            delay_seconds: Artificial delay between messages.

        Returns:
            int: Number of transactions streamed.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV data file not found: {path}")

        df = pd.read_csv(path, nrows=nrows)
        records = df.to_dict(orient="records")
        return self.stream_records(records, topic=topic, delay_seconds=delay_seconds)

    def flush(self, timeout: float = 5.0) -> None:
        """Flush pending producer buffers."""
        if not self.mock_mode and self._kafka_producer is not None:
            self._kafka_producer.flush(timeout)
