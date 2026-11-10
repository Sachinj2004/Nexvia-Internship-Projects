"""
Real-Time Financial Fraud Detection Platform (AIML-01)
In-Memory Mock Kafka Broker: src/streaming/mock_broker.py

Provides a lightweight, thread-safe, in-memory publish-subscribe broker
emulating Kafka topic partitioning, offset tracking, and consumer groups.
Enables end-to-end streaming tests with zero external infrastructure overhead.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InMemoryMessage:
    """Represents a message stored within a mock Kafka topic."""
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    offset: int
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize message value to JSON string."""
        return json.dumps(self.value)


class MockKafkaBroker:
    """
    Thread-safe in-memory message broker simulating Kafka topics,
    consumer groups, and offset management.
    """

    _instance: Optional["MockKafkaBroker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._lock = threading.RLock()
        self._topics: Dict[str, List[InMemoryMessage]] = {}
        self._consumer_offsets: Dict[Tuple[str, str], int] = {}  # (group_id, topic) -> next_offset

    @classmethod
    def get_shared_instance(cls) -> "MockKafkaBroker":
        """Singleton accessor for shared in-process broker simulation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def create_topic(self, topic: str) -> None:
        """Register a new topic if it does not already exist."""
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []

    def publish(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None
    ) -> InMemoryMessage:
        """
        Publish a message to the specified topic.

        Args:
            topic: Target topic name.
            value: Message payload dictionary.
            key: Optional message partition key.

        Returns:
            InMemoryMessage: The committed message containing its assigned offset.
        """
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []

            offset = len(self._topics[topic])
            message = InMemoryMessage(
                topic=topic,
                key=str(key) if key is not None else None,
                value=dict(value),
                offset=offset,
                timestamp=time.time()
            )
            self._topics[topic].append(message)
            return message

    def poll(
        self,
        group_id: str,
        topics: List[str],
        max_messages: int = 10,
        timeout: float = 0.0
    ) -> List[InMemoryMessage]:
        """
        Poll for unread messages across assigned topics for a consumer group.

        Args:
            group_id: Unique consumer group identifier for offset tracking.
            topics: List of topic names to consume from.
            max_messages: Maximum batch size to retrieve.
            timeout: Optional sleep wait time if queue is currently empty.

        Returns:
            List[InMemoryMessage]: Batch of unread messages.
        """
        messages: List[InMemoryMessage] = []
        start_time = time.time()

        while True:
            with self._lock:
                for topic in topics:
                    if len(messages) >= max_messages:
                        break

                    topic_messages = self._topics.get(topic, [])
                    offset_key = (group_id, topic)
                    current_offset = self._consumer_offsets.get(offset_key, 0)

                    available = topic_messages[current_offset: current_offset + (max_messages - len(messages))]
                    if available:
                        messages.extend(available)
                        self._consumer_offsets[offset_key] = current_offset + len(available)

            if messages or (time.time() - start_time) >= timeout:
                break
            time.sleep(0.01)

        return messages

    def get_messages(self, topic: str) -> List[InMemoryMessage]:
        """
        Read-only inspection of all messages published to a topic without advancing offsets.
        """
        with self._lock:
            return list(self._topics.get(topic, []))

    def get_topic_size(self, topic: str) -> int:
        """Return total message count for a topic."""
        with self._lock:
            return len(self._topics.get(topic, []))

    def reset_topic(self, topic: str) -> None:
        """Clear all messages and offsets for a specific topic."""
        with self._lock:
            self._topics[topic] = []
            keys_to_delete = [k for k in self._consumer_offsets if k[1] == topic]
            for k in keys_to_delete:
                del self._consumer_offsets[k]

    def reset_all(self) -> None:
        """Clear all topics and consumer group offsets."""
        with self._lock:
            self._topics.clear()
            self._consumer_offsets.clear()
