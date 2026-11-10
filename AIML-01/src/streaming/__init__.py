"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Streaming Package: src/streaming/__init__.py
"""

from src.streaming.consumer import FraudTransactionConsumer
from src.streaming.mock_broker import InMemoryMessage, MockKafkaBroker
from src.streaming.producer import FraudTransactionProducer

__all__ = [
    "InMemoryMessage",
    "MockKafkaBroker",
    "FraudTransactionProducer",
    "FraudTransactionConsumer",
]
