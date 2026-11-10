"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Analyst Feedback & Case Management Store: src/analyst_feedback.py

Provides thread-safe local SQLite persistence for analyst review cases, ground-truth
feedback labels, case statuses (New, Under Review, Confirmed Fraud, Legitimate),
and investigation notes. Survives application restarts.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("nexvia.analyst_feedback")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "analyst_feedback.db"

VALID_STATUSES = {"New", "Under Review", "Confirmed Fraud", "Legitimate"}


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Create a SQLite connection with row factory enabled."""
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize the feedback database table and indexes if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_label TEXT NOT NULL,
                model_decision TEXT NOT NULL,
                case_status TEXT NOT NULL DEFAULT 'New',
                analyst_notes TEXT DEFAULT '',
                analyst_id TEXT DEFAULT 'analyst_1',
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_status ON cases(case_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transaction_id ON cases(transaction_id)")
        conn.commit()


def record_case(
    transaction_id: str,
    amount: float,
    fraud_probability: float,
    risk_label: str,
    model_decision: str,
    timestamp: Optional[str] = None,
    case_status: str = "New",
    analyst_notes: str = "",
    analyst_id: str = "system",
    db_path: Optional[Union[str, Path]] = None
) -> int:
    """
    Record an alert or transaction for analyst review.
    If the transaction_id already exists, updates its probability and timestamp.
    """
    init_feedback_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    ts = timestamp or now_iso

    if case_status not in VALID_STATUSES:
        raise ValueError(f"Invalid case_status '{case_status}'. Allowed: {VALID_STATUSES}")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cases (
                transaction_id, timestamp, amount, fraud_probability,
                risk_label, model_decision, case_status, analyst_notes, analyst_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transaction_id) DO UPDATE SET
                fraud_probability = excluded.fraud_probability,
                risk_label = excluded.risk_label,
                model_decision = excluded.model_decision,
                updated_at = excluded.updated_at
        """, (
            str(transaction_id),
            ts,
            float(amount),
            float(fraud_probability),
            str(risk_label),
            str(model_decision),
            case_status,
            str(analyst_notes),
            str(analyst_id),
            now_iso
        ))
        conn.commit()
        return cursor.lastrowid or 0


def update_case_status(
    case_id: int,
    case_status: str,
    analyst_notes: Optional[str] = None,
    analyst_id: str = "analyst_1",
    db_path: Optional[Union[str, Path]] = None
) -> bool:
    """
    Update the investigation status and optional notes for a case.
    """
    if case_status not in VALID_STATUSES:
        raise ValueError(f"Invalid case_status '{case_status}'. Allowed: {VALID_STATUSES}")

    init_feedback_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if analyst_notes is not None:
            cursor.execute("""
                UPDATE cases
                SET case_status = ?, analyst_notes = ?, analyst_id = ?, updated_at = ?
                WHERE id = ?
            """, (case_status, str(analyst_notes), str(analyst_id), now_iso, int(case_id)))
        else:
            cursor.execute("""
                UPDATE cases
                SET case_status = ?, analyst_id = ?, updated_at = ?
                WHERE id = ?
            """, (case_status, str(analyst_id), now_iso, int(case_id)))
        conn.commit()
        return cursor.rowcount > 0


def get_all_cases(
    status_filter: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieve cases ordered by most recently updated."""
    init_feedback_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if status_filter and status_filter in VALID_STATUSES:
            cursor.execute("""
                SELECT * FROM cases
                WHERE case_status = ?
                ORDER BY id DESC
                LIMIT ?
            """, (status_filter, limit))
        else:
            cursor.execute("""
                SELECT * FROM cases
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_case_summary(db_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """Retrieve count of cases grouped by status."""
    init_feedback_db(db_path)
    summary = {s: 0 for s in VALID_STATUSES}
    summary["Total"] = 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT case_status, COUNT(*) as cnt FROM cases GROUP BY case_status")
        for row in cursor.fetchall():
            st = row["case_status"]
            cnt = int(row["cnt"])
            if st in summary:
                summary[st] = cnt
            summary["Total"] += cnt

    return summary
