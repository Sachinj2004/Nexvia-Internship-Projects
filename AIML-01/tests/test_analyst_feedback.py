"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Unit Tests: tests/test_analyst_feedback.py

Verifies analyst case recording, status updates (New, Under Review, Confirmed Fraud, Legitimate),
SQLite persistence, invalid status rejection, filtering, and summary counts.
"""

from pathlib import Path
import sys
import tempfile
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyst_feedback import (
    VALID_STATUSES,
    get_all_cases,
    get_case_summary,
    init_feedback_db,
    record_case,
    update_case_status,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database path for isolated testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_feedback.db"
        yield db_path


def test_init_db(temp_db):
    """Verify database initialization creates tables and indexes without error."""
    init_feedback_db(temp_db)
    assert temp_db.exists()
    cases = get_all_cases(db_path=temp_db)
    assert cases == []


def test_record_and_get_case(temp_db):
    """Verify recording a case and retrieving it with all fields intact."""
    row_id = record_case(
        transaction_id="TX_1001",
        amount=149.99,
        fraud_probability=0.887,
        risk_label="Fraud",
        model_decision="FLAGGED_FOR_REVIEW",
        db_path=temp_db
    )
    assert row_id > 0

    cases = get_all_cases(db_path=temp_db)
    assert len(cases) == 1
    c = cases[0]
    assert c["transaction_id"] == "TX_1001"
    assert c["amount"] == 149.99
    assert c["fraud_probability"] == 0.887
    assert c["case_status"] == "New"
    assert c["model_decision"] == "FLAGGED_FOR_REVIEW"


def test_update_case_status_valid(temp_db):
    """Verify updating a case status and analyst notes."""
    record_case(
        transaction_id="TX_2002",
        amount=50.00,
        fraud_probability=0.92,
        risk_label="Fraud",
        model_decision="FLAGGED_FOR_REVIEW",
        db_path=temp_db
    )
    cases = get_all_cases(db_path=temp_db)
    case_id = cases[0]["id"]

    success = update_case_status(
        case_id=case_id,
        case_status="Confirmed Fraud",
        analyst_notes="Cardholder confirmed unauthorized transaction.",
        analyst_id="investigator_jane",
        db_path=temp_db
    )
    assert success is True

    updated_cases = get_all_cases(db_path=temp_db)
    u = updated_cases[0]
    assert u["case_status"] == "Confirmed Fraud"
    assert u["analyst_notes"] == "Cardholder confirmed unauthorized transaction."
    assert u["analyst_id"] == "investigator_jane"


def test_update_case_status_invalid(temp_db):
    """Verify that unsupported case statuses raise ValueError."""
    record_case(
        transaction_id="TX_3003",
        amount=10.00,
        fraud_probability=0.1,
        risk_label="Legitimate",
        model_decision="APPROVED",
        db_path=temp_db
    )
    cases = get_all_cases(db_path=temp_db)
    case_id = cases[0]["id"]

    with pytest.raises(ValueError, match="Invalid case_status"):
        update_case_status(case_id=case_id, case_status="INVALID_STATUS", db_path=temp_db)


def test_get_case_summary_counts(temp_db):
    """Verify aggregate summary counts grouped by status."""
    record_case("TX_A", 100.0, 0.9, "Fraud", "FLAGGED_FOR_REVIEW", case_status="New", db_path=temp_db)
    record_case("TX_B", 200.0, 0.8, "Fraud", "FLAGGED_FOR_REVIEW", case_status="Under Review", db_path=temp_db)
    record_case("TX_C", 300.0, 0.95, "Fraud", "FLAGGED_FOR_REVIEW", case_status="Confirmed Fraud", db_path=temp_db)
    record_case("TX_D", 400.0, 0.1, "Legitimate", "APPROVED", case_status="Legitimate", db_path=temp_db)

    summary = get_case_summary(db_path=temp_db)
    assert summary["Total"] == 4
    assert summary["New"] == 1
    assert summary["Under Review"] == 1
    assert summary["Confirmed Fraud"] == 1
    assert summary["Legitimate"] == 1
