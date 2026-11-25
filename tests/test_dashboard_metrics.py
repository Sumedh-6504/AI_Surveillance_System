"""Regression tests for event statistics and dashboard threat scoring."""
from src.database.models import init_db, save_event, get_statistics
from src.dashboard.metrics import compute_threat_level


def test_alert_statistics_exclude_routine_detections(tmp_path):
    db_path = str(tmp_path / "surveillance.db")
    init_db(db_path)
    for _ in range(12):
        assert save_event(db_path, "detection", "low", "cam1", rule_name="OBJECT_DETECTED") > 0
    assert save_event(db_path, "alert", "medium", "cam1", rule_name="LOITERING") > 0
    all_stats = get_statistics(db_path)
    alert_stats = get_statistics(db_path, event_type="alert")
    assert all_stats["total_events"] == 13
    assert alert_stats["total_events"] == 1
    assert alert_stats["by_severity"] == {"medium": 1}
    assert alert_stats["by_type"] == {"LOITERING": 1}


def test_routine_detection_does_not_increase_threat_level():
    assert compute_threat_level(0, 0, 0, 0) == ("CLEAR", "low", 0)


def test_severity_priority():
    assert compute_threat_level(0, 0, 1) == ("MODERATE", "medium", 50)
    assert compute_threat_level(0, 1, 1) == ("HIGH", "high", 75)
    assert compute_threat_level(1, 1, 1) == ("CRITICAL", "critical", 100)
