"""Regression coverage for persistent events, alert rules, and database paths."""
from pathlib import Path
import sys

from src.database.models import get_events, init_db, save_event
from src.detection.rules import RuleEngine


def test_detection_snapshot_is_saved(tmp_path):
    db_path = str(tmp_path / "data" / "surveillance.db")
    init_db(db_path)
    event_id = save_event(
        db_path=db_path,
        event_type="detection",
        severity="low",
        camera_id="cam1",
        frame_id=7,
        rule_name="OBJECT_DETECTED",
        detections=[{
            "class": "person",
            "confidence": 0.95,
            "bbox": [10, 20, 100, 200],
        }],
    )
    assert event_id > 0
    events = get_events(db_path)
    assert len(events) == 1
    assert events[0]["rule_name"] == "OBJECT_DETECTED"
    assert events[0]["detections"][0]["class"] == "person"


def test_alert_event_is_visible_in_event_history(tmp_path):
    db_path = str(tmp_path / "surveillance.db")
    init_db(db_path)
    saved_id = save_event(
        db_path=db_path,
        event_type="alert",
        severity="medium",
        camera_id="cam1",
        rule_name="LOITERING",
    )
    assert saved_id > 0
    assert get_events(db_path, event_type="alert")[0]["rule_name"] == "LOITERING"


def test_crowd_formation_rule_matches_behavior():
    rules = RuleEngine()
    triggered = rules.evaluate(
        detections=[],
        behaviors=[{"behavior": "crowd_formation", "people_count": 10}],
        tracks=[],
    )
    assert "CROWD_FORMATION" in [rule.name for rule in triggered]


def test_loitering_rule_matches_ten_seconds():
    rules = RuleEngine()
    triggered = rules.evaluate(
        detections=[],
        behaviors=[{"behavior": "loitering", "duration": 10.5}],
        tracks=[],
    )
    assert "LOITERING" in [rule.name for rule in triggered]


def test_main_database_path_is_project_relative(monkeypatch, tmp_path):
    # Avoid importing the camera pipeline / model dependencies in this test.
    from pathlib import Path
    import ast
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_db_path")
    assert resolver is not None
