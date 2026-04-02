"""Unit tests for database modules."""
import pytest
import sqlite3
import time
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import (
    init_db, save_event, get_events, save_track,
    update_track, get_statistics, cleanup_old_data,
    get_connection
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    try:
        init_db(path)
        yield path
    finally:
        import os
        os.close(fd)
        os.unlink(path)


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_init_creates_tables(self, temp_db):
        """Test initialization creates all tables."""
        with get_connection(temp_db) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='events'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='detections'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='tracks'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='alerts'
            """)
            assert cursor.fetchone() is not None

    def test_init_creates_indexes(self, temp_db):
        """Test initialization creates indexes."""
        with get_connection(temp_db) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_events_timestamp'
            """)
            assert cursor.fetchone() is not None

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_detections_event'
            """)
            assert cursor.fetchone() is not None


class TestSaveEvent:
    """Tests for saving events."""

    def test_save_event_basic(self, temp_db):
        """Test saving a basic event."""
        event_id = save_event(
            db_path=temp_db,
            event_type="detection",
            severity="medium",
            camera_id="cam1",
            frame_id=100
        )

        assert event_id > 0

        # Verify saved
        events = get_events(temp_db, limit=1)
        assert len(events) == 1
        assert events[0]["event_type"] == "detection"
        assert events[0]["camera_id"] == "cam1"
        assert events[0]["frame_id"] == 100

    def test_save_event_with_detections(self, temp_db):
        """Test saving event with detections."""
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 200]},
            {"class": "bag", "confidence": 0.75, "bbox": [300, 300, 350, 350]}
        ]

        event_id = save_event(
            db_path=temp_db,
            event_type="alert",
            severity="high",
            camera_id="cam1",
            frame_id=100,
            rule_name="SUSPICIOUS_OBJECT",
            detections=detections
        )

        events = get_events(temp_db, limit=1)
        assert len(events) == 1

        event = events[0]
        assert event["detection_count"] == 2
        assert len(event["detections"]) == 2
        assert event["detections"][0]["class"] == "person"

    def test_save_event_with_metadata(self, temp_db):
        """Test saving event with metadata."""
        metadata = {"location": "entrance", "zone": "A"}

        event_id = save_event(
            db_path=temp_db,
            event_type="behavior",
            severity="low",
            camera_id="cam1",
            metadata=metadata
        )

        events = get_events(temp_db, limit=1)
        assert events[0]["metadata"] == metadata


class TestGetEvents:
    """Tests for querying events."""

    def test_get_events_limit(self, temp_db):
        """Test event limit parameter."""
        for i in range(50):
            save_event(
                db_path=temp_db,
                event_type="detection",
                severity="low",
                camera_id="cam1"
            )

        events = get_events(temp_db, limit=10)
        assert len(events) == 10

    def test_get_events_filter_by_type(self, temp_db):
        """Test filtering events by type."""
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "alert", "high", "cam1")
        save_event(temp_db, "behavior", "medium", "cam1")

        # Filter by detection
        events = get_events(temp_db, event_type="detection")
        assert len(events) == 1
        assert events[0]["event_type"] == "detection"

    def test_get_events_filter_by_severity(self, temp_db):
        """Test filtering events by severity."""
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "detection", "high", "cam1")
        save_event(temp_db, "detection", "critical", "cam1")

        events = get_events(temp_db, severity="high")
        assert len(events) == 1
        assert events[0]["severity"] == "high"

    def test_get_events_filter_by_camera(self, temp_db):
        """Test filtering events by camera."""
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "detection", "low", "cam2")
        save_event(temp_db, "detection", "low", "cam1")

        events = get_events(temp_db, camera_id="cam2")
        assert len(events) == 1

    def test_get_events_filter_by_time(self, temp_db):
        """Test filtering events by time range."""
        now = time.time()

        save_event(temp_db, "detection", "low", "cam1")
        time.sleep(0.1)
        mid_time = time.time()
        time.sleep(0.1)
        save_event(temp_db, "detection", "low", "cam1")

        events = get_events(temp_db, start_time=mid_time)
        assert len(events) == 1


class TestTracks:
    """Tests for track operations."""

    def test_save_track(self, temp_db):
        """Test saving a track."""
        track_id = save_track(
            db_path=temp_db,
            track_id=1,
            camera_id="cam1",
            start_time=time.time(),
            entry_point=(100, 200),
            behavior="walking"
        )

        assert track_id > 0

    def test_update_track(self, temp_db):
        """Test updating a track."""
        start_time = time.time()
        save_track(
            db_path=temp_db,
            track_id=1,
            camera_id="cam1",
            start_time=start_time,
            entry_point=(100, 200)
        )

        end_time = start_time + 30
        update_track(
            db_path=temp_db,
            track_id=1,
            end_time=end_time,
            exit_point=(500, 600),
            behavior="running"
        )

        # Verify update (check via raw query since we don't have get_track)
        with get_connection(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE track_id = 1")
            row = dict(cursor.fetchone())

            assert row["exit_x"] == 500
            assert row["exit_y"] == 600
            assert row["behavior"] == "running"


class TestStatistics:
    """Tests for statistics."""

    def test_get_statistics(self, temp_db):
        """Test getting statistics."""
        # Add various events
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "alert", "high", "cam1")
        save_event(temp_db, "alert", "critical", "cam2")

        stats = get_statistics(temp_db)

        assert stats["total_events"] == 4
        assert stats["by_type"]["detection"] == 2
        assert stats["by_type"]["alert"] == 2
        assert stats["by_severity"]["low"] == 2
        assert stats["by_severity"]["high"] == 1
        assert stats["by_severity"]["critical"] == 1
        assert stats["by_camera"]["cam1"] == 3
        assert stats["by_camera"]["cam2"] == 1

    def test_get_statistics_time_filter(self, temp_db):
        """Test statistics with time filter."""
        now = time.time()
        hour_ago = now - 3600

        # Old events
        with get_connection(temp_db) as conn:
            cursor = conn.cursor()
            for _ in range(5):
                cursor.execute("""
                    INSERT INTO events (timestamp, event_type, severity, camera_id)
                    VALUES (?, ?, ?, ?)
                """, (hour_ago - 100, "detection", "low", "cam1"))
            conn.commit()

        # Recent events
        save_event(temp_db, "detection", "low", "cam1")
        save_event(temp_db, "alert", "high", "cam1")

        stats = get_statistics(temp_db, start_time=hour_ago)
        assert stats["total_events"] == 2


class TestCleanup:
    """Tests for data cleanup."""

    def test_cleanup_old_data(self, temp_db):
        """Test cleaning up old data."""
        now = time.time()
        old_time = now - (10 * 24 * 60 * 60)  # 10 days ago

        # Add old event
        with get_connection(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (timestamp, event_type, severity, camera_id)
                VALUES (?, ?, ?, ?)
            """, (old_time, "detection", "low", "cam1"))
            conn.commit()

        # Add recent event
        save_event(temp_db, "detection", "low", "cam1")

        # Cleanup data older than 7 days
        deleted = cleanup_old_data(temp_db, max_age_days=7)

        assert deleted == 1

        # Only recent event should remain
        events = get_events(temp_db, limit=10)
        assert len(events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
