"""Database models and utilities using SQLite."""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def get_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = "data/surveillance.db"):
    """Initialize database schema."""
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Events table - stores all detected events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                frame_id INTEGER,
                rule_name TEXT,
                detection_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Detections table - individual detections within events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                track_id INTEGER,
                class TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)

        # Tracks table - tracking information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_sec REAL,
                entry_x REAL,
                entry_y REAL,
                exit_x REAL,
                exit_y REAL,
                behavior TEXT,
                FOREIGN KEY (track_id) REFERENCES detections(track_id)
            )
        """)

        # Alerts table - dispatched alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                alert_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                sent_at REAL,
                error_message TEXT,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)

        # Cameras table - camera configuration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT UNIQUE NOT NULL,
                name TEXT,
                rtsp_url TEXT,
                location TEXT,
                enabled INTEGER DEFAULT 1,
                last_seen REAL
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_event
            ON detections(event_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracks_camera
            ON tracks(camera_id)
        """)

        conn.commit()
        print(f"Database initialized: {db_path}")


def save_event(db_path: str, event_type: str, severity: str,
               camera_id: str, frame_id: int = 0,
               rule_name: str = None, detections: List[Dict] = None,
               metadata: Dict = None) -> int:
    """
    Save an event to the database.

    Args:
        db_path: Path to database file
        event_type: Type of event (detection, behavior, alert)
        severity: Severity level (low, medium, high, critical)
        camera_id: Camera identifier
        frame_id: Frame number
        rule_name: Name of triggered rule
        detections: List of detection dictionaries
        metadata: Additional metadata

    Returns:
        Event ID (0 if failed)
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()

            # Insert event
            cursor.execute("""
                INSERT INTO events (timestamp, event_type, severity, camera_id,
                                   frame_id, rule_name, detection_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                event_type,
                severity,
                camera_id,
                frame_id,
                rule_name,
                len(detections) if detections else 0,
                json.dumps(metadata) if metadata else None
            ))

            event_id = cursor.lastrowid

            # Insert detections
            if detections:
                for det in detections:
                    # Normalize bbox - ensure it's a list of 4 floats
                    bbox = det.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        bbox = [0.0, 0.0, 0.0, 0.0]
                    else:
                        try:
                            bbox = [float(x) for x in bbox]
                        except (TypeError, ValueError):
                            bbox = [0.0, 0.0, 0.0, 0.0]

                    cursor.execute("""
                        INSERT INTO detections (event_id, track_id, class,
                                               confidence, bbox_x1, bbox_y1,
                                               bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event_id,
                        det.get("track_id"),
                        det.get("class", "unknown"),
                        float(det.get("confidence", 0.0)),
                        bbox[0], bbox[1], bbox[2], bbox[3]
                    ))

            conn.commit()
            return event_id
    except Exception as e:
        print(f"Database error in save_event: {e}")
        return 0


def get_events(db_path: str, limit: int = 100,
               event_type: str = None,
               severity: str = None,
               camera_id: str = None,
               start_time: float = None,
               end_time: float = None) -> List[Dict[str, Any]]:
    """
    Query events from the database.

    Args:
        db_path: Path to database file
        limit: Maximum number of events
        event_type: Filter by event type
        severity: Filter by severity
        camera_id: Filter by camera
        start_time: Start timestamp filter
        end_time: End timestamp filter

    Returns:
        List of event dictionaries
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event = dict(row)

            # Parse metadata JSON
            if event.get("metadata"):
                try:
                    event["metadata"] = json.loads(event["metadata"])
                except json.JSONDecodeError:
                    event["metadata"] = {}

            # Get associated detections
            cursor.execute(
                "SELECT * FROM detections WHERE event_id = ?",
                (event["id"],)
            )
            event["detections"] = [dict(d) for d in cursor.fetchall()]

            events.append(event)

        return events


def save_track(db_path: str, track_id: int, camera_id: str,
               start_time: float, entry_point: tuple,
               behavior: str = None) -> int:
    """Save or update track information."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tracks (track_id, camera_id, start_time,
                               entry_x, entry_y, behavior)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            track_id, camera_id, start_time,
            entry_point[0], entry_point[1], behavior
        ))

        conn.commit()
        return cursor.lastrowid


def update_track(db_path: str, track_id: int,
                 end_time: float = None,
                 exit_point: tuple = None,
                 behavior: str = None):
    """Update track with end information."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        updates = []
        params = []

        if end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)
            updates.append("duration_sec = ?")
            params.append(end_time - cursor.execute(
                "SELECT start_time FROM tracks WHERE track_id = ?",
                (track_id,)
            ).fetchone()[0])

        if exit_point:
            updates.append("exit_x = ?")
            params.append(exit_point[0])
            updates.append("exit_y = ?")
            params.append(exit_point[1])

        if behavior:
            updates.append("behavior = ?")
            params.append(behavior)

        if updates:
            params.append(track_id)
            cursor.execute(f"""
                UPDATE tracks SET {', '.join(updates)}
                WHERE track_id = ?
            """, params)

            conn.commit()


def get_statistics(db_path: str,
                   start_time: float = None,
                   end_time: float = None) -> Dict[str, Any]:
    """Get summary statistics from the database."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Build time filter
        time_filter = ""
        params = []
        if start_time:
            time_filter += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            time_filter += " AND timestamp <= ?"
            params.append(end_time)

        # Total events
        cursor.execute(f"""
            SELECT COUNT(*) FROM events WHERE 1=1 {time_filter}
        """, params)
        total_events = cursor.fetchone()[0]

        # Events by type
        cursor.execute(f"""
            SELECT event_type, COUNT(*) FROM events
            WHERE 1=1 {time_filter} GROUP BY event_type
        """, params)
        by_type = {r[0]: r[1] for r in cursor.fetchall()}

        # Events by severity
        cursor.execute(f"""
            SELECT severity, COUNT(*) FROM events
            WHERE 1=1 {time_filter} GROUP BY severity
        """, params)
        by_severity = {r[0]: r[1] for r in cursor.fetchall()}

        # Events by camera
        cursor.execute(f"""
            SELECT camera_id, COUNT(*) FROM events
            WHERE 1=1 {time_filter} GROUP BY camera_id
        """, params)
        by_camera = {r[0]: r[1] for r in cursor.fetchall()}

        # Average detections per event
        cursor.execute(f"""
            SELECT AVG(detection_count) FROM events
            WHERE 1=1 {time_filter}
        """, params)
        avg_detections = cursor.fetchone()[0] or 0

        return {
            "total_events": total_events,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_camera": by_camera,
            "avg_detections": avg_detections
        }


def cleanup_old_data(db_path: str, max_age_days: int = 7):
    """Delete data older than max_age_days."""
    cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Delete old events (cascades to detections)
        cursor.execute("""
            DELETE FROM events WHERE timestamp < ?
        """, (cutoff,))

        # Delete old tracks
        cursor.execute("""
            DELETE FROM tracks WHERE start_time < ?
        """, (cutoff,))

        deleted = cursor.rowcount
        conn.commit()

        return deleted
