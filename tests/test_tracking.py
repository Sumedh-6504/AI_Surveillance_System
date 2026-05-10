"""Unit tests for tracking modules."""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.deepsort import DeepSORTTracker, KalmanFilter, Track
from src.tracking.trajectory import TrajectoryAnalyzer


class TestKalmanFilter:
    """Tests for Kalman filter."""

    def test_kalman_initialization(self):
        """Test Kalman filter initializes correctly."""
        kf = KalmanFilter(dt=1.0)
        assert kf.state is None
        assert kf.P.shape == (6, 6)
        assert kf.F.shape == (6, 6)
        assert kf.H.shape == (4, 6)

    def test_kalman_first_update(self):
        """Test first update initializes state."""
        kf = KalmanFilter()
        measurement = np.array([100, 100, 50, 50])  # x, y, w, h
        kf.update(measurement)

        assert kf.state is not None
        assert len(kf.state) == 6
        # Center should be at measurement position
        assert kf.state[0] == 100  # center x
        assert kf.state[1] == 100  # center y

    def test_kalman_predict_after_update(self):
        """Test prediction works after update."""
        kf = KalmanFilter()
        kf.update(np.array([100, 100, 50, 50]))

        predicted = kf.predict()
        assert predicted is not None
        # Should return bbox format
        assert len(predicted) == 4


class TestTrack:
    """Tests for Track class."""

    def test_track_creation(self):
        """Test track creation from bbox."""
        bbox = np.array([50, 50, 150, 150])
        track = Track(1, bbox)

        assert track.track_id == 1
        assert track.age == 1
        assert track.consecutive_absence == 0
        assert len(track.history) == 1

    def test_track_update(self):
        """Test track state update."""
        bbox = np.array([50, 50, 150, 150])
        track = Track(1, bbox)

        # Update with new position
        new_bbox = np.array([60, 60, 160, 160])
        track.update(new_bbox)

        assert track.age == 2
        assert track.total_count == 2
        assert track.consecutive_absence == 0
        assert len(track.history) == 2

    def test_track_mark_missed(self):
        """Test marking track as missed."""
        bbox = np.array([50, 50, 150, 150])
        track = Track(1, bbox)

        track.mark_missed()
        assert track.consecutive_absence == 1

        track.mark_missed()
        assert track.consecutive_absence == 2

    def test_track_predict(self):
        """Test track prediction."""
        bbox = np.array([50, 50, 150, 150])
        track = Track(1, bbox)

        predicted = track.predict()
        assert predicted is not None
        assert len(predicted) == 4


class TestDeepSORTTracker:
    """Tests for DeepSORT tracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = DeepSORTTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 0

    def test_tracker_creates_tracks(self):
        """Test tracker creates tracks from detections."""
        tracker = DeepSORTTracker(min_hits=1)

        detections = [
            {"bbox": [50, 50, 150, 150], "class": "person", "confidence": 0.9},
            {"bbox": [200, 200, 300, 300], "class": "person", "confidence": 0.8}
        ]

        tracks = tracker.update(detections)

        assert len(tracks) == 2
        assert tracks[0]["track_id"] == 0
        assert tracks[1]["track_id"] == 1

    def test_tracker_maintains_ids(self):
        """Test tracker maintains consistent IDs across frames."""
        tracker = DeepSORTTracker(min_hits=1)

        # First frame
        detections1 = [{"bbox": [50, 50, 150, 150], "class": "person", "confidence": 0.9}]
        tracks1 = tracker.update(detections1)
        first_id = tracks1[0]["track_id"]

        # Second frame - same object, slightly moved
        detections2 = [{"bbox": [55, 55, 155, 155], "class": "person", "confidence": 0.9}]
        tracks2 = tracker.update(detections2)

        # Should maintain same track ID
        assert tracks2[0]["track_id"] == first_id

    def test_tracker_new_tracks_for_new_objects(self):
        """Test tracker creates new tracks for new objects."""
        tracker = DeepSORTTracker(min_hits=1)

        # First detection
        detections1 = [{"bbox": [50, 50, 150, 150], "class": "person", "confidence": 0.9}]
        tracks1 = tracker.update(detections1)
        first_id = tracks1[0]["track_id"]

        # Second detection - far away (new object)
        detections2 = [{"bbox": [500, 500, 600, 600], "class": "person", "confidence": 0.9}]
        tracks2 = tracker.update(detections2)

        # Should have new track ID
        assert tracks2[0]["track_id"] != first_id

    def test_iou_calculation(self):
        """Test IoU calculation."""
        tracker = DeepSORTTracker()

        # Identical boxes
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([0, 0, 100, 100])
        iou = tracker._iou(box1, box2)
        assert iou == 1.0

        # Non-overlapping boxes
        box3 = np.array([200, 200, 300, 300])
        iou = tracker._iou(box1, box3)
        assert iou == 0.0

        # Partially overlapping
        box4 = np.array([50, 50, 150, 150])
        iou = tracker._iou(box1, box4)
        assert 0 < iou < 1


class TestTrajectoryAnalyzer:
    """Tests for trajectory analysis."""

    def test_analyzer_initialization(self):
        """Test trajectory analyzer initializes correctly."""
        analyzer = TrajectoryAnalyzer(frame_size=(1280, 720))
        assert analyzer.frame_size == (1280, 720)
        assert len(analyzer.trajectories) == 0

    def test_trajectory_update(self):
        """Test trajectory updates."""
        analyzer = TrajectoryAnalyzer(frame_size=(1280, 720))

        tracks = [
            {"track_id": 1, "bbox": [100, 100, 200, 200]}
        ]

        detections = analyzer.update(tracks, time.time())

        assert len(analyzer.trajectories[1]) == 1
        # Normalized coordinates
        assert 0 < analyzer.trajectories[1][0][0] < 1
        assert 0 < analyzer.trajectories[1][0][1] < 1

    def test_loitering_detection(self):
        """Test loitering detection."""
        analyzer = TrajectoryAnalyzer(frame_size=(1280, 720))

        # Add stationary positions
        base_time = time.time()
        for i in range(35):
            tracks = [
                {"track_id": 1, "bbox": [100, 100, 200, 200]}
            ]
            analyzer.update(tracks, base_time + i)

        # Should detect loitering
        detections = analyzer.update(tracks, base_time + 36)

        loitering = [d for d in detections if d.get("behavior") == "loitering"]
        assert len(loitering) > 0

    def test_track_direction(self):
        """Test direction calculation."""
        analyzer = TrajectoryAnalyzer(frame_size=(1280, 720))

        # Simulate rightward movement
        base_time = time.time()
        for i in range(15):
            x = 100 + i * 10
            tracks = [{"track_id": 1, "bbox": [x, 100, x + 50, 200]}]
            analyzer.update(tracks, base_time + i)

        direction = analyzer.get_track_direction(1)
        assert direction == "right"

    def test_cleanup(self):
        """Test cleanup of inactive tracks."""
        analyzer = TrajectoryAnalyzer(frame_size=(1280, 720))

        # Add tracks
        analyzer.update([{"track_id": 1, "bbox": [100, 100, 200, 200]}], time.time())
        analyzer.update([{"track_id": 2, "bbox": [300, 300, 400, 400]}], time.time())

        assert len(analyzer.trajectories) == 2

        # Cleanup keeping only track 1
        analyzer.cleanup({1})

        assert 1 in analyzer.trajectories
        assert 2 not in analyzer.trajectories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
