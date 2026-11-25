"""Trajectory analysis for tracked objects."""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time


class TrajectoryAnalyzer:
    """
    Analyzes movement patterns and trajectories of tracked objects.
    Detects: loitering, wrong-way movement, speed anomalies.
    """

    def __init__(self, frame_size: Tuple[int, int] = (1280, 720)):
        self.frame_size = frame_size
        self.trajectories: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        self.zone_entries: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.speed_history: Dict[int, List[float]] = defaultdict(list)

        # Define zones (normalized coordinates)
        self.zones = {
            "entrance": [(0.0, 0.5), (0.3, 1.0)],
            "exit": [(0.7, 0.5), (1.0, 1.0)],
            "restricted": [(0.4, 0.3), (0.6, 0.7)],
        }

    def update(self, tracks: List[Dict[str, Any]],
               frame_time: float) -> List[Dict[str, Any]]:
        """
        Update trajectories and analyze movement.

        Args:
            tracks: List of tracked objects with bbox and track_id
            frame_time: Current timestamp

        Returns:
            List of trajectory-based detections
        """
        detections = []

        for track in tracks:
            track_id = track.get("track_id")
            if track_id is None:
                continue

            bbox = track.get("bbox")
            if bbox is None:
                continue

            # Get center point
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Normalize coordinates
            norm_x = center_x / self.frame_size[0]
            norm_y = center_y / self.frame_size[1]

            # Update trajectory
            self.trajectories[track_id].append((norm_x, norm_y, frame_time))

            # Keep a time-based history long enough for the 10-second
            # loitering threshold, regardless of processing frame rate.
            cutoff = frame_time - 15.0
            self.trajectories[track_id] = [
                point for point in self.trajectories[track_id]
                if point[2] >= cutoff
            ]

            # Analyze behavior
            behavior = self._analyze_track(track_id, frame_time)
            if behavior:
                detections.append(behavior)

        return detections

    def _analyze_track(self, track_id: int,
                       current_time: float) -> Optional[Dict[str, Any]]:
        """Analyze a single track for suspicious behavior."""
        trajectory = self.trajectories[track_id]

        if len(trajectory) < 15:  # Need at least 1 second of data
            return None

        # Check for loitering
        loitering = self._check_loitering(track_id, current_time)
        if loitering:
            return loitering

        # Check for wrong-way movement
        wrong_way = self._check_wrong_way(track_id)
        if wrong_way:
            return wrong_way

        # Check for speed anomaly
        speed_anomaly = self._check_speed_anomaly(track_id)
        if speed_anomaly:
            return speed_anomaly

        return None

    def _check_loitering(self, track_id: int,
                         current_time: float,
                         threshold_sec: float = 10.0) -> Optional[Dict[str, Any]]:
        """Check if object is loitering in one area."""
        trajectory = self.trajectories[track_id]

        if len(trajectory) < 30:  # Need 2 seconds of data
            return None

        # Get oldest point in window
        oldest_time = trajectory[0][2]
        time_span = current_time - oldest_time

        if time_span < threshold_sec:
            return None

        # Calculate bounding box of trajectory
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        # If movement is confined to small area
        if x_range < 0.1 and y_range < 0.1:  # 10% of frame
            return {
                "behavior": "loitering",
                "track_id": track_id,
                "confidence": min(1.0, time_span / 30.0),
                "duration": time_span
            }

        return None

    def _check_wrong_way(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Check if object is moving in wrong direction."""
        trajectory = self.trajectories[track_id]

        if len(trajectory) < 20:
            return None

        # Simple heuristic: check if moving right-to-left in exit zone
        recent = trajectory[-20:]

        start_x = recent[0][0]
        end_x = recent[-1][0]
        avg_y = sum(p[1] for p in recent) / len(recent)

        # Exit zone is on the right (0.7-1.0)
        if avg_y > 0.5 and start_x > end_x:  # Moving left in exit zone
            # Check if in exit zone
            if start_x > 0.7 or end_x > 0.7:
                return {
                    "behavior": "wrong_way",
                    "track_id": track_id,
                    "confidence": 0.8
                }

        return None

    def _check_speed_anomaly(self, track_id: int,
                             min_speed: float = 0.02,
                             max_speed: float = 0.15) -> Optional[Dict[str, Any]]:
        """Check for abnormal speed (too fast or too slow)."""
        trajectory = self.trajectories[track_id]

        if len(trajectory) < 10:
            return None

        # Calculate speed over recent trajectory
        recent = trajectory[-10:]
        total_distance = 0.0

        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        time_span = recent[-1][2] - recent[0][2]
        if time_span <= 0:
            return None

        speed = total_distance / time_span
        self.speed_history[track_id].append(speed)

        # Keep last 30 speed measurements
        if len(self.speed_history[track_id]) > 30:
            self.speed_history[track_id] = self.speed_history[track_id][-30:]

        # Check for running (too fast)
        if speed > max_speed:
            return {
                "behavior": "running",
                "track_id": track_id,
                "confidence": min(1.0, speed / 0.2),
                "speed": speed
            }

        return None

    def check_zone_entry(self, track_id: int,
                         zone_name: str) -> bool:
        """Check if track entered a specific zone."""
        trajectory = self.trajectories.get(track_id, [])

        if len(trajectory) < 2:
            return False

        zone = self.zones.get(zone_name)
        if not zone:
            return False

        [(x1, y1), (x2, y2)] = zone

        # Check if current position is in zone
        curr_x, curr_y, _ = trajectory[-1]

        in_zone = (x1 <= curr_x <= x2) and (y1 <= curr_y <= y2)

        # Check if previously outside zone
        prev_x, prev_y, _ = trajectory[-2]
        was_outside = not ((x1 <= prev_x <= x2) and (y1 <= prev_y <= y2))

        if in_zone and was_outside:
            self.zone_entries[track_id][zone_name] = time.time()
            return True

        return False

    def get_track_direction(self, track_id: int) -> Optional[str]:
        """Get primary direction of movement."""
        trajectory = self.trajectories.get(track_id, [])

        if len(trajectory) < 10:
            return None

        start_x, start_y, _ = trajectory[0]
        end_x, end_y, _ = trajectory[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float, float]]:
        """Get full trajectory for a track."""
        return list(self.trajectories.get(track_id, []))

    def cleanup(self, active_track_ids: set):
        """Remove trajectories for inactive tracks."""
        inactive = set(self.trajectories.keys()) - active_track_ids
        for track_id in inactive:
            del self.trajectories[track_id]
            if track_id in self.zone_entries:
                del self.zone_entries[track_id]
            if track_id in self.speed_history:
                del self.speed_history[track_id]
