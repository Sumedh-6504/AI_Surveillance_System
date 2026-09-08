"""DeepSORT multi-object tracker."""
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """Simple Kalman filter for tracking."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        # State: [x, y, vx, vy, w, h]
        self.state = None
        self.P = np.eye(6) * 1000  # Covariance
        self.Q = np.eye(6) * 0.01  # Process noise
        self.R = np.eye(4) * 10    # Measurement noise

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

    def predict(self) -> np.ndarray:
        """Predict next state."""
        if self.state is None:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement: np.ndarray):
        """Update state with measurement."""
        if self.state is None:
            self.state = np.array([
                measurement[0], measurement[1], 0, 0,
                measurement[2], measurement[3]
            ])
            return

        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class Track:
    """Represents a single tracked object."""

    def __init__(self, track_id: int, bbox: np.ndarray,
                 cls: str = "person", confidence: float = 1.0):
        self.track_id = track_id
        self.cls = cls
        self.confidence = confidence
        self.kf = KalmanFilter()
        self.kf.state = np.array([
            (bbox[0] + bbox[2]) / 2,  # center x
            (bbox[1] + bbox[3]) / 2,  # center y
            0, 0,  # velocity
            bbox[2] - bbox[0],  # width
            bbox[3] - bbox[1],  # height
        ])
        self.age = 1
        self.total_count = 1
        self.consecutive_absence = 0
        self.history = deque(maxlen=30)
        self.history.append(bbox.copy())

    def predict(self) -> Optional[np.ndarray]:
        """Predict next position."""
        state = self.kf.predict()
        if state is None:
            return None

        x, y, _, _, w, h = state
        return np.array([
            x - w/2, y - h/2,
            x + w/2, y + h/2
        ])

    def update(self, bbox: np.ndarray,
               cls: str = None, confidence: float = None):
        """Update track with new measurement."""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        measurement = np.array([center_x, center_y, w, h])
        self.kf.update(measurement)

        # Refresh class/confidence from the latest detection
        if cls is not None:
            self.cls = cls
        if confidence is not None:
            self.confidence = confidence

        self.age += 1
        self.total_count += 1
        self.consecutive_absence = 0
        self.history.append(bbox.copy())

    def mark_missed(self):
        """Mark track as missed."""
        self.consecutive_absence += 1

    def get_state(self) -> Optional[np.ndarray]:
        """Get current state as bbox."""
        state = self.kf.state
        if state is None:
            return None

        x, y, _, _, w, h = state
        return np.array([
            x - w/2, y - h/2,
            x + w/2, y + h/2
        ])


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker.
    Simplified implementation optimized for CPU.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of {bbox, class, confidence}

        Returns:
            List of tracked objects with track_id
        """
        # Convert detections to array
        if len(detections) > 0:
            det_boxes = np.array([d["bbox"] for d in detections])
            det_classes = np.array([d["class"] for d in detections])
            det_confs = np.array([d["confidence"] for d in detections])
        else:
            det_boxes = np.array([]).reshape(0, 4)
            det_classes = np.array([])
            det_confs = np.array([])

        # Get predicted tracks
        active_tracks = []
        for track in self.tracks:
            if track.consecutive_absence < self.max_age:
                active_tracks.append(track)

        if len(active_tracks) == 0:
            # No active tracks, create new ones
            self.tracks = []
            for i, det in enumerate(detections):
                track = Track(
                    self.next_id, det["bbox"],
                    cls=det.get("class", "person"),
                    confidence=det.get("confidence", 1.0)
                )
                self.tracks.append(track)
                self.next_id += 1

            return [
                {
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "track_id": track.track_id
                }
                for track, det in zip(self.tracks, detections)
            ]

        # Predict track positions
        pred_boxes = []
        valid_tracks = []
        for track in active_tracks:
            pred_box = track.predict()
            if pred_box is not None:
                pred_boxes.append(pred_box)
                valid_tracks.append(track)

        if len(pred_boxes) == 0 or len(det_boxes) == 0:
            # No overlap, mark all tracks as missed
            for track in valid_tracks:
                track.mark_missed()
            # Create new tracks for detections
            for det in detections:
                track = Track(
                    self.next_id, det["bbox"],
                    cls=det.get("class", "person"),
                    confidence=det.get("confidence", 1.0)
                )
                self.tracks.append(track)
                self.next_id += 1
            return self._get_results(detections, valid_tracks)


        pred_boxes = np.array(pred_boxes)

        # Compute cost matrix (IoU)
        cost_matrix = self._compute_iou_cost(pred_boxes, det_boxes)

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Update matched tracks
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.iou_threshold:
                continue
            det = detections[c]
            valid_tracks[r].update(
                det_boxes[c],
                cls=det.get("class"),
                confidence=det.get("confidence")
            )
            matched_tracks.add(r)
            matched_dets.add(c)

        # Mark unmatched tracks as missed
        for i, track in enumerate(valid_tracks):
            if i not in matched_tracks:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                track = Track(
                    self.next_id, det["bbox"],
                    cls=det.get("class", "person"),
                    confidence=det.get("confidence", 1.0)
                )
                self.tracks.append(track)
                self.next_id += 1

        return self._get_results(detections, valid_tracks)

    def _compute_iou_cost(self, pred_boxes: np.ndarray,
                          det_boxes: np.ndarray) -> np.ndarray:
        """Compute IoU cost matrix."""
        n_pred = len(pred_boxes)
        n_det = len(det_boxes)
        cost = np.zeros((n_pred, n_det))

        for i, pred in enumerate(pred_boxes):
            for j, det in enumerate(det_boxes):
                cost[i, j] = 1.0 - self._iou(pred, det)

        return cost

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        union = area_a + area_b - inter
        if union == 0:
            return 0

        return inter / union

    def _get_results(self, detections: List[Dict],
                     tracks: List[Track]) -> List[Dict[str, Any]]:
        """Get tracking results for active tracks in current frame."""
        results = []
        seen_ids = set()

        # Confirmed tracks seen in current frame
        for track in tracks:
            if track.age >= self.min_hits and track.consecutive_absence == 0:
                bbox = track.get_state()
                if bbox is not None:
                    results.append({
                        "bbox": bbox,
                        "class": track.cls,
                        "confidence": track.confidence,
                        "track_id": track.track_id,
                        "history": list(track.history)
                    })
                    seen_ids.add(track.track_id)

        # Also include new (tentative) detections not yet confirmed
        for track in self.tracks:
            if track.track_id not in seen_ids and track.consecutive_absence == 0:
                bbox = track.get_state()
                if bbox is not None:
                    results.append({
                        "bbox": bbox,
                        "class": track.cls,
                        "confidence": track.confidence,
                        "track_id": track.track_id,
                        "history": list(track.history)
                    })
                    seen_ids.add(track.track_id)

        return results


    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks."""
        return [
            {
                "track_id": t.track_id,
                "age": t.age,
                "history": list(t.history)
            }
            for t in self.tracks
            if t.consecutive_absence < self.max_age
        ]

    def cleanup(self):
        """Remove old tracks."""
        self.tracks = [
            t for t in self.tracks
            if t.consecutive_absence < self.max_age
        ]
