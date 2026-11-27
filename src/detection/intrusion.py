"""Normalized rectangular intrusion zone and per-person entry detection.

This is geometry applied to YOLO/tracker detections; it is not a trained
intrusion-classification model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RestrictedZone:
    x1: float = 0.35
    y1: float = 0.25
    x2: float = 0.75
    y2: float = 0.85

    def __post_init__(self):
        if not (0 <= self.x1 < self.x2 <= 1 and 0 <= self.y1 < self.y2 <= 1):
            raise ValueError("Zone must be ordered and inside normalized [0, 1] coordinates")

    def pixels(self, shape):
        height, width = shape[:2]
        return (round(self.x1 * width), round(self.y1 * height),
                round(self.x2 * width), round(self.y2 * height))

    def contains_person(self, bbox, shape):
        """Use person's bottom-center (approximate ground contact point)."""
        if bbox is None or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = self.pixels(shape)
        px = (float(bbox[0]) + float(bbox[2])) / 2
        py = float(bbox[3])
        return x1 <= px <= x2 and y1 <= py <= y2


class IntrusionMonitor:
    def __init__(self, zone, cooldown_sec=20.0):
        self.zone = zone
        self.cooldown_sec = cooldown_sec
        self.inside = set()
        self.last_alert = {}

    def update(self, tracks, frame_shape, now):
        """Return newly entered tracked people; never repeat while inside."""
        current = set()
        entered = []
        for track in tracks:
            if track.get("class") != "person":
                continue
            track_id = track.get("track_id")
            if track_id is None:
                continue
            if self.zone.contains_person(track.get("bbox"), frame_shape):
                current.add(track_id)
                if track_id not in self.inside and (
                    now - self.last_alert.get(track_id, float("-inf"))
                    >= self.cooldown_sec
                ):
                    entered.append(track)
                    self.last_alert[track_id] = now
        self.inside = current
        # Avoid indefinitely retaining cooldown state for lost trackers.
        if len(self.last_alert) > 1000:
            self.last_alert = {
                key: value for key, value in self.last_alert.items()
                if now - value < self.cooldown_sec * 2
            }
        return entered
