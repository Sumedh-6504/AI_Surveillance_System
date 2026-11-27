"""Pose-based *suspected* fall detection with temporal confirmation.

Uses an optional YOLOv8 pose model to identify shoulder/hip geometry. A fall
requires an earlier upright pose, a downward transition, and a sustained
horizontal posture. Not intended for medical or safety-critical monitoring.
"""
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class FallConfig:
    confirmation_seconds: float = 0.8
    cooldown_seconds: float = 20.0
    min_drop_ratio: float = 0.12
    max_missing_seconds: float = 5.0


def pose_state(keypoints, confidence, bbox, frame_height, min_confidence=0.45):
    """Return (posture, normalized hip height) or None if pose is unreliable.

    COCO indices: 5/6 shoulders, 11/12 hips. Both keypoints of a pair must
    be visible; otherwise do not guess.
    """
    required = (5, 6, 11, 12)
    if keypoints is None or confidence is None or len(keypoints) <= 12:
        return None
    if any(float(confidence[i]) < min_confidence for i in required):
        return None
    sx = (float(keypoints[5][0]) + float(keypoints[6][0])) / 2
    sy = (float(keypoints[5][1]) + float(keypoints[6][1])) / 2
    hx = (float(keypoints[11][0]) + float(keypoints[12][0])) / 2
    hy = (float(keypoints[11][1]) + float(keypoints[12][1])) / 2
    dx, dy = abs(hx - sx), abs(hy - sy)
    if dx + dy < 12 or frame_height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(0.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if dy >= dx * 1.15 and height > width * 0.9:
        posture = "upright"
    elif dx >= dy * 1.15 and width >= height * 0.9:
        posture = "horizontal"
    else:
        posture = "other"
    return posture, hy / frame_height


def bbox_iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, float(a[2])-float(a[0])) * max(0.0, float(a[3])-float(a[1]))
    bb = max(0.0, float(b[2])-float(b[0])) * max(0.0, float(b[3])-float(b[1]))
    union = aa + bb - intersection
    return intersection / union if union else 0.0


class FallStateMachine:
    """State machine; emit one event on an upright-to-lying transition."""
    def __init__(self, config=None):
        self.config = config or FallConfig()
        self.states = {}

    def update(self, track_id, observation, now):
        state = self.states.setdefault(track_id, {
            "baseline": None, "candidate_since": None,
            "armed": False, "alerted": False, "last_seen": now,
            "last_alert": float("-inf"),
        })
        state["last_seen"] = now
        if observation is None:
            state["candidate_since"] = None
            return False
        posture, hip_y = observation
        if posture == "upright":
            state["baseline"] = hip_y
            state["candidate_since"] = None
            state["armed"] = True
            state["alerted"] = False
            return False
        drop = hip_y - state["baseline"] if state["baseline"] is not None else 0
        if posture == "horizontal" and state["armed"] and (
            drop >= self.config.min_drop_ratio
        ):
            if state["candidate_since"] is None:
                state["candidate_since"] = now
            if (not state["alerted"]
                and now - state["candidate_since"] >= self.config.confirmation_seconds
                and now - state["last_alert"] >= self.config.cooldown_seconds):
                state["alerted"] = True
                state["last_alert"] = now
                return True
        else:
            state["candidate_since"] = None
        return False

    def cleanup(self, active_ids, now):
        active_ids = set(active_ids)
        for track_id, state in list(self.states.items()):
            if track_id not in active_ids and (
                now - state["last_seen"] > self.config.max_missing_seconds
            ):
                del self.states[track_id]


class PoseFallDetector:
    """Optional lightweight pose inference; no model download during startup."""
    def __init__(self, model_path, device="auto", config=None):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {path}. Download yolov8n-pose.pt "
                "before enabling fall detection."
            )
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        if device == "cpu":
            self.device = "cpu"
        elif device in ("auto", "cuda"):
            try:
                import torch
                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False
            self.device = "0" if cuda else "cpu"
            if device == "cuda" and not cuda:
                print("Pose model: CUDA unavailable, using CPU")
        else:
            raise ValueError("device must be auto, cpu or cuda")
        self.state = FallStateMachine(config)
        print(f"Pose-based fall detector enabled on {self.device}")

    def detect(self, frame, tracks, now=None, debug=False):
        """Return suspected-fall alerts, matching pose boxes to person tracks."""
        if now is None:
            now = time.monotonic()
        active = {t.get("track_id") for t in tracks if t.get("track_id") is not None}
        # Always run pose inference: regular object detection may temporarily
        # miss a person who is bending or lying down during a fall.
        prediction = self.model.predict(
            source=frame, device=self.device, imgsz=640,
            conf=0.4, classes=[0], verbose=False
        )[0]
        boxes = prediction.boxes
        keypoints = prediction.keypoints
        poses = []
        if boxes is not None and keypoints is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            xy = keypoints.xy.cpu().numpy()
            kpconf = keypoints.conf
            kpconf = kpconf.cpu().numpy() if kpconf is not None else None
            for i, box in enumerate(xyxy):
                if kpconf is None:
                    continue  # No confidence = no reliable pose assessment.
                poses.append((box, xy[i], kpconf[i]))
        emitted = []
        used = set()
        # Stable state for an unambiguous single-person video: tracker IDs
        # can change as the subject bends or falls. The state still expires
        # after a configurable gap with no detections.
        person_tracks = [t for t in tracks if t.get("class") == "person"
                         and t.get("track_id") is not None]
        single_person = len(person_tracks) <= 1 and len(poses) == 1
        active_state_ids = set()
        stable_key = "__single_visible_person__"
        if debug:
            print(f"[FALL DEBUG] time={now:.2f}s, active_tracks={len(active)}, poses={len(poses)}")
        # If the primary detector loses this single person, use the pose
        # model's own bounding box to continue the same posture history.
        # Never infer a unique person in a multi-person frame.
        if single_person and not person_tracks:
            bbox, points, confidence = poses[0]
            observation = pose_state(points, confidence, bbox, frame.shape[0])
            active_state_ids.add(stable_key)
            if self.state.update(stable_key, observation, now):
                emitted.append({
                    "track_id": -1,  # Explicit unknown tracker ID.
                    "bbox": [float(v) for v in bbox],
                    "confidence": float(min(confidence[i] for i in (5, 6, 11, 12))),
                    "behavior": "suspected_fall",
                })
            if debug:
                state = self.state.states.get(stable_key, {})
                print(f"[FALL DEBUG] pose-only posture={observation[0] if observation else 'unknown'} "
                      f"armed={state.get('armed')} candidate={state.get('candidate_since')} "
                      f"alerted={state.get('alerted')}")
        for track in tracks:
            if track.get("class") != "person" or track.get("track_id") is None:
                continue
            track_id = track["track_id"]
            state_id = stable_key if single_person else track_id
            active_state_ids.add(state_id)
            candidates = [(bbox_iou(track["bbox"], p[0]), i)
                          for i, p in enumerate(poses) if i not in used]
            if not candidates:
                if debug:
                    print(f"[FALL DEBUG] track={track_id}: no pose candidates")
                self.state.update(state_id, None, now)
                continue
            similarity, index = max(candidates)
            if similarity < 0.3:
                if debug:
                    print(f"[FALL DEBUG] track={track_id}: low box IoU={similarity:.2f}")
                self.state.update(state_id, None, now)
                continue
            used.add(index)
            bbox, points, confidence = poses[index]
            observation = pose_state(points, confidence, bbox, frame.shape[0])
            if self.state.update(state_id, observation, now):
                emitted.append({
                    "track_id": track_id,
                    "bbox": [float(v) for v in track["bbox"]],
                    "confidence": float(track.get("confidence", 0)),
                    "behavior": "suspected_fall",
                })
            if debug:
                state = self.state.states.get(state_id, {})
                baseline = state.get("baseline")
                drop = (observation[1] - baseline) if observation and baseline is not None else None
                print(f"[FALL DEBUG] track={track_id} IoU={similarity:.2f} "
                      f"posture={observation[0] if observation else 'unknown'} "
                      f"hip_drop={drop if drop is not None else 'n/a'} "
                      f"armed={state.get('armed')} "
                      f"candidate={state.get('candidate_since')} "
                      f"alerted={state.get('alerted')}")
        self.state.cleanup(active_state_ids, now)
        return emitted
