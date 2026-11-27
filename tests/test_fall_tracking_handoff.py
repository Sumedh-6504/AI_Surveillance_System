"""Integration-like regression: state survives tracker ID churn in a one-person clip."""
import numpy as np
from src.detection.fall_pose import PoseFallDetector, FallStateMachine, FallConfig


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data


class FakeResult:
    def __init__(self, bbox, points):
        self.boxes = type("Boxes", (), {"xyxy": FakeTensor([bbox])})()
        self.keypoints = type("Kps", (), {
            "xy": FakeTensor([points]), "conf": FakeTensor([[0.99] * 17])
        })()


class FakeModel:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
    def predict(self, **kwargs):
        return [next(self.predictions)]


def joints(horizontal=False):
    points = [[0.0, 0.0] for _ in range(17)]
    if horizontal:
        for i, val in [(5, (20, 66)), (6, (20, 70)),
                       (11, (80, 66)), (12, (80, 70))]:
            points[i] = val
    else:
        for i, val in [(5, (46, 10)), (6, (54, 10)),
                       (11, (46, 44)), (12, (54, 44))]:
            points[i] = val
    return points


def test_one_person_fall_survives_track_id_changes():
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    detector.device = "cpu"
    upright = [40, 0, 60, 78]
    lying = [10, 50, 90, 92]
    detector.model = FakeModel([
        FakeResult(upright, joints()),
        FakeResult(lying, joints(horizontal=True)),
        FakeResult(lying, joints(horizontal=True)),
    ])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    def track(i, bbox):
        return [{"track_id": i, "bbox": bbox, "class": "person",
                 "confidence": 0.92}]
    assert detector.detect(frame, track(1, upright), 0.0) == []
    assert detector.detect(frame, track(3, lying), 1.0) == []
    alerts = detector.detect(frame, track(5, lying), 2.0)
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 5
    assert alerts[0]["behavior"] == "suspected_fall"


def test_multi_person_video_does_not_transfer_history():
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine()
    detector.device = "cpu"
    box = [40, 0, 60, 78]
    detector.model = FakeModel([FakeResult(box, joints())])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame, [{"track_id": 1, "bbox": box,
                             "class": "person", "confidence": .9}], 0.0)
    assert "__single_visible_person__" in detector.state.states
    # No pose corresponding to the second track: state isn't moved to it.
    assert 2 not in detector.state.states



def test_pose_only_detection_continues_fall_when_object_track_disappears():
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    detector.device = "cpu"
    upright = [40, 0, 60, 78]
    lying = [10, 50, 90, 92]
    detector.model = FakeModel([
        FakeResult(upright, joints()),
        FakeResult(lying, joints(horizontal=True)),
        FakeResult(lying, joints(horizontal=True)),
    ])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    track = [{"track_id": 7, "bbox": upright, "class": "person", "confidence": 0.9}]
    assert detector.detect(frame, track, 0.0) == []
    assert detector.detect(frame, [], 1.0) == []
    alerts = detector.detect(frame, [], 2.0)
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == -1
    assert alerts[0]["behavior"] == "suspected_fall"


def test_pose_only_lying_at_start_does_not_trigger():
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine(FallConfig(confirmation_seconds=0.5))
    detector.device = "cpu"
    lying = [10, 50, 90, 92]
    detector.model = FakeModel([
        FakeResult(lying, joints(horizontal=True)),
        FakeResult(lying, joints(horizontal=True)),
    ])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(frame, [], 0.0) == []
    assert detector.detect(frame, [], 2.0) == []


def test_pose_missing_does_not_create_fall_alert():
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine(FallConfig(confirmation_seconds=0.5))
    detector.device = "cpu"
    class NoPoseModel:
        def predict(self, **kwargs):
            return [type("Result", (), {"boxes": None, "keypoints": None})()]
    detector.model = NoPoseModel()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(frame, [], 0.0) == []
