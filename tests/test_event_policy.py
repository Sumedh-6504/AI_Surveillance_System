"""Verify scene-based detection event throttling."""
from src.detection.event_policy import detection_signature, should_save_detection


def test_identical_scene_is_throttled():
    detections = [{"class": "person"}]
    signature = detection_signature(detections)
    assert not should_save_detection(detections, signature, 100, 110)
    assert should_save_detection(detections, signature, 100, 160)


def test_changed_scene_is_recorded_after_short_cooldown():
    previous = detection_signature([{"class": "person"}])
    current = [{"class": "person"}, {"class": "person"}]
    assert not should_save_detection(current, previous, 100, 103)
    assert should_save_detection(current, previous, 100, 106)


def test_order_and_empty_scene():
    scene1 = [{"class": "person"}, {"class": "car"}]
    scene2 = [{"class": "car"}, {"class": "person"}]
    assert detection_signature(scene1) == detection_signature(scene2)
    assert not should_save_detection([], (), 0, 1000)
