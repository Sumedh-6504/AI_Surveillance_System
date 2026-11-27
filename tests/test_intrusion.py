"""Zone geometry and intrusion entry regression tests."""
import pytest
from src.detection.intrusion import RestrictedZone, IntrusionMonitor


def track(track_id=1, bbox=(40, 20, 60, 80), cls="person"):
    return {"track_id": track_id, "bbox": bbox, "class": cls, "confidence": 0.91}


def test_zone_validates_coordinates():
    with pytest.raises(ValueError):
        RestrictedZone(x1=0.9, x2=0.1)


def test_bottom_center_zone_entry_and_no_repeat():
    zone = RestrictedZone(0.4, 0.4, 0.7, 0.9)
    monitor = IntrusionMonitor(zone, cooldown_sec=10)
    shape = (100, 100, 3)
    person = track()
    assert monitor.update([person], shape, 100) == [person]
    assert monitor.update([person], shape, 101) == []
    assert monitor.update([], shape, 102) == []
    assert monitor.update([person], shape, 103) == []
    assert monitor.update([], shape, 110) == []
    assert monitor.update([person], shape, 111) == [person]


def test_outside_person_and_nonperson_are_ignored():
    zone = RestrictedZone(0.4, 0.4, 0.7, 0.9)
    monitor = IntrusionMonitor(zone)
    outside = track(bbox=(1, 1, 20, 30))
    car = track(2, cls="car")
    assert monitor.update([outside, car], (100, 100, 3), 50) == []


def test_zone_scales_with_frame_size():
    zone = RestrictedZone(0.25, 0.25, 0.75, 0.75)
    assert zone.pixels((200, 400, 3)) == (100, 50, 300, 150)
