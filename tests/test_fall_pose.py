"""Tests for suspected fall state machine and posture confidence gates."""
from src.detection.fall_pose import FallConfig, FallStateMachine, pose_state


def test_upright_to_sustained_fall_alerts_once():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    assert not state.update(1, ("upright", 0.40), 0.0)
    assert not state.update(1, ("horizontal", 0.65), 0.1)
    assert not state.update(1, ("horizontal", 0.65), 0.7)
    assert state.update(1, ("horizontal", 0.65), 1.0)
    assert not state.update(1, ("horizontal", 0.65), 2.0)


def test_initial_lying_down_not_reported_as_fall():
    state = FallStateMachine()
    assert not state.update(1, ("horizontal", 0.75), 0)
    assert not state.update(1, ("horizontal", 0.75), 30)


def test_sitting_or_bending_interrupts_confirmation():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    assert not state.update(1, ("upright", 0.30), 0.0)
    assert not state.update(1, ("horizontal", 0.60), 0.1)
    assert not state.update(1, ("other", 0.50), 0.5)
    assert not state.update(1, ("horizontal", 0.60), 0.8)
    assert not state.update(1, ("horizontal", 0.60), 1.2)


def test_no_alert_without_vertical_drop():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    assert not state.update(2, ("upright", 0.50), 0.0)
    assert not state.update(2, ("horizontal", 0.54), 0.1)
    assert not state.update(2, ("horizontal", 0.54), 2.0)


def test_cooldown_and_rearming():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.5, cooldown_seconds=20))
    state.update(2, ("upright", 0.3), 1)
    state.update(2, ("horizontal", 0.6), 2)
    assert state.update(2, ("horizontal", 0.6), 3)
    state.update(2, ("upright", 0.3), 5)
    state.update(2, ("horizontal", 0.6), 6)
    assert not state.update(2, ("horizontal", 0.6), 7)
    state.update(2, ("upright", 0.3), 24)
    state.update(2, ("horizontal", 0.6), 25)
    assert state.update(2, ("horizontal", 0.6), 26)


def test_missing_confidence_does_not_infer_pose():
    points = [[0, 0]] * 17
    confidence = [0.0] * 17
    assert pose_state(points, confidence, [0, 0, 50, 100], 100) is None


def test_valid_vertical_posture():
    points = [[0, 0] for _ in range(17)]
    for index, point in [(5, (40, 20)), (6, (60, 20)), (11, (40, 70)), (12, (60, 70))]:
        points[index] = point
    confidence = [0.9] * 17
    assert pose_state(points, confidence, [20, 10, 80, 95], 100)[0] == "upright"


def test_valid_horizontal_posture():
    points = [[0, 0] for _ in range(17)]
    for index, point in [(5, (20, 60)), (6, (20, 64)), (11, (80, 60)), (12, (80, 64))]:
        points[index] = point
    confidence = [0.9] * 17
    assert pose_state(points, confidence, [10, 40, 100, 80], 100)[0] == "horizontal"



def test_noisy_horizontal_hip_position_preserves_fall_confirmation():
    """Regression for the recorded clip: 4.87s fall onset, 5.27s and 5.70s horizontal."""
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8, min_drop_ratio=0.12))
    key = "__single_visible_person__"
    assert not state.update(key, ("upright", 0.40), 3.97)
    assert not state.update(key, ("horizontal", 0.58), 4.87)
    # Pose remains horizontal but hips jump upward, which must not reset candidate.
    assert not state.update(key, ("horizontal", 0.46), 5.27)
    assert state.update(key, ("horizontal", 0.50), 5.70)
    assert not state.update(key, ("horizontal", 0.60), 6.20)


def test_noisy_pose_without_initial_drop_does_not_start_fall():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8, min_drop_ratio=0.12))
    assert not state.update(1, ("upright", 0.4), 1.0)
    assert not state.update(1, ("horizontal", 0.45), 1.3)
    assert not state.update(1, ("horizontal", 0.46), 2.3)
    assert not state.update(1, ("horizontal", 0.50), 3.3)


def test_unknown_pose_still_cancels_candidate():
    state = FallStateMachine(FallConfig(confirmation_seconds=0.8))
    state.update(1, ("upright", 0.3), 0)
    state.update(1, ("horizontal", 0.6), 1)
    state.update(1, None, 1.4)
    assert not state.update(1, ("horizontal", 0.6), 2.0)
    assert state.update(1, ("horizontal", 0.6), 3.0)
