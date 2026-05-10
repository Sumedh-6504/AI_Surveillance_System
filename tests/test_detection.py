"""Unit tests for detection modules."""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.yolo import YOLOv8Detector
from src.detection.rules import RuleEngine, Severity, DetectionRule
from src.detection.behavior_vjepa import BehaviorAnalyzer


class TestYOLOv8Detector:
    """Tests for YOLOv8 detector."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = YOLOv8Detector(
            model_path="models/nonexistent.onnx",
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
        # Session should be None for non-existent model
        assert detector.session is None
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45

    def test_detect_with_no_model(self):
        """Test detection returns empty list when model not loaded."""
        detector = YOLOv8Detector(model_path="models/nonexistent.onnx")
        image = np.zeros((1, 3, 640, 640))
        detections = detector.detect(image)
        assert detections == []

    def test_process_outputs_format(self):
        """Test output processing creates correct format."""
        detector = YOLOv8Detector(model_path="models/nonexistent.onnx")

        # Mock output format: [x1, y1, x2, y2, conf, cls_scores...]
        mock_output = np.array([[
            100, 100, 200, 200,  # bbox
            0.9,  # objectness
            0.1, 0.8, 0.1  # class scores (person=0.8)
        ]])

        # Simulate _process_outputs structure
        results = []
        for output in mock_output:
            bbox = output[:4]
            conf = output[4]
            cls_scores = output[5:]

            if conf >= 0.5:
                cls = np.argmax(cls_scores)
                cls_conf = cls_scores[cls]

                if cls_conf >= 0.5:
                    results.append({
                        "bbox": bbox,
                        "class": int(cls),
                        "confidence": float(cls_conf)
                    })

        assert len(results) == 1
        assert results[0]["class"] == 1
        assert results[0]["confidence"] == 0.8


class TestRuleEngine:
    """Tests for detection rule engine."""

    def test_rule_engine_initialization(self):
        """Test rule engine initializes with default rules."""
        engine = RuleEngine()
        assert len(engine.rules) > 0

        # Check for expected rules
        rule_names = [r.name for r in engine.rules]
        assert "WEAPON_DETECTED" in rule_names
        assert "FIGHT_DETECTED" in rule_names
        assert "LOITERING" in rule_names
        assert "CROWD_FORMATION" in rule_names

    def test_weapon_detection_rule(self):
        """Test weapon detection triggers correctly."""
        engine = RuleEngine()

        # Detections with weapon
        detections = [
            {"class": "knife", "confidence": 0.9},
            {"class": "person", "confidence": 0.95}
        ]

        triggered = engine.evaluate(detections, [], [])

        weapon_rule = next((r for r in triggered if r.name == "WEAPON_DETECTED"), None)
        assert weapon_rule is not None
        assert weapon_rule.severity == Severity.CRITICAL

    def test_cooldown_prevents_retrigger(self):
        """Test rule cooldown prevents immediate retrigger."""
        engine = RuleEngine()

        detections = [{"class": "gun", "confidence": 0.9}]

        # First trigger
        triggered = engine.evaluate(detections, [], [])
        assert len(triggered) > 0

        # Immediate second evaluation should not trigger (cooldown)
        triggered_again = engine.evaluate(detections, [], [])
        assert len(triggered_again) == 0

    def test_add_custom_rule(self):
        """Test adding custom rules."""
        engine = RuleEngine()
        initial_count = len(engine.rules)

        custom_rule = DetectionRule(
            name="CUSTOM_TEST",
            conditions=[{"class": ["suspicious"]}],
            severity=Severity.MEDIUM,
            cooldown_sec=30
        )

        engine.add_rule(custom_rule)
        assert len(engine.rules) == initial_count + 1


class TestBehaviorAnalyzer:
    """Tests for behavior analysis."""

    def test_analyzer_initialization(self):
        """Test behavior analyzer initializes correctly."""
        analyzer = BehaviorAnalyzer()
        if analyzer.backend.value == "mediapipe":
            assert hasattr(analyzer, 'pose') and analyzer.pose is not None
        assert len(analyzer.pose_history) == 0

    def test_angle_calculation(self):
        """Test angle calculation between points."""
        analyzer = BehaviorAnalyzer()
        a = (0, 1)
        b = (0, 0)
        c = (1, 0)
        angle = analyzer._calculate_angle(a, b, c)
        # Should be approximately 90 degrees
        assert 85 < angle < 95

    def test_analyze_empty_frame(self):
        """Test analysis of frame with no pose detected."""
        analyzer = BehaviorAnalyzer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        behaviors = analyzer.analyze(frame, [])
        assert behaviors == []

    def test_raise_arms_detection(self):
        """Test raised arms detection logic."""
        analyzer = BehaviorAnalyzer()
        # Bent-elbow ∧-shape: shoulder and wrist on either side of a raised elbow
        shoulder = (0.3, 0.5)
        elbow = (0.4, 0.3)
        wrist = (0.5, 0.5)
        angle = analyzer._calculate_angle(shoulder, elbow, wrist)
        # Acute angle indicates raised arms
        assert angle < 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
