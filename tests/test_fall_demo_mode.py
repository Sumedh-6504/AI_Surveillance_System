"""Regression checks for isolated exam demo and pose-debug output."""
import ast
from pathlib import Path

from src.detection.fall_pose import FallStateMachine, PoseFallDetector


def test_cli_offers_isolated_fall_mode():
    main_file = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(main_file.read_text(encoding="utf-8"))
    main_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.get_source_segment(main_file.read_text(encoding="utf-8"), main_function)
    assert '"--mode"' in source
    assert '"--debug-fall"' in source


def test_debug_no_person_and_no_pose_never_generates_fall(capsys):
    import numpy as np
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine()
    detector.device = "cpu"
    class EmptyModel:
        def predict(self, **kwargs):
            result = type("Result", (), {"boxes": None, "keypoints": None})()
            return [result]
    detector.model = EmptyModel()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(frame=frame, tracks=[], now=1.0, debug=True) == []
    assert "poses=0" in capsys.readouterr().out
