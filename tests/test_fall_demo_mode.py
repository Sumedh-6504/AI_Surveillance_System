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


def test_debug_no_active_tracks_never_generates_fall(capsys):
    detector = PoseFallDetector.__new__(PoseFallDetector)
    detector.state = FallStateMachine()
    assert detector.detect(frame=None, tracks=[], now=1.0, debug=True) == []
    assert "no active person tracks" in capsys.readouterr().out
