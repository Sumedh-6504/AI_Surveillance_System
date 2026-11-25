"""Tests for laptop camera/video input and ONNX provider selection."""
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.camera.capture import CameraCapture
from src.detection.yolo import YOLOv8Detector


def test_local_video_is_not_treated_as_a_reconnecting_camera(tmp_path, monkeypatch):
    path = tmp_path / "demo.mp4"
    path.write_bytes(b"placeholder")
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.side_effect = [
        (True, np.zeros((32, 32, 3), dtype=np.uint8)),
        (False, None),
    ]
    import src.camera.capture as capture_module
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", lambda source: fake_cap)
    camera = CameraCapture(str(path))
    assert camera.is_video_file
    assert camera.connect()
    assert fake_cap.read.call_count == 0  # No warmup frames are discarded
    assert camera.read() is not None
    assert camera.read() is None
    assert camera.end_of_stream
    assert fake_cap.read.call_count == 2
    camera.release()


def test_webcam_index_is_numeric(monkeypatch):
    import src.camera.capture as capture_module
    seen = []
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (True, np.zeros((16, 16, 3), dtype=np.uint8))
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", lambda src: (seen.append(src) or fake_cap))
    monkeypatch.setattr(capture_module.time, "sleep", lambda _: None)
    camera = CameraCapture("0")
    assert camera.connect()
    assert seen == [0]
    assert not camera.is_video_file
    camera.release()


def test_cpu_provider_forced(monkeypatch, tmp_path):
    import src.detection.yolo as yolo_module
    path = tmp_path / "model.onnx"
    path.write_bytes(b"placeholder")
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    seen = []
    monkeypatch.setattr(yolo_module.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(yolo_module.ort, "InferenceSession", lambda model, providers: (seen.append(providers) or session))
    detector = YOLOv8Detector(str(path), device="cpu")
    assert detector.active_provider == "CPUExecutionProvider"
    assert seen == [["CPUExecutionProvider"]]


def test_auto_cuda_falls_back_to_cpu(monkeypatch, tmp_path):
    import src.detection.yolo as yolo_module
    path = tmp_path / "model.onnx"
    path.write_bytes(b"placeholder")
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    seen = []
    def open_session(model, providers):
        seen.append(providers)
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA libraries missing")
        return session
    monkeypatch.setattr(yolo_module.ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(yolo_module.ort, "InferenceSession", open_session)
    detector = YOLOv8Detector(str(path))
    assert detector.active_provider == "CPUExecutionProvider"
    assert len(seen) == 2


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        YOLOv8Detector(model_path="missing.onnx", device="invalid")
