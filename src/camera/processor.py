"""Frame preprocessing for model inference."""
import cv2
import numpy as np
from typing import Tuple


class FrameProcessor:
    """
    Preprocess frames for YOLOv8 inference.
    Handles resizing, normalization, and color conversion.
    """

    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame for YOLOv8 inference.

        Args:
            frame: BGR image from camera

        Returns:
            Preprocessed RGB image normalized to [0, 1]
        """
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)

        # BGR to RGB conversion
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC to CHW format for model
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw detection boxes on frame.

        Args:
            frame: Original BGR frame
            detections: List of {bbox, class, confidence}

        Returns:
            Frame with drawn detections
        """
        for det in detections:
            bbox = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]

            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            color = (0, 255, 0) if cls == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cls}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame
