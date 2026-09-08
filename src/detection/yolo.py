"""YOLOv8 object detection using ONNX runtime."""
import onnxruntime as ort
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class YOLOv8Detector:
    """
    YOLOv8 detector using ONNX runtime for CPU inference.
    Optimized for Intel UHD graphics.
    """

    def __init__(self, model_path: str = "models/yolov8n.onnx",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self._load_model()

    def _load_model(self):
        """Load ONNX model."""
        if not Path(self.model_path).exists():
            print(f"Model not found: {self.model_path}")
            print("Please export YOLOv8 model to ONNX format first.")
            return

        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image (batched, CHW format)

        Returns:
            List of detections with bbox, class, confidence
        """
        if self.session is None:
            return []

        # Run inference
        outputs = self.session.run(None, {self.input_name: image})

        # Process outputs
        detections = self._process_outputs(outputs[0])

        return detections

    def _process_outputs(self, outputs: np.ndarray) -> List[Dict[str, Any]]:
        """Process YOLOv8 outputs to detection boxes."""
        results = []

        # YOLOv8 output shape: (1, 4+num_classes, num_detections)
        # Transpose to (num_detections, 4+num_classes)
        predictions = outputs[0].T

        for pred in predictions:
            # First 4 values: cx, cy, w, h (center format)
            cx, cy, w, h = pred[:4]
            cls_scores = pred[4:]

            cls = np.argmax(cls_scores)
            conf = float(cls_scores[cls])

            if conf < self.confidence_threshold:
                continue

            # Convert center format to corner format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            results.append({
                "bbox": [x1, y1, x2, y2],
                "class": int(cls),
                "confidence": conf
            })

        # Apply NMS
        if results:
            results = self._nms(results)

        return results

    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Non-Maximum Suppression."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        # Simple NMS implementation
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.iou_threshold]

        return [detections[i] for i in keep]
