# Phase 3: Pose-based suspected-fall monitoring

This phase adds an **optional** pose model for suspected fall alerts. It analyzes a change from an earlier upright pose to a lower, sustained horizontal posture. The decision combines learned body keypoints from YOLOv8 pose with deterministic temporal rules; it is **not** a trained fall-classification model and is not reliable enough for emergency response without further validation.

## 1. Install the pose weights once (Internet needed for download)

In the repository root on Windows PowerShell:

```powershell
git switch snehasich
git pull origin snehasich
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
New-Item -ItemType Directory -Force -Path models
Move-Item .\yolov8n-pose.pt .\models\yolov8n-pose.pt
```

The model file must exist at `models/yolov8n-pose.pt`. The detection code does not download model files on startup, so after this initial setup, the program can run without Internet.

## 2. Enable fall detection

In `configs/default.yaml`:

```yaml
fall_detection:
  enabled: true
  model_path: "models/yolov8n-pose.pt"
  confirmation_seconds: 0.8
  cooldown_seconds: 20
  min_drop_ratio: 0.12
```

If the file is missing, the program prints a warning and continues with object detection/intrusion detection. Fall detection defaults to **disabled** so existing installations will not break.

## 3. Demonstrate using a SAFE recorded test clip

```powershell
python main.py --source "videos/fall_test.mp4" --device auto
```

Do not perform a real fall during examination. Use a permitted prerecorded video that clearly shows an upright person and a subsequent fall with visible shoulders and hips. The system requires a previously upright pose, hip displacement, and a horizontal posture that persists for at least 0.8 seconds. A person already lying down at the start is **not** classified as a fall.

In another terminal:

```powershell
python -m streamlit run dashboard/app.py
```

Choose **Alerts only** and then **Inspect Event Details**. Events are labeled `SUSPECTED_FALL` with **HIGH** severity. The system saves a screenshot in `data/evidence/`, displays it in the dashboard, and records the track ID and event metadata in SQLite.

## 4. CPU/GPU behavior

The pose model uses NVIDIA CUDA when PyTorch reports it available and you use `--device auto`; otherwise it runs on CPU. Inference can be substantially slower on a CPU. Your object detector uses ONNX Runtime and may use a different provider. The RTX 3050 4 GB should be tested for memory use with both models running simultaneously.

## 5. Test

```powershell
python -m pytest tests/test_fall_pose.py tests/test_intrusion.py tests/test_evidence.py -v
```

## Limitations and verification

- The pose model and temporal heuristic can miss falls or flag ordinary actions; the sensitivity thresholds are starting points, not validated values.
- It works best with **one fully visible person**. Keypoints from distant, partially hidden, or sideways people may be unreliable.
- Tracking ID changes and video FPS differences affect temporal confirmation.
- A recorded fall clip should be compared with several normal walking, sitting, bending, and lying-down clips. Record true positives, false positives, false negatives, and frame rate for your exam report.
- Falls are *suspected* incidents requiring human review. This project is not certified as a medical or emergency-monitoring device.
- Evidence screenshots stay on the laptop. Obtain consent from recorded participants and delete retained evidence after your demonstration.

