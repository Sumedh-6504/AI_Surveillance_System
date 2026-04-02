# AI Surveillance System - Step-by-Step Setup Guide

**Optimized for: 16GB RAM, 8GB VRAM, Intel UHD Graphics**

---

## Prerequisites

| Component | Required | Recommended |
|-----------|----------|-------------|
| OS | Windows 10/11 | Windows 11 |
| RAM | 16GB | 16GB+ |
| GPU | Intel UHD | 8GB VRAM (optional CUDA) |
| Python | 3.10+ | 3.11 |
| Storage | 10GB free | SSD recommended |

---

## Phase 1: Environment Setup (15 minutes)

### Step 1: Install Python 3.11

1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   # Should show: Python 3.11.x
   ```

### Step 2: Create Project Directory

```bash
cd D:\Temp\AI_Surveillance
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# You should see (.venv) prefix in terminal
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Phase 2: Install Dependencies (20 minutes)

### Step 1: Install Core Dependencies

```bash
# Install CPU-optimized packages first
pip install numpy opencv-python pyyaml pydantic scipy

# Install ONNX runtime (CPU)
pip install onnxruntime

# For Intel OpenVINO acceleration (optional, for Intel UHD)
pip install openvino onnxruntime-openvino

# Install YOLOv8
pip install ultralytics

# Install MediaPipe (for behavior analysis)
pip install mediapipe

# Install tracking
pip install deep-sort-realtime scikit-learn

# Install dashboard
pip install streamlit flask flask-socketio

# Install alerting
pip install twilio requests

# Install testing
pip install pytest pytest-cov
```

### Step 2: Verify Installations

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

---

## Phase 3: Download YOLOv8 Model (5 minutes)

### Step 1: Export YOLOv8 to ONNX Format

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

### Step 2: Create Models Directory

```bash
mkdir models
move *.onnx models\
```

### Step 3: Verify Model

```bash
dir models\*.onnx
# Should show: yolov8n.onnx
```

---

## Phase 4: Configure Camera (5 minutes)

### Option A: RTSP IP Camera

Edit `configs/default.yaml`:

```yaml
camera:
  rtsp_url: "rtsp://192.168.1.100:554/stream1"  # Your camera IP
  fps: 15
  resolution: [1280, 720]
```

### Option B: USB Webcam

Edit `configs/default.yaml`:

```yaml
camera:
  rtsp_url: "0"  # Use 0 for default webcam
  fps: 15
  resolution: [1280, 720]
```

### Option C: Video File (Testing)

Edit `configs/default.yaml`:

```yaml
camera:
  rtsp_url: "videos/test_video.mp4"  # Path to video file
  fps: 15
  resolution: [1280, 720]
```

---

## Phase 5: Initialize Database (2 minutes)

```bash
# Create data directory
mkdir data

# Initialize database (automatic on first run)
python main.py --config configs/default.yaml
# Press 'q' to quit after seeing database initialized
```

---

## Phase 6: Run the Surveillance System

### Option 1: Full Surveillance Mode

```bash
# From project root directory
python main.py --config configs/default.yaml --db data/surveillance.db
```

**Controls:**
- Press `q` to quit
- Window shows real-time detections with bounding boxes

### Option 2: Demo Mode (No Camera)

```bash
python main.py --demo
```

### Option 3: With Custom Config

```bash
python main.py --config configs/custom.yaml --db data/mydb.db
```

---

## Phase 7: Run Dashboards

### Dashboard 1: Monitoring Dashboard

Open a **new terminal**:

```bash
# Activate venv if not already
.venv\Scripts\activate

# Run monitoring dashboard
streamlit run dashboard\app.py
```

This opens in browser at `http://localhost:8501`

**Features:**
- Real-time event feed
- Severity breakdown charts
- Event filtering by type/camera
- Database cleanup tools

### Dashboard 2: Team Progress

Open another **new terminal**:

```bash
streamlit run team_dashboard.py
```

This opens at `http://localhost:8502`

**Features:**
- Individual task tracking
- Overall progress metrics
- Project timeline

---

## Phase 8: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
start htmlcov\index.html
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

### Issue: "ONNX Runtime Error: Could not find library"

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime
```

### Issue: "Camera not connecting"

1. Check RTSP URL format: `rtsp://username:password@IP:port/stream`
2. Test with VLC Player first
3. Try lower FPS (10 instead of 15)

### Issue: "Out of Memory"

For 16GB RAM systems, edit `configs/default.yaml`:

```yaml
optimization:
  max_cameras: 1
  frame_skip: 4      # Increase from 3
  batch_size: 1
  cpu_workers: 1     # Reduce from 2
```

### Issue: MediaPipe fails on Intel UHD

Use CPU-only mode in `src/detection/behavior.py`:

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lite model
    enable_segmentation=False,
    min_detection_confidence=0.5
)
```

---

## Performance Optimization for 16GB RAM

### 1. Enable OpenVINO (Intel UHD Graphics)

```bash
pip install openvino-dev
python -c "from openvino.runtime import Core; print('OpenVINO available')"
```

Update `configs/default.yaml`:

```yaml
detection:
  use_openvino: true
```

### 2. Reduce Model Size

Use YOLOv8 nano model (already default):

```yaml
detection:
  yolo_model: "models/yolov8n.onnx"  # Nano = fastest
```

### 3. Adjust Frame Processing

```yaml
optimization:
  frame_skip: 3      # Process every 4th frame
  max_cameras: 1     # Single camera
```

### 4. Memory Management

Add to `main.py` after processing:

```python
import gc
# In _run_pipeline loop:
if self.frame_count % 100 == 0:
    gc.collect()
```

---

## Expected Performance (16GB RAM, Intel UHD)

| Metric | Expected |
|--------|----------|
| FPS | 8-15 FPS |
| Latency | <200ms |
| Memory Usage | 4-8GB |
| CPU Usage | 40-70% |

---

## Quick Reference Commands

```bash
# Activate environment
.venv\Scripts\activate

# Run surveillance
python main.py

# Run monitoring dashboard
streamlit run dashboard\app.py

# Run tests
pytest tests/ -v

# Check dependencies
pip list

# Export YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

---

## Next Steps

1. Configure your camera RTSP URL
2. Test with demo mode first
3. Run monitoring dashboard in background
4. Set up email/SMS alerts in `configs/default.yaml`
5. Review team dashboard for task assignments
