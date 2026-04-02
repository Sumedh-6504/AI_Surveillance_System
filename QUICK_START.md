# Quick Start Guide - AI Surveillance System

## Start the Camera and Detect Suspicious Activity

### Option 1: Run the Setup Script (Recommended for First Time)

Double-click `setup_and_run.bat` or run from terminal:
```bash
setup_and_run.bat
```

This will:
1. Create virtual environment (if needed)
2. Install all dependencies
3. Export YOLOv8 model to ONNX format
4. Start the surveillance system

---

### Option 2: Manual Setup (Step by Step)

#### Step 1: Open Terminal and Navigate to Project
```bash
cd D:\Temp\AI_Surveillance
```

#### Step 2: Create and Activate Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` prefix in your terminal.

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Export YOLOv8 Model to ONNX
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mkdir models
move *.onnx models\
```

#### Step 5: Create Data Directory
```bash
mkdir data
```

#### Step 6: Start the Surveillance System
```bash
python main.py --config configs\default.yaml
```

A window will open showing the camera feed with real-time detection. **Press 'q' to quit.**

---

### Option 3: Run Dashboard (Monitoring UI)

Open a **NEW terminal** and run:

```bash
cd D:\Temp\AI_Surveillance
.venv\Scripts\activate
streamlit run dashboard\app.py
```

This opens in your browser at `http://localhost:8501`

---

## Configuration

### Using USB Webcam (Default)
The default config uses your USB webcam. No changes needed.

### Using RTSP IP Camera
Edit `configs\default.yaml`:
```yaml
camera:
  rtsp_url: "rtsp://username:password@192.168.1.100:554/stream1"
```

### Using a Video File (Testing)
Edit `configs\default.yaml`:
```yaml
camera:
  rtsp_url: "videos/test_video.mp4"
```

---

## What Gets Detected

### Objects
- **Person** - Human detection
- **Vehicle** - Cars, trucks, motorcycles
- **Bag** - Backpacks, bags
- **Knife** - Bladed weapons
- **Gun** - Firearms

### Behaviors
- **Raised Arms** - Potential fighting stance
- **Fighting Stance** - Wide stance with raised arms
- **Running** - Fast movement detected
- **Loitering** - Person staying in one area >60 seconds
- **Crowd Formation** - 10+ people in frame
- **Weapon Present** - Knife or gun detected

### Alerts Triggered By
| Alert | Severity | Trigger |
|-------|----------|---------|
| WEAPON_DETECTED | Critical | Knife or gun in frame |
| FIGHT_DETECTED | High | Fighting stance or raised arms |
| LOITERING | Medium | Person in same area >60 sec |
| CROWD_FORMATION | Medium | 10+ people detected |

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### "Model not found: models/yolov8n.onnx"
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
mkdir models
move *.onnx models\
```

### Camera Not Opening
1. Check if camera is connected
2. Try closing other apps using the camera
3. For RTSP: Test URL in VLC Player first

### Out of Memory
Edit `configs\default.yaml`:
```yaml
optimization:
  frame_skip: 4      # Process every 5th frame
  max_cameras: 1
  cpu_workers: 1
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `python main.py` | Start surveillance with default config |
| `python main.py --demo` | Run demo mode (no camera) |
| `streamlit run dashboard\app.py` | Open monitoring dashboard |
| `pytest tests/ -v` | Run all tests |
| `pip list` | Show installed packages |

---

## Expected Performance (16GB RAM, Intel UHD)

| Metric | Value |
|--------|-------|
| FPS | 8-15 |
| Latency | <200ms |
| Memory | 4-8GB |
