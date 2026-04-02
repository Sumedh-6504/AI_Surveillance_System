# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Suspicious Activity Detection System - Real-time surveillance using YOLOv8 (ONNX), DeepSORT, and VL-JEPA2 for detecting suspicious activities like fighting, running, loitering, and crowd formation.

## Common Commands

```bash
# Run surveillance system
python main.py --config configs/default.yaml

# Run in demo mode (no camera)
python main.py --demo

# Run monitoring dashboard
streamlit run dashboard/app.py

# Run team progress dashboard
streamlit run team_dashboard.py

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Architecture

The system follows a pipeline architecture:

```
Camera → Frame Processor → YOLOv8 Detector → DeepSORT Tracker → Behavior Analyzer → Rule Engine → Alert Engine
         ↓                                              ↓
      Database ←─────────────────────────────────────────┘
```

**Core Modules:**
- `src/camera/` - Camera capture (RTSP/USB) with auto-reconnect
- `src/detection/` - YOLOv8 ONNX detection + VL-JEPA2 behavior analysis
- `src/tracking/` - DeepSORT multi-object tracking + trajectory analysis
- `src/alerting/` - Alert engine with rate limiting + email/SMS/webhook channels
- `src/database/` - SQLite persistence for events

**Entry Points:**
- `main.py` - Main surveillance system (SurveillanceSystem class)
- `dashboard/app.py` - Streamlit monitoring dashboard
- `team_dashboard.py` - Team progress tracking

## Configuration

Edit `configs/default.yaml` to configure:
- Camera source (`rtsp_url: "0"` for webcam, RTSP URL for IP camera)
- Detection model and thresholds
- Behavior analysis backend (VL-JEPA2 with INT8 quantization for 16GB RAM)
- Alert channels (email, SMS, webhook)
- Alert rules (loitering, crowd, weapon, fight detection)

## Key Implementation Notes

- VL-JEPA2 is the primary behavior analyzer backend (facebook/vjepa2-vitl-fpc64-256, 0.7B params)
- INT8 quantization is required for 16GB RAM systems (`use_quantization: true`)
- Frame skipping is used for performance optimization (`frame_skip` in config)
- The system uses SQLite for event storage in `data/surveillance.db`