# Laptop demonstration: camera, video and inference device

The `snehasich` branch can use the laptop's webcam (default) or an existing local video file through the **same detection, tracking, rules, database and dashboard pipeline**.

## Setup on Windows PowerShell

From the repository root, activate your virtual environment and install the dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CUDA acceleration on an RTX 3050, first verify that the NVIDIA driver is installed with `nvidia-smi`. ONNX Runtime only uses the NVIDIA GPU if a compatible `onnxruntime-gpu` installation and its CUDA/cuDNN runtime libraries are available. Follow the current ONNX Runtime GPU installation matrix for your CUDA setup rather than assuming a GPU alone is sufficient. If unavailable, `--device auto` continues on CPU.

## Run webcam or MP4

```powershell
# Built-in laptop webcam
python main.py --source 0 --device auto

# Explicit CPU mode (works without an NVIDIA GPU)
python main.py --source 0 --device cpu

# Offline demo video from the repository folder
python main.py --source "videos/demo.mp4" --device auto

# Full path to a video (quotes required when paths have spaces)
python main.py --source "C:\Users\Student\Videos\demo.mp4" --device cpu

# Start dashboard in another terminal
python -m streamlit run dashboard/app.py
```

Video files stop at end-of-file without an endless reconnect loop. Press **q** to close the display. The database is shared by both modes.

## Test

```powershell
python -m pytest tests/test_video_source.py -v
```

Look for `YOLO ONNX inference provider: CPUExecutionProvider` or `CUDAExecutionProvider` in the terminal; the name reports the provider selected by ONNX Runtime.

**Limitations:** This stage does not add fall detection, a media upload page, or automatic model optimization. A single prerecorded video should first be checked for correct FPS, behavior thresholds and alert quality. Avoid presenting unvalidated fighting/weapon detections as tested functionality.
