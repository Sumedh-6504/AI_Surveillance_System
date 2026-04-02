@echo off
echo ============================================
echo AI Surveillance System - Setup and Run
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet

REM Check if YOLO model exists
if not exist "models\yolov8n.onnx" (
    echo.
    echo Exporting YOLOv8 model to ONNX format...
    mkdir models 2>nul
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
    move *.onnx models\ 2>nul
    echo Model exported to models\yolov8n.onnx
)

REM Create data directory
if not exist "data" mkdir data

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To start the surveillance system:
echo   1. Activate venv: .venv\Scripts\activate
echo   2. Run: python main.py --config configs\default.yaml
echo.
echo To run the dashboard (in a new terminal):
echo   streamlit run dashboard\app.py
echo.

REM Ask user if they want to start now
set /p start_now="Do you want to start the surveillance system now? (Y/N): "
if /i "%start_now%"=="Y" (
    echo.
    echo Starting surveillance system...
    echo Press 'q' in the camera window to quit
    echo.
    python main.py --config configs\default.yaml
)
