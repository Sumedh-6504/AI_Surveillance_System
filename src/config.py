"""Configuration management using Pydantic."""
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CameraConfig(BaseModel):
    rtsp_url: str
    fps: int = 15
    resolution: List[int] = [1280, 720]


class DetectionConfig(BaseModel):
    yolo_model: str = "yolov8n.onnx"
    use_openvino: bool = False
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    classes: List[str] = ["person", "vehicle", "bag"]


class BehaviorConfig(BaseModel):
    backend: str = "mediapipe"  # "mediapipe", "vljepa2", or "hybrid"
    model_name: str = "facebook/vjepa2-vitl-fpc64-256"
    use_quantization: bool = True
    use_openvino: bool = False
    input_resolution: int = 224
    confidence_threshold: float = 0.6


class TrackingConfig(BaseModel):
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


class EmailConfig(BaseModel):
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender: str = "alerts@example.com"
    password: str = ""
    recipients: list = []


class SMSConfig(BaseModel):
    enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    from_number: str = ""
    to_numbers: list = []


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: str = ""


class AlertingConfig(BaseModel):
    email: EmailConfig
    sms: SMSConfig
    webhook: WebhookConfig = WebhookConfig()


class RuleConfig(BaseModel):
    enabled: bool = True
    threshold_seconds: int = 60


class RulesConfig(BaseModel):
    loitering: RuleConfig
    crowd: RuleConfig


class OptimizationConfig(BaseModel):
    max_cameras: int = 1
    frame_skip: int = 3
    batch_size: int = 1
    cpu_workers: int = 2
    gc_interval: int = 100


class Config(BaseModel):
    camera: CameraConfig
    detection: DetectionConfig
    behavior: BehaviorConfig = BehaviorConfig()
    tracking: TrackingConfig
    alerting: AlertingConfig
    rules: RulesConfig
    optimization: OptimizationConfig


def load_config(config_path: str = "configs/default.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return Config(**data)
