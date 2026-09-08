"""Main entry point for the AI Surveillance System."""
import cv2
import time
import argparse
from pathlib import Path
from typing import Optional

from src.config import load_config
from src.camera import CameraCapture, FrameProcessor
from src.detection import YOLOv8Detector
from src.detection.behavior_vjepa import BehaviorAnalyzer,BehaviorBackend
from src.tracking.deepsort import DeepSORTTracker
from src.tracking.trajectory import TrajectoryAnalyzer
from src.detection.rules import RuleEngine
from src.alerting.engine import AlertEngine
from src.database.models import init_db, save_event, get_events


class SurveillanceSystem:
    """
    Main surveillance system orchestrator.
    Coordinates all subsystems: capture, detection, tracking, alerting.
    """

    def __init__(self, config_path: str = "configs/default.yaml",
                 db_path: str = "data/surveillance.db"):
        # Load configuration
        print(f"Loading config from {config_path}...")
        self.config = load_config(config_path)
        self.db_path = db_path

        # Initialize database
        init_db(db_path)

        # Initialize camera
        camera_cfg = self.config.camera
        self.camera = CameraCapture(camera_cfg.rtsp_url, camera_cfg.fps)
        self.frame_processor = FrameProcessor(
            input_size=(640, 640)
        )

        # Initialize detection
        det_cfg = self.config.detection
        self.detector = YOLOv8Detector(
            model_path=det_cfg.yolo_model,
            confidence_threshold=det_cfg.confidence_threshold,
            iou_threshold=det_cfg.iou_threshold
        )

        # Initialize behavior analyzer using hybrid backend
        # Hybrid = rule-based + VL-JEPA2 temporal clip support
        self.behavior_analyzer = BehaviorAnalyzer(backend=BehaviorBackend.HYBRID)

        # Initialize tracking
        track_cfg = self.config.tracking
        self.tracker = DeepSORTTracker(
            max_age=track_cfg.max_age,
            min_hits=track_cfg.min_hits,
            iou_threshold=track_cfg.iou_threshold
        )
        self.trajectory_analyzer = TrajectoryAnalyzer(
            frame_size=(camera_cfg.resolution[0], camera_cfg.resolution[1])
        )

        # Initialize rule engine
        self.rule_engine = RuleEngine()

        # Initialize alert engine
        alert_cfg = {
            "email": {
                "enabled": self.config.alerting.email.enabled,
                "smtp_server": self.config.alerting.email.smtp_server,
                "smtp_port": self.config.alerting.email.smtp_port,
                "sender": self.config.alerting.email.sender,
                "password": self.config.alerting.email.password,
                "recipients": self.config.alerting.email.recipients,
            },
            "sms": {
                "enabled": self.config.alerting.sms.enabled,
                "twilio_account_sid": self.config.alerting.sms.twilio_account_sid,
                "twilio_auth_token": self.config.alerting.sms.twilio_auth_token,
                "from_number": self.config.alerting.sms.from_number,
                "to_numbers": self.config.alerting.sms.to_numbers,
            },
            "webhook": {
                "enabled": self.config.alerting.webhook.enabled,
                "url": self.config.alerting.webhook.url,
            }
        }
        self.alert_engine = AlertEngine(alert_cfg)

        # State
        self.running = False
        self.frame_count = 0
        self.fps_history = []

    def start(self):
        """Start the surveillance system."""
        print("Starting surveillance system...")
        self.running = True

        # Connect to camera
        if not self.camera.connect():
            print("Failed to connect to camera!")
            return

        print(f"Connected to camera: {self.config.camera.rtsp_url}")
        self._run_pipeline()

    def stop(self):
        """Stop the surveillance system."""
        print("Stopping surveillance system...")
        self.running = False
        self.camera.release()
        self.behavior_analyzer.release()

    def _run_pipeline(self):
        """Main processing pipeline."""
        opt_cfg = self.config.optimization
        frame_skip = opt_cfg.frame_skip

        print("Starting processing pipeline...")
        print(f"Frame skip: {frame_skip}, Press 'q' to quit")

        while self.running:
            frame_time = time.time()

            # Read frame
            frame = self.camera.read()
            if frame is None:
                print("Warning: Failed to read frame")
                time.sleep(0.1)
                continue

            # Debug: check frame validity
            if frame.mean() < 10:
                print(f"Warning: Frame too dark (brightness={frame.mean():.1f})")

            self.frame_count += 1

            # Skip frames for performance
            if self.frame_count % (frame_skip + 1) != 0:
                continue

            # Process frame
            results = self._process_frame(frame)

            # Display results
            self._display_frame(frame, results)

            # Calculate FPS
            elapsed = time.time() - frame_time
            self.fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(self.fps_history) > 30:
                self.fps_history = self.fps_history[-30:]

            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            print(f"\rFPS: {avg_fps:.1f}, Detections: {len(results.get('detections', []))}", end="")

    def _process_frame(self, frame) -> dict:
        """Process a single frame through the pipeline."""
        results = {
            "detections": [],
            "tracks": [],
            "behaviors": [],
            "alerts": []
        }

        # Preprocess frame
        processed = self.frame_processor.process(frame)

        # Run detection
        detections = self.detector.detect(processed)

        # Scale coordinates from model input space (640x640) to original frame space
        detections = self.frame_processor.scale_coords(detections, frame.shape)

        # Map class IDs to names
        class_names = ["person", "vehicle", "bag", "knife", "gun"]
        for det in detections:
            cls_id = det.get("class", 0)
            det["class"] = class_names[cls_id] if cls_id < len(class_names) else "unknown"

        results["detections"] = detections

        # Run tracking
        tracks = self.tracker.update(detections)
        results["tracks"] = tracks

        # Analyze behavior
        behaviors = self.behavior_analyzer.analyze(frame, detections)
        results["behaviors"] = behaviors

        # Analyze trajectories
        frame_time = time.time()
        active_track_ids = {t.get("track_id") for t in tracks}
        trajectory_alerts = self.trajectory_analyzer.update(tracks, frame_time)
        self.trajectory_analyzer.cleanup(active_track_ids)
        results["alerts"].extend(trajectory_alerts)

        # Evaluate rules
        triggered_rules = self.rule_engine.evaluate(detections, behaviors, tracks)

        for rule in triggered_rules:
            # Create alert (may return None due to rate limiting)
            alert = self.alert_engine.add_alert(
                rule=rule,
                detections=detections,
                frame_id=self.frame_count
            )

            # Always save event to database, regardless of alert status
            # This ensures we have a record even when alerts are rate-limited
            try:
                # Normalize detections: ensure bbox and required fields exist
                normalized_detections = []
                for det in detections:
                    norm_det = {
                        "class": det.get("class", "unknown"),
                        "confidence": det.get("confidence", 0.0),
                        "track_id": det.get("track_id"),
                        "bbox": det.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    }
                    # Ensure bbox is valid (4 floats)
                    bbox = norm_det["bbox"]
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        norm_det["bbox"] = [0.0, 0.0, 0.0, 0.0]
                    else:
                        norm_det["bbox"] = [float(x) for x in bbox]
                    normalized_detections.append(norm_det)

                event_id = save_event(
                    db_path=self.db_path,
                    event_type="alert",
                    severity=rule.severity.value,
                    camera_id="cam1",
                    frame_id=self.frame_count,
                    rule_name=rule.name,
                    detections=normalized_detections
                )
                print(f"Event saved: rule={rule.name}, event_id={event_id}")
            except Exception as e:
                print(f"ERROR: Failed to save event to database: {e}")

            if alert:
                results["alerts"].append({
                    "rule": rule.name,
                    "severity": rule.severity.value
                })

        # Dispatch alerts
        self.alert_engine.dispatch()

        return results

    def _display_frame(self, frame, results: dict):
        """Draw results on frame and display."""
        # Draw detections
        frame = self.frame_processor.draw_detections(frame, results.get("detections", []))

        # Draw tracks
        for track in results.get("tracks", []):
            track_id = track.get("track_id")
            if track_id is not None:
                bbox = track.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw alerts
        alerts = results.get("alerts", [])
        if alerts:
            cv2.putText(frame, f"ALERTS: {len(alerts)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("AI Surveillance", frame)

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Surveillance System")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--db",
        default="data/surveillance.db",
        help="Path to database file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode without camera"
    )

    args = parser.parse_args()

    if args.demo:
        print("Running in demo mode...")
        run_demo()
        return

    system = SurveillanceSystem(
        config_path=args.config,
        db_path=args.db
    )

    try:
        system.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.stop()


def run_demo():
    """Run a simple demo without camera."""
    print("Demo mode: Initialize with test data")
    # This would use a video file or generate synthetic data
    # For now, just verify imports work
    from src.config import load_config
    from src.detection.rules import RuleEngine, Severity

    rule_engine = RuleEngine()
    print(f"Rule engine initialized with {len(rule_engine.rules)} rules")

    for rule in rule_engine.rules:
        print(f"  - {rule.name} ({rule.severity.value})")


if __name__ == "__main__":
    main()
