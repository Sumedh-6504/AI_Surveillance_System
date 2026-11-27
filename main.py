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
from src.detection.event_policy import detection_signature, should_save_detection
from src.detection.intrusion import RestrictedZone, IntrusionMonitor
from src.detection.fall_pose import PoseFallDetector, FallConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "surveillance.db"


def resolve_db_path(db_path: str) -> str:
    """Resolve relative SQLite paths against the repository root."""
    path = Path(db_path).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


class SurveillanceSystem:
    """
    Main surveillance system orchestrator.
    Coordinates all subsystems: capture, detection, tracking, alerting.
    """

    def __init__(self, config_path: str = "configs/default.yaml",
                 db_path: str = str(DEFAULT_DB_PATH),
                 source: str = None, device: str = "auto",
                 mode: str = "all", debug_fall: bool = False):
        # Load configuration
        print(f"Loading config from {config_path}...")
        self.config = load_config(config_path)
        if mode not in ("all", "fall", "intrusion"):
            raise ValueError("mode must be all, fall, or intrusion")
        self.mode = mode
        self.debug_fall = debug_fall
        if mode == "fall":
            self.config.fall_detection.enabled = True
            self.config.intrusion.enabled = False
        elif mode == "intrusion":
            self.config.fall_detection.enabled = False
            self.config.intrusion.enabled = True
        print(f"Surveillance mode: {self.mode}; fall enabled={self.config.fall_detection.enabled}; intrusion enabled={self.config.intrusion.enabled}")
        self.db_path = resolve_db_path(db_path)
        self.source = source if source is not None else self.config.camera.rtsp_url
        if not self.source.isdigit() and not self.source.startswith(("rtsp://", "http://", "https://")):
            source_path = Path(self.source).expanduser()
            self.source = str((source_path if source_path.is_absolute() else PROJECT_ROOT / source_path).resolve())

        # Initialize database
        init_db(self.db_path)

        # Initialize camera
        camera_cfg = self.config.camera
        self.camera = CameraCapture(self.source, camera_cfg.fps)
        self.frame_processor = FrameProcessor(
            input_size=(640, 640)
        )

        # Initialize detection
        det_cfg = self.config.detection
        self.detector = YOLOv8Detector(
            model_path=det_cfg.yolo_model,
            confidence_threshold=det_cfg.confidence_threshold,
            iou_threshold=det_cfg.iou_threshold,
            device=device
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
        intrusion_cfg = self.config.intrusion
        self.intrusion = (
            IntrusionMonitor(RestrictedZone(*intrusion_cfg.zone),
                             cooldown_sec=intrusion_cfg.cooldown_seconds)
            if intrusion_cfg.enabled else None
        )

        # Optional pose model: separate from YOLO object detection.
        self.fall_detector = None
        fall_cfg = self.config.fall_detection
        if fall_cfg.enabled:
            pose_path = Path(fall_cfg.model_path).expanduser()
            if not pose_path.is_absolute():
                pose_path = PROJECT_ROOT / pose_path
            try:
                self.fall_detector = PoseFallDetector(
                    str(pose_path), device=device,
                    config=FallConfig(
                        confirmation_seconds=fall_cfg.confirmation_seconds,
                        cooldown_seconds=fall_cfg.cooldown_seconds,
                        min_drop_ratio=fall_cfg.min_drop_ratio,
                    ),
                )
            except (FileNotFoundError, ImportError) as exc:
                print(f"Fall detection disabled: {exc}")

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
        self.last_detection_save = 0.0
        self.last_detection_signature = None

    def start(self):
        """Start the surveillance system."""
        print("Starting surveillance system...")
        self.running = True

        # Connect to camera
        if not self.camera.connect():
            print("Failed to connect to camera!")
            return

        print(f"Connected to video source: {self.source}")
        if self.mode == "fall" and self.fall_detector is None:
            print("ERROR: Fall mode requested but pose model is not loaded; aborting.")
            return
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
        # A recorded fall can happen within skipped frames. Process every
        # frame in fall-only video mode; this favors accuracy over speed.
        frame_skip = (0 if self.mode == "fall" and self.camera.is_video_file
                      else opt_cfg.frame_skip)

        print("Starting processing pipeline...")
        print(f"Frame skip: {frame_skip}, Press 'q' to quit")

        while self.running:
            frame_time = time.time()

            # Read frame
            frame = self.camera.read()
            if frame is None:
                if self.camera.end_of_stream:
                    print("\nVideo file complete.")
                    break
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

        # Map class IDs to names (supporting standard COCO YOLOv8 class indices)
        coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
            24: "backpack", 26: "handbag", 28: "suitcase", 43: "knife", 76: "scissors"
        }
        class_names = ["person", "vehicle", "bag", "knife", "gun"]
        for det in detections:
            cls_val = det.get("class", 0)
            if isinstance(cls_val, int):
                if cls_val in coco_names:
                    det["class"] = coco_names[cls_val]
                elif cls_val < len(class_names):
                    det["class"] = class_names[cls_val]
                else:
                    det["class"] = "unknown"

        results["detections"] = detections

        # Store a snapshot when the scene changes, with a 60-second heartbeat
        # for unchanged scenes. Alerts are still saved independently.
        now = time.time()
        if should_save_detection(
            detections, self.last_detection_signature,
            self.last_detection_save, now,
        ):
            event_id = save_event(
                db_path=self.db_path,
                event_type="detection",
                severity="low",
                camera_id="cam1",
                frame_id=self.frame_count,
                rule_name="OBJECT_DETECTED",
                detections=detections,
            )
            if event_id > 0:
                self.last_detection_save = now
                self.last_detection_signature = detection_signature(detections)
                print(f"\nDetection saved: event_id={event_id}")
            else:
                print("\nERROR: Failed to save detection event")

        # Run tracking
        tracks = self.tracker.update(detections)
        results["tracks"] = tracks

        # Detect new entries to the configured restricted zone.
        # Only tracked people are eligible, not arbitrary objects.
        if self.intrusion is not None:
            entries = self.intrusion.update(tracks, frame.shape, time.time())
            results["intrusions"] = entries
            for entered in entries:
                self._record_intrusion(frame, entered)
                results["alerts"].append({
                    "rule": "RESTRICTED_ZONE_INTRUSION", "severity": "high",
                })

        # Evaluate pose transitions: upright -> downward motion -> horizontal.
        # A single lying/sitting pose must not create a fall alert.
        if self.fall_detector is not None:
            try:
                fall_time = time.monotonic()
                if self.camera.is_video_file and self.camera.cap is not None:
                    cap = self.camera.cap
                    media_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if media_ms > 0:
                        fall_time = media_ms / 1000.0
                    elif fps > 0:
                        fall_time = frame_number / fps
                falls = self.fall_detector.detect(
                    frame, tracks, fall_time,
                    debug=self.debug_fall and self.frame_count % 15 == 0,
                )
                for fall in falls:
                    self._record_fall(frame, fall)
                    results["alerts"].append({
                        "rule": "SUSPECTED_FALL", "severity": "high",
                    })
            except Exception as exc:
                print(f"Fall model inference failed; disabling pose analysis: {exc}")
                self.fall_detector = None

        # Analyze behavior
        behaviors = self.behavior_analyzer.analyze(frame, detections, tracks)
        results["behaviors"] = behaviors

        # Analyze trajectories
        frame_time = time.time()
        active_track_ids = {t.get("track_id") for t in tracks}
        trajectory_alerts = self.trajectory_analyzer.update(tracks, frame_time)
        self.trajectory_analyzer.cleanup(active_track_ids)
        results["alerts"].extend(trajectory_alerts)

        # Evaluate rules (include both behavior and trajectory detections)
        all_behaviors = behaviors + trajectory_alerts
        triggered_rules = self.rule_engine.evaluate(detections, all_behaviors, tracks)


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
                if event_id > 0:
                    print(f"Event saved: rule={rule.name}, event_id={event_id}")
                else:
                    print(f"ERROR: Failed to save alert event for rule={rule.name}")
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

    def _record_intrusion(self, frame, track):
        """Persist an intrusion event and its JPEG evidence to local storage."""
        from uuid import uuid4
        evidence_dir = Path(self.db_path).parent / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        filename = f"intrusion_{uuid4().hex}.jpg"
        image_path = evidence_dir / filename
        annotated = frame.copy()
        bbox = track.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, "RESTRICTED ZONE INTRUSION", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if self.intrusion is not None:
            zx1, zy1, zx2, zy2 = self.intrusion.zone.pixels(frame.shape)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
        evidence_name = None
        if cv2.imwrite(str(image_path), annotated):
            evidence_name = filename
        else:
            print("WARNING: Failed to save intrusion screenshot")
        event_id = save_event(
            db_path=self.db_path, event_type="alert", severity="high",
            camera_id="cam1", frame_id=self.frame_count,
            rule_name="RESTRICTED_ZONE_INTRUSION",
            detections=[{
                "class": "person", "confidence": float(track.get("confidence", 0)),
                "track_id": int(track["track_id"]),
                "bbox": [float(value) for value in bbox],
            }],
            metadata={"track_id": int(track["track_id"]),
                      "evidence_file": evidence_name},
        )
        if event_id > 0:
            print(f"\nINTRUSION saved: event_id={event_id}, evidence={evidence_name}")
        else:
            print("\nERROR: Intrusion event was not saved")
            if evidence_name:
                image_path.unlink(missing_ok=True)

    def _record_fall(self, frame, fall):
        """Save a high-severity suspected-fall event and its local evidence."""
        from uuid import uuid4
        evidence_dir = Path(self.db_path).parent / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        filename = f"fall_{uuid4().hex}.jpg"
        image_path = evidence_dir / filename
        annotated = frame.copy()
        bbox = fall["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, "SUSPECTED FALL - REVIEW REQUIRED",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        evidence_name = filename if cv2.imwrite(str(image_path), annotated) else None
        if evidence_name is None:
            print("WARNING: Fall evidence screenshot could not be saved")
        event_id = save_event(
            db_path=self.db_path, event_type="alert", severity="high",
            camera_id="cam1", frame_id=self.frame_count,
            rule_name="SUSPECTED_FALL",
            detections=[{
                "class": "person", "track_id": int(fall["track_id"]),
                "confidence": float(fall["confidence"]),
                "bbox": [float(v) for v in bbox],
            }],
            metadata={
                "track_id": int(fall["track_id"]),
                "evidence_file": evidence_name,
                "assessment": "Possible fall based on pose and temporal motion; unverified",
            },
        )
        if event_id > 0:
            print(f"\nSuspected fall saved: event_id={event_id}, evidence={evidence_name}")
        else:
            print("\nERROR: Failed to persist suspected-fall event")
            if evidence_name:
                image_path.unlink(missing_ok=True)

    def _display_frame(self, frame, results: dict):
        """Draw results on frame and display."""
        # Draw detections
        frame = self.frame_processor.draw_detections(frame, results.get("detections", []))
        if self.intrusion is not None:
            zx1, zy1, zx2, zy2 = self.intrusion.zone.pixels(frame.shape)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (zx1, max(20, zy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
        default=str(DEFAULT_DB_PATH),
        help="Path to database file"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Webcam index (0), RTSP URL, or path to an MP4 file"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX inference provider preference (default: auto)"
    )
    parser.add_argument(
        "--mode",
        choices=("all", "fall", "intrusion"),
        default="all",
        help="Isolate fall detection or intrusion detection for exam demos"
    )
    parser.add_argument(
        "--debug-fall",
        action="store_true",
        help="Print periodic pose association and fall-state diagnostics"
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
        db_path=args.db,
        source=args.source,
        device=args.device,
        mode=args.mode,
        debug_fall=args.debug_fall
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
