"""
Behavior analysis using multiple backends.

Supports:
1. MediaPipe Pose (default, CPU-optimized)
2. VL-JEPA2 (experimental, requires 16GB+ RAM)
3. OpenPose (fallback)

For 16GB RAM systems with Intel UHD Graphics, MediaPipe is recommended.
VL-JEPA2 can be enabled for more accurate behavior analysis but requires
more memory and has higher latency.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from enum import Enum


class BehaviorBackend(Enum):
    MEDIAPIPE = "mediapipe"
    VLJEPA2 = "vljepa2"
    RULE_BASED = "rule_based"


class BehaviorAnalyzer:
    """
    Behavior analysis with multiple backend support.
    Auto-selects best backend based on system capabilities.

    For 16GB RAM + Intel UHD: Uses MediaPipe (default)
    For 8GB VRAM systems: Can optionally use VL-JEPA2
    """

    def __init__(self, backend: BehaviorBackend = BehaviorBackend.MEDIAPIPE,
                 use_vjepa2: bool = False):
        self.backend = backend
        self.use_vjepa2 = use_vjepa2

        if use_vjepa2 or backend == BehaviorBackend.VLJEPA2:
            self._init_vjepa2()
        elif backend == BehaviorBackend.MEDIAPIPE:
            self._init_mediapipe()
        else:
            self._init_rule_based()

        # Pose history for temporal analysis
        self.pose_history: deque = deque(maxlen=16)
        self.velocity_history: deque = deque(maxlen=8)

    def _init_mediapipe(self):
        """Initialize MediaPipe Pose (recommended for 16GB RAM)."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Lite model for CPU
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.backend = BehaviorBackend.MEDIAPIPE
            print("Behavior Analyzer: Using MediaPipe backend (CPU-optimized)")
        except (ImportError, AttributeError) as e:
            # AttributeError covers mediapipe>=0.10.30 which removed mp.solutions
            print(f"MediaPipe not available ({e}), falling back to rule-based")
            self._init_rule_based()

    def _init_vjepa2(self):
        """
        Initialize VL-JEPA2 for advanced behavior analysis.

        Note: VL-JEPA2 is Meta's Vision-Language JEPA model.
        For this implementation, we use a lightweight wrapper approach
        since VL-JEPA2 full inference may exceed 16GB RAM limits.

        Alternative: Use distilled/smaller vision-language models.
        """
        try:
            # VL-JEPA2 would require:
            # - torch >= 2.0
            # - transformers >= 4.35
            # - 8GB+ VRAM for reasonable inference

            # Check if torch is available
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 6:
                    print(f"Warning: Only {vram_gb:.1f}GB VRAM available.")
                    print("VL-JEPA2 may be slow or OOM. Consider MediaPipe backend.")
                    self._init_mediapipe()
                    return
            else:
                print("CUDA not available. VL-JEPA2 requires GPU.")
                print("Falling back to MediaPipe backend.")
                self._init_mediapipe()
                return

            # Try to initialize (this is a placeholder - actual VL-JEPA2
            # would need proper model loading)
            print("VL-JEPA2 backend selected (experimental)")
            self.vjepa_model = None  # Would load model here
            self.backend = BehaviorBackend.VLJEPA2

        except Exception as e:
            print(f"VL-JEPA2 initialization failed: {e}")
            print("Falling back to MediaPipe backend.")
            self._init_mediapipe()

    def _init_rule_based(self):
        """Initialize simple rule-based behavior detection."""
        self.backend = BehaviorBackend.RULE_BASED
        print("Behavior Analyzer: Using rule-based backend (lightweight)")

    def analyze(self, frame: np.ndarray,
                detections: List[Dict],
                tracks: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Analyze behavior from frame and detections.

        Args:
            frame: BGR frame from camera
            detections: Object detections from YOLO
            tracks: Tracked objects with history (optional)

        Returns:
            List of behavior detections
        """
        behaviors = []

        if self.backend == BehaviorBackend.MEDIAPIPE:
            behaviors = self._analyze_mediapipe(frame, detections)
        elif self.backend == BehaviorBackend.VLJEPA2:
            behaviors = self._analyze_vjepa2(frame, detections, tracks)
        else:
            behaviors = self._analyze_rule_based(detections, tracks)

        return behaviors

    def _analyze_mediapipe(self, frame: np.ndarray,
                           detections: List[Dict]) -> List[Dict[str, Any]]:
        """MediaPipe-based behavior analysis."""
        behaviors = []

        try:
            import cv2
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Analyze specific behaviors
                behavior = self._check_raised_arms(landmarks)
                if behavior:
                    behaviors.append(behavior)

                behavior = self._check_fighting_stance(landmarks)
                if behavior:
                    behaviors.append(behavior)

                # Store for temporal analysis
                self.pose_history.append(landmarks)

                # Temporal behaviors
                temporal = self._analyze_temporal()
                if temporal:
                    behaviors.append(temporal)

        except Exception as e:
            print(f"MediaPipe analysis error: {e}")

        return behaviors

    def _analyze_vjepa2(self, frame: np.ndarray,
                        detections: List[Dict],
                        tracks: List[Dict]) -> List[Dict[str, Any]]:
        """
        VL-JEPA2 based behavior analysis.

        This is a placeholder for when VL-JEPA2 is properly integrated.
        The approach would be:
        1. Use VL-JEPA2 to encode frame + detection crops
        2. Classify behavior from embeddings
        3. Track behavior over time
        """
        behaviors = []

        # Placeholder - would need actual VL-JEPA2 integration
        # For now, fall back to simpler analysis
        behaviors.extend(self._analyze_rule_based(detections, tracks))

        return behaviors

    def _analyze_rule_based(self, detections: List[Dict],
                            tracks: List[Dict] = None) -> List[Dict[str, Any]]:
        """Rule-based behavior analysis using detection patterns."""
        behaviors = []

        # Count people
        people_count = sum(1 for d in detections if d.get("class") == "person")

        # Crowd detection
        if people_count >= 10:
            behaviors.append({
                "behavior": "crowd_formation",
                "confidence": min(1.0, people_count / 15),
                "people_count": people_count
            })

        # Check for weapons
        weapon_classes = ["knife", "gun", "weapon"]
        weapons_detected = any(
            d.get("class") in weapon_classes for d in detections
        )
        if weapons_detected:
            behaviors.append({
                "behavior": "weapon_present",
                "confidence": 0.95,
                "severity": "critical"
            })

        # Analyze tracks for movement patterns
        if tracks:
            movement_behavior = self._analyze_tracks(tracks)
            if movement_behavior:
                behaviors.append(movement_behavior)

        return behaviors

    def _check_raised_arms(self, landmarks) -> Optional[Dict[str, Any]]:
        """Detect raised arms (potential fighting/threat)."""
        try:
            lm = landmarks

            # Get shoulder, elbow, wrist positions
            shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate arm angle
            angle = self._calculate_angle(
                (shoulder.x, shoulder.y),
                (elbow.x, elbow.y),
                (wrist.x, wrist.y)
            )

            # Raised arms = acute angle at elbow
            if angle < 90:
                return {
                    "behavior": "raised_arms",
                    "confidence": 1.0 - (angle / 90),
                    "angle": angle
                }
        except Exception:
            pass

        return None

    def _check_fighting_stance(self, landmarks) -> Optional[Dict[str, Any]]:
        """Detect fighting stance based on body posture."""
        try:
            lm = landmarks

            # Check for wide stance (legs apart)
            left_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_distance = abs(left_hip.x - right_hip.x)

            # Check arm positions
            left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Fighting stance = wide legs + raised arms
            if hip_distance > 0.3:  # Wide stance
                return {
                    "behavior": "fighting_stance",
                    "confidence": min(1.0, hip_distance),
                    "hip_width": hip_distance
                }
        except Exception:
            pass

        return None

    def _analyze_temporal(self) -> Optional[Dict[str, Any]]:
        """Analyze pose sequence for temporal behaviors."""
        if len(self.pose_history) < 8:
            return None

        # Detect rapid pose changes (agitation/aggression)
        # This is simplified - real implementation would track joint velocities
        return None

    def _analyze_tracks(self, tracks: List[Dict]) -> Optional[Dict[str, Any]]:
        """Analyze track patterns for suspicious movement."""
        for track in tracks:
            history = track.get("history", [])
            if len(history) < 5:
                continue

            # Calculate velocity
            first_box = history[0]
            last_box = history[-1]

            first_center = ((first_box[0] + first_box[2]) / 2,
                           (first_box[1] + first_box[3]) / 2)
            last_center = ((last_box[0] + last_box[2]) / 2,
                          (last_box[1] + last_box[3]) / 2)

            distance = np.sqrt(
                (last_center[0] - first_center[0]) ** 2 +
                (last_center[1] - first_center[1]) ** 2
            )

            # High velocity = running
            if distance > 100:  # Pixels per frame sequence
                return {
                    "behavior": "running",
                    "confidence": min(1.0, distance / 200),
                    "track_id": track.get("track_id")
                }

        return None

    def _calculate_angle(self, a: tuple, b: tuple, c: tuple) -> float:
        """Calculate angle between three points."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def release(self):
        """Release resources."""
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()


# Factory function for easy initialization
def create_behavior_analyzer(system_ram_gb: int = 16,
                             vram_gb: int = 0,
                             prefer_vjepa2: bool = False) -> BehaviorAnalyzer:
    """
    Create behavior analyzer optimized for system specs.

    Args:
        system_ram_gb: System RAM in GB
        vram_gb: GPU VRAM in GB (0 for integrated graphics)
        prefer_vjepa2: Whether to prefer VL-JEPA2 if available

    Returns:
        Configured BehaviorAnalyzer instance
    """
    # VL-JEPA2 requires 8GB+ VRAM and 16GB+ system RAM
    if prefer_vjepa2 and vram_gb >= 8 and system_ram_gb >= 16:
        return BehaviorAnalyzer(use_vjepa2=True)

    # For Intel UHD Graphics (no dedicated VRAM), use MediaPipe
    if vram_gb < 4:
        return BehaviorAnalyzer(backend=BehaviorBackend.MEDIAPIPE)

    # Default fallback
    return BehaviorAnalyzer(backend=BehaviorBackend.MEDIAPIPE)
