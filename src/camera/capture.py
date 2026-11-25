"""Camera capture module using OpenCV."""
import cv2
import threading
from typing import Optional, Tuple
import time
from pathlib import Path


class CameraCapture:
    """
    OpenCV-based camera capture with auto-reconnect.
    Supports RTSP, HTTP streams, and USB cameras.
    """

    def __init__(self, rtsp_url: str, fps: int = 15):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[cv2.Mat] = None
        self.lock = threading.Lock()
        self.running = False
        self.reconnect_delay = 5.0
        self.last_reconnect = 0.0
        self.is_video_file = self._is_file_source(rtsp_url)
        self.end_of_stream = False

    @staticmethod
    def _is_file_source(source: str) -> bool:
        """Treat an existing local file as finite video, not a reconnecting stream."""
        return Path(source).is_file()

    def connect(self) -> bool:
        """Connect to camera stream."""
        try:
            # Numeric webcam indices stay integers; MP4 paths stay strings.
            source = int(self.rtsp_url) if self.rtsp_url.isdigit() else self.rtsp_url
            self.end_of_stream = False
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                return False

            # Never consume the opening frames of a prerecorded test video.
            if not self.is_video_file:
                print("Warming up camera...")
                for _ in range(5):
                    self.cap.read()
                    time.sleep(0.1)

            self.running = True
            print("Camera connected and ready")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def read(self) -> Optional[cv2.Mat]:
        """Read a frame from the camera."""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            if self.is_video_file:
                self.end_of_stream = True
                self.running = False
            else:
                self._attempt_reconnect()
            return None

        with self.lock:
            self.frame = frame

        return frame

    def _attempt_reconnect(self):
        """Attempt to reconnect to camera."""
        now = time.time()
        if now - self.last_reconnect < self.reconnect_delay:
            return

        print("Attempting to reconnect to camera...")
        self.last_reconnect = now

        if self.cap:
            self.cap.release()

        self.connect()

    def release(self):
        """Release camera resources."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame_shape(self) -> Tuple[int, int, int]:
        """Get frame dimensions."""
        if self.frame is not None:
            return self.frame.shape
        return (720, 1280, 3)
