"""FedTrap camera abstraction — desktop and Raspberry Pi sources.

Provides:
  ImageFileSource   — single image or directory of images
  VideoFileSource   — video file via OpenCV
  OpenCVCamera      — USB webcam via OpenCV
  PiCameraSource    — Raspberry Pi Camera Module 3 via Picamera2
  MotionDetector    — frame-differencing motion detection
  create_camera()   — factory that picks the right source
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Abstract base
# =====================================================================

class CameraSource:
    """Abstract camera interface."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError


# =====================================================================
# Desktop sources
# =====================================================================

class ImageFileSource(CameraSource):
    """Reads images from a file path or all images in a directory."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, path_or_dir: str):
        self._paths: List[str] = []
        if os.path.isfile(path_or_dir):
            self._paths = [path_or_dir]
        elif os.path.isdir(path_or_dir):
            self._paths = sorted(
                os.path.join(path_or_dir, f)
                for f in os.listdir(path_or_dir)
                if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            )
        self._idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._idx < len(self._paths):
            frame = cv2.imread(self._paths[self._idx])
            self._idx += 1
            if frame is not None:
                return True, frame
        return False, None

    def release(self) -> None:
        pass

    def is_opened(self) -> bool:
        return self._idx < len(self._paths)


class VideoFileSource(CameraSource):
    """Reads frames from a video file."""

    def __init__(self, video_path: str):
        self._cap = cv2.VideoCapture(video_path)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()


class OpenCVCamera(CameraSource):
    """Live USB webcam via OpenCV."""

    def __init__(self, device_id: int = 0):
        self._cap = cv2.VideoCapture(device_id)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()


# =====================================================================
# Raspberry Pi Camera Module 3 via Picamera2
# =====================================================================

class PiCameraSource(CameraSource):
    """Pi Camera Module 3 using Picamera2 (the modern API)."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        self._opened = False
        try:
            from picamera2 import Picamera2  # type: ignore

            self._cam = Picamera2()
            config = self._cam.create_still_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            self._cam.configure(config)
            self._cam.start()
            self._opened = True
            logger.info("PiCamera2 started (%s @ %d fps)", resolution, fps)
        except ImportError:
            logger.warning("picamera2 not available — PiCameraSource disabled")
        except Exception as e:
            logger.error("Failed to start Picamera2: %s", e)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        try:
            frame = self._cam.capture_array()
            # Picamera2 returns RGB; OpenCV expects BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception:
            return False, None

    def release(self) -> None:
        if self._opened:
            self._cam.stop()
            self._cam.close()
            self._opened = False

    def is_opened(self) -> bool:
        return self._opened


# =====================================================================
# Motion detection
# =====================================================================

class MotionDetector:
    """Simple frame-differencing motion detector."""

    def __init__(self, threshold: int = 25, min_area: int = 500):
        self.threshold = threshold
        self.min_area = min_area
        self._bg_frame: Optional[np.ndarray] = None

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[bool, np.ndarray, List]:
        """Returns (motion_detected, thresh_mask, valid_contours)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._bg_frame is None:
            self._bg_frame = gray
            return False, np.zeros_like(gray), []

        delta = cv2.absdiff(self._bg_frame, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        return len(valid) > 0, thresh, valid

    def update_background(self, frame: np.ndarray) -> None:
        """Optionally update the background model."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._bg_frame = cv2.GaussianBlur(gray, (21, 21), 0)


# =====================================================================
# Factory
# =====================================================================

def create_camera(source: str, config: Optional[Dict] = None) -> CameraSource:
    """Create the appropriate camera source from a source descriptor."""
    if source == "picamera":
        return PiCameraSource()
    elif source in ("webcam", "0"):
        return OpenCVCamera(0)
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in (".mp4", ".avi", ".mkv", ".mov", ".wmv"):
            return VideoFileSource(source)
        return ImageFileSource(source)
    elif os.path.isdir(source):
        return ImageFileSource(source)
    else:
        logger.warning("Unknown source '%s' — treating as image path", source)
        return ImageFileSource(source)
