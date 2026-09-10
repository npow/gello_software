"""Minimal RGB camera driver for V4L2 USB cameras."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from gello.cameras.camera import CameraDriver


class V4L2Camera(CameraDriver):
    """Return the latest RGB frame from a V4L2 stream without blocking the control loop."""

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        read_wait_timeout_sec: float = 2.0,
        max_frame_age_sec: float = 1.5,
    ):
        self.device = device
        self._cap = cv2.VideoCapture()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            self._cap.open(device, cv2.CAP_V4L2)
            if self._cap.isOpened():
                break
            self._cap.release()
            time.sleep(0.2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open V4L2 camera {device}")

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, int(fps))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._capture_period_sec = 1.0 / max(1, int(fps))
        self._read_wait_timeout_sec = float(read_wait_timeout_sec)
        self._max_frame_age_sec = float(max_frame_age_sec)
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_frame_timestamp: Optional[float] = None
        self._last_capture_error: Optional[BaseException] = None

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"v4l2_capture_{device.rsplit('/', 1)[-1]}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Continuously drain device and publish newest RGB frame."""
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                with self._cap_lock:
                    ok, bgr = self._cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"No frame from V4L2 camera {self.device}")

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._latest_rgb = rgb
                    self._latest_depth = np.zeros((*rgb.shape[:2], 1), dtype=np.uint16)
                    self._latest_frame_timestamp = time.time()
                    self._last_capture_error = None
                    self._frame_ready.set()
            except Exception as exc:
                with self._frame_lock:
                    self._last_capture_error = exc
                self._stop_event.wait(0.05)
                continue

            remaining = self._capture_period_sec - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._frame_ready.wait(timeout=self._read_wait_timeout_sec):
            raise RuntimeError(f"Timed out waiting for V4L2 camera {self.device}")

        with self._frame_lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            timestamp = self._latest_frame_timestamp
            last_error = self._last_capture_error

        if rgb is None or depth is None or timestamp is None:
            if last_error is not None:
                raise RuntimeError(
                    f"V4L2 camera {self.device} capture error"
                ) from last_error
            raise RuntimeError(f"V4L2 camera {self.device} has no frame yet")

        if img_size is not None:
            rgb = cv2.resize(rgb, img_size)
            depth = cv2.resize(depth, img_size)
            if depth.ndim == 2:
                depth = depth[:, :, None]
        return rgb, depth

    def close(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        with self._cap_lock:
            self._cap.release()
