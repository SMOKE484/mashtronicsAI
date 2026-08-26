from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from driveway_guard.sources.base import FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, path: Path, frame_stride: int = 1):
        self._path = Path(path)
        self._frame_stride = max(1, frame_stride)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {self._path}")

        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if frame_idx % self._frame_stride == 0:
                timestamp_s = frame_idx / self._fps
                yield frame_idx, timestamp_s, frame
            frame_idx += 1

    def close(self) -> None:
        self._cap.release()
