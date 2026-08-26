from pathlib import Path

import cv2
import numpy as np


class AnnotatedVideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int):
        self._path = Path(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for: {self._path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
