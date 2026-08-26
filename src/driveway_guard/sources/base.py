from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class FrameSource(ABC):
    """A sequence of timestamped frames.

    Implemented by both finite sources (a video file) and unbounded ones
    (a future live RTSP stream) so the pipeline never assumes a known
    clip length or derives timestamps from frame_idx/fps math.
    """

    @property
    @abstractmethod
    def frame_width(self) -> int: ...

    @property
    @abstractmethod
    def frame_height(self) -> int: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yields (frame_idx, timestamp_s, frame_bgr)."""
        ...

    def close(self) -> None:
        pass
