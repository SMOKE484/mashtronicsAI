import numpy as np


def crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float,
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, int, int]:
    """Crops `bbox` out of `frame` with `pad_ratio` extra margin on each
    side, clamped to frame bounds. Returns (crop, offset_x, offset_y) so
    coordinates found in the crop can be translated back to frame space."""
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(frame_w, int(x2 + pad_x))
    cy2 = min(frame_h, int(y2 + pad_y))
    return frame[cy1:cy2, cx1:cx2], cx1, cy1
