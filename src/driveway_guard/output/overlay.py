import cv2
import numpy as np

from driveway_guard.detection.types import ObjectClass, TrackedObject

_COLORS = {
    ObjectClass.PERSON: (0, 220, 0),
    ObjectClass.VEHICLE: (0, 140, 255),
}


def draw_tracks(frame: np.ndarray, tracked_objects: list[TrackedObject]) -> np.ndarray:
    for obj in tracked_objects:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox_xyxy)
        color = _COLORS[obj.cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.cls.value}#{obj.track_id} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_text_banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    for idx, line in enumerate(lines):
        y = 24 + idx * 22
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
        )
    return frame
