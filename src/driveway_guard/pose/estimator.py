import math

import numpy as np
from ultralytics import YOLO

from driveway_guard.detection.types import TrackedObject
from driveway_guard.imaging import crop_with_padding

# COCO 17-keypoint skeleton edges, reused by the overlay renderer.
COCO_POSE_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]

# (x, y, confidence), in full-frame pixel coordinates.
Keypoint = tuple[float, float, float]


class PoseEstimator:
    """Runs pose estimation only on person-tracks close to a vehicle-track,
    to avoid the compute cost of pose on everyone in frame."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        proximity_norm: float = 0.15,
        pad_ratio: float = 0.15,
    ):
        self._model = YOLO(model_path)
        self._device = device
        self._proximity_norm = proximity_norm
        self._pad_ratio = pad_ratio

    def gated_person_ids(
        self,
        persons: list[TrackedObject],
        vehicles: list[TrackedObject],
        frame_diag: float,
    ) -> set[int]:
        gated: set[int] = set()
        for p in persons:
            for v in vehicles:
                dist = _centroid_distance(p.centroid, v.centroid)
                if frame_diag > 0 and dist / frame_diag <= self._proximity_norm:
                    gated.add(p.track_id)
                    break
        return gated

    def estimate(
        self,
        frame: np.ndarray,
        persons: list[TrackedObject],
        vehicles: list[TrackedObject],
    ) -> dict[int, list[Keypoint]]:
        h, w = frame.shape[:2]
        frame_diag = math.hypot(w, h)
        gated_ids = self.gated_person_ids(persons, vehicles, frame_diag)

        results: dict[int, list[Keypoint]] = {}
        for p in persons:
            if p.track_id not in gated_ids:
                continue
            crop, offset_x, offset_y = crop_with_padding(
                frame, p.bbox_xyxy, self._pad_ratio, w, h
            )
            if crop.size == 0:
                continue
            pred = self._model(crop, device=self._device, verbose=False)
            keypoints = pred[0].keypoints
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                continue
            xy = keypoints.xy[0].cpu().numpy()
            if keypoints.conf is not None:
                conf = keypoints.conf[0].cpu().numpy()
            else:
                conf = np.ones(len(xy))
            results[p.track_id] = [
                (float(x + offset_x), float(y + offset_y), float(c))
                for (x, y), c in zip(xy, conf)
            ]
        return results


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
