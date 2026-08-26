import math

import numpy as np
from ultralytics import YOLO

from driveway_guard.detection.types import TrackedObject
from driveway_guard.imaging import crop_with_padding


class WeaponDetector:
    """Proximity-gated firearm detection near a vehicle window/door.

    Expects a YOLO-format checkpoint fine-tuned for firearms — none is
    bundled with this project; see README for sourcing one. This is the
    highest-uncertainty component of v1: small/dark-object detection is
    false-positive-prone (phones, dark clothing folds), so a positive
    result here should feed a sustained-frames debounce in the scorer
    rather than firing on a single frame, and needs its own precision
    evaluation on real footage before being trusted.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf: float = 0.4,
        proximity_norm: float = 0.15,
        pad_ratio: float = 0.4,
    ):
        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
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
                dist = math.hypot(p.centroid[0] - v.centroid[0], p.centroid[1] - v.centroid[1])
                if frame_diag > 0 and dist / frame_diag <= self._proximity_norm:
                    gated.add(p.track_id)
                    break
        return gated

    def detect(
        self,
        frame: np.ndarray,
        persons: list[TrackedObject],
        vehicles: list[TrackedObject],
    ) -> dict[int, tuple[float, tuple[float, float, float, float]]]:
        """Returns {person_track_id: (confidence, bbox_xyxy_in_frame)} for
        gated persons with a detected weapon above threshold."""
        h, w = frame.shape[:2]
        frame_diag = math.hypot(w, h)
        gated_ids = self.gated_person_ids(persons, vehicles, frame_diag)

        results: dict[int, tuple[float, tuple[float, float, float, float]]] = {}
        for p in persons:
            if p.track_id not in gated_ids:
                continue
            crop, offset_x, offset_y = crop_with_padding(
                frame, p.bbox_xyxy, self._pad_ratio, w, h
            )
            if crop.size == 0:
                continue
            pred = self._model(crop, device=self._device, conf=self._conf, verbose=False)
            boxes = pred[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            confs = boxes.conf.cpu().numpy()
            best_idx = int(confs.argmax())
            best_conf = float(confs[best_idx])
            x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best_idx]
            results[p.track_id] = (
                best_conf,
                (float(x1 + offset_x), float(y1 + offset_y), float(x2 + offset_x), float(y2 + offset_y)),
            )
        return results
