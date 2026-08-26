import numpy as np
from ultralytics import YOLO

from driveway_guard.detection.types import ObjectClass, TrackedObject

# COCO class ids relevant to driveway monitoring.
_PERSON_CLASS_ID = 0
_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
_TRACKED_CLASS_IDS = {_PERSON_CLASS_ID, *_VEHICLE_CLASS_IDS}


class Tracker:
    """Wraps Ultralytics YOLO detection + ByteTrack tracking.

    Kept behind this interface so a manual detector+tracker pipeline
    (e.g. a separate detector with the `supervision` library's ByteTrack)
    could be substituted later without touching downstream stages.
    """

    def __init__(self, model_path: str, conf: float = 0.35, device: str = "cpu"):
        self._model = YOLO(model_path)
        self._conf = conf
        self._device = device

    def track(self, frame: np.ndarray) -> list[TrackedObject]:
        results = self._model.track(
            frame,
            persist=True,
            conf=self._conf,
            classes=list(_TRACKED_CLASS_IDS),
            device=self._device,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracked: list[TrackedObject] = []
        track_ids = boxes.id.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for track_id, class_id, confidence, box in zip(
            track_ids, class_ids, confidences, xyxy
        ):
            cls = (
                ObjectClass.PERSON
                if int(class_id) == _PERSON_CLASS_ID
                else ObjectClass.VEHICLE
            )
            tracked.append(
                TrackedObject(
                    track_id=int(track_id),
                    cls=cls,
                    confidence=float(confidence),
                    bbox_xyxy=(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                )
            )
        return tracked
