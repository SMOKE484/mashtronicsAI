from dataclasses import dataclass
from enum import Enum


class ObjectClass(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"


@dataclass(slots=True)
class TrackedObject:
    track_id: int
    cls: ObjectClass
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
