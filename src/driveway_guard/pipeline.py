import logging
import math

import cv2
import numpy as np

from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.output.overlay import draw_text_banner, draw_tracks
from driveway_guard.output.video_writer import AnnotatedVideoWriter
from driveway_guard.scoring.events import FlaggedEvent
from driveway_guard.scoring.weapon import WeaponScorer
from driveway_guard.sources.base import FrameSource

logger = logging.getLogger(__name__)

_WEAPON_BOX_COLOR = (0, 0, 255)


class Pipeline:
    """Wires detection/tracking, weapon detection, and weapon-event scoring
    together per frame.

    Weapon detection only for now -- struggle/boxing-in/sprint/convergence
    are retired pending their own incremental rebuild, see HANDOVER.md and
    the approved plan for this phase.
    """

    def __init__(
        self,
        tracker: Tracker,
        weapon_detector: WeaponDetector,
        scorer: WeaponScorer,
        proximity_norm: float,
        video_writer: AnnotatedVideoWriter | None = None,
    ):
        self._tracker = tracker
        self._weapon_detector = weapon_detector
        self._scorer = scorer
        self._proximity_norm = proximity_norm
        self._video_writer = video_writer
        self.events: list[FlaggedEvent] = []

    def process_frame(self, frame_idx: int, timestamp_s: float, frame: np.ndarray) -> np.ndarray:
        tracked_objects = self._tracker.track(frame)
        persons = [o for o in tracked_objects if o.cls == ObjectClass.PERSON]
        vehicles = [o for o in tracked_objects if o.cls == ObjectClass.VEHICLE]

        annotated = draw_tracks(frame.copy(), tracked_objects)

        weapon_hits = self._weapon_detector.detect(frame, persons, vehicles)
        for confidence, bbox in weapon_hits.values():
            x1, y1, x2, y2 = (int(c) for c in bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _WEAPON_BOX_COLOR, 2)
            cv2.putText(
                annotated,
                f"weapon {confidence:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _WEAPON_BOX_COLOR,
                2,
            )

        h, w = frame.shape[:2]
        frame_diag = math.hypot(w, h)
        new_events = self._scorer.process_frame(
            frame_idx,
            timestamp_s,
            persons,
            vehicles,
            weapon_hits,
            frame_diag,
            self._proximity_norm,
        )
        if new_events:
            self.events.extend(new_events)
            for event in new_events:
                logger.warning(
                    "FLAGGED %s at t=%.2fs (track_ids=%s, peak_score=%.2f)",
                    event.event_type,
                    event.start_timestamp_s,
                    event.track_ids,
                    event.peak_score,
                )

        active_types = {e.event_type for e in self.events if e.end_timestamp_s >= timestamp_s - 3.0}
        if active_types:
            annotated = draw_text_banner(annotated, [f"RISK: {t}" for t in sorted(active_types)])

        return annotated

    def run(self, source: FrameSource) -> None:
        for frame_idx, timestamp_s, frame in source:
            annotated = self.process_frame(frame_idx, timestamp_s, frame)
            if self._video_writer is not None:
                self._video_writer.write(annotated)
            if frame_idx % 100 == 0:
                logger.info("processed frame %d (t=%.2fs)", frame_idx, timestamp_s)
