import logging
import math

import numpy as np

from driveway_guard.calibration.schema import CalibrationConfig
from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.features.convergence import compute_convergence
from driveway_guard.features.extractor import FeatureExtractor
from driveway_guard.output.overlay import draw_skeleton, draw_text_banner, draw_tracks
from driveway_guard.output.video_writer import AnnotatedVideoWriter
from driveway_guard.pose.estimator import COCO_POSE_EDGES, PoseEstimator
from driveway_guard.scoring.base import RiskScorer
from driveway_guard.scoring.events import FlaggedEvent
from driveway_guard.sources.base import FrameSource

logger = logging.getLogger(__name__)


class Pipeline:
    """Wires detection/tracking, pose, weapon detection, feature
    extraction, and risk scoring together per frame."""

    def __init__(
        self,
        tracker: Tracker,
        pose_estimator: PoseEstimator | None = None,
        weapon_detector: WeaponDetector | None = None,
        feature_extractor: FeatureExtractor | None = None,
        scorer: RiskScorer | None = None,
        calibration: CalibrationConfig | None = None,
        video_writer: AnnotatedVideoWriter | None = None,
    ):
        self._tracker = tracker
        self._pose_estimator = pose_estimator
        self._weapon_detector = weapon_detector
        self._feature_extractor = feature_extractor
        self._scorer = scorer
        self._calibration = calibration
        self._video_writer = video_writer
        self.events: list[FlaggedEvent] = []

    def process_frame(self, frame_idx: int, timestamp_s: float, frame: np.ndarray) -> np.ndarray:
        tracked_objects = self._tracker.track(frame)
        persons = [o for o in tracked_objects if o.cls == ObjectClass.PERSON]
        vehicles = [o for o in tracked_objects if o.cls == ObjectClass.VEHICLE]

        annotated = draw_tracks(frame.copy(), tracked_objects)

        poses = {}
        if self._pose_estimator is not None:
            poses = self._pose_estimator.estimate(frame, persons, vehicles)
            for keypoints in poses.values():
                xy = [(x, y) for x, y, _ in keypoints]
                annotated = draw_skeleton(annotated, xy, COCO_POSE_EDGES)

        weapon_hits = {}
        if self._weapon_detector is not None:
            weapon_hits = self._weapon_detector.detect(frame, persons, vehicles)

        if self._feature_extractor is None:
            return annotated

        h, w = frame.shape[:2]
        frame_diag = math.hypot(w, h)
        pair_records = self._feature_extractor.extract_pairs(
            frame_idx, timestamp_s, frame_diag, persons, vehicles, poses, self._calibration
        )
        for record in pair_records:
            if record.person_track_id in weapon_hits:
                conf, bbox = weapon_hits[record.person_track_id]
                record.weapon_detected = True
                record.weapon_confidence = conf
                record.weapon_bbox_xyxy = bbox

        blocking_observations = self._feature_extractor.extract_blocking(
            frame_idx, timestamp_s, vehicles, persons, self._calibration
        )
        convergence_records = compute_convergence(frame_idx, timestamp_s, pair_records)

        banner: list[str] = []
        if self._scorer is not None:
            new_events = self._scorer.process_frame(
                frame_idx, timestamp_s, pair_records, blocking_observations, convergence_records
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
            banner = sorted(active_types)

        if banner:
            annotated = draw_text_banner(annotated, [f"RISK: {t}" for t in banner])

        return annotated

    def run(self, source: FrameSource) -> None:
        for frame_idx, timestamp_s, frame in source:
            annotated = self.process_frame(frame_idx, timestamp_s, frame)
            if self._video_writer is not None:
                self._video_writer.write(annotated)
            if frame_idx % 100 == 0:
                logger.info("processed frame %d (t=%.2fs)", frame_idx, timestamp_s)
