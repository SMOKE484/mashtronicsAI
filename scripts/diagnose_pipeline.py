"""Diagnose why a clip produces 0 (or few) flagged events, without the
EventAggregator's threshold+debounce hiding how close the raw signals got.

Standalone utility, not part of the `driveway_guard` package. Runs the same
detection -> pose -> weapon -> feature-extraction -> scoring stages the real
pipeline runs, but calls score_struggle/score_sprint/score_weapon/
score_convergence directly per frame and tracks the *best* value each signal
ever reaches across the whole clip, instead of only reporting flagged
events. Comparing those maxes against RuleThresholds tells you which of two
different problems you have:

- Maxes close to but under threshold -> a tuning problem (thresholds are
  stricter than real footage warrants).
- Maxes nowhere close -> a detection/tracking/pose robustness problem on
  this footage (or the scenario genuinely doesn't produce the signal the
  rule is looking for at all).

Usage:
    python scripts/diagnose_pipeline.py --video /path/to/clip.mp4 [--weapon-model weapon_model.pt] [--calib calib.json] [--device cuda:0]
"""

import argparse
import json
import logging
import math
from pathlib import Path

from driveway_guard.calibration.schema import CalibrationConfig
from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.features.convergence import compute_convergence
from driveway_guard.features.extractor import FeatureExtractor
from driveway_guard.pose.estimator import PoseEstimator
from driveway_guard.scoring.rules import (
    RuleThresholds,
    score_convergence,
    score_sprint,
    score_struggle,
    score_weapon,
)
from driveway_guard.sources.video_file import VideoFileSource

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose raw signal levels across a whole clip, bypassing event debounce"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write summary JSON")
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--pose-proximity-norm", type=float, default=0.15)
    parser.add_argument("--weapon-model", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")

    thresholds = RuleThresholds()
    source = VideoFileSource(args.video, frame_stride=args.frame_stride)
    logger.info(
        "resolution: %dx%d | fps: %.3f | frame_count: %d",
        source.frame_width,
        source.frame_height,
        source.fps,
        source.frame_count,
    )

    calibration = None
    if args.calib is not None:
        calibration = CalibrationConfig.load(
            args.calib, expected_width=source.frame_width, expected_height=source.frame_height
        )

    tracker = Tracker(args.detector_model, conf=args.conf, device=args.device)
    pose_estimator = PoseEstimator(
        args.pose_model, device=args.device, proximity_norm=args.pose_proximity_norm
    )
    weapon_detector = None
    if args.weapon_model is not None:
        weapon_detector = WeaponDetector(str(args.weapon_model), device=args.device)
    feature_extractor = FeatureExtractor(close_proximity_norm=thresholds.proximity_norm_threshold)

    stats = {
        "frames_total": 0,
        "frames_with_pair": 0,
        "person_frames": 0,
        "vehicle_frames": 0,
        "min_distance_norm": math.inf,
        "max_dwell_s": 0.0,
        "max_close_dwell_s": 0.0,
        "max_approach_speed_px_s": -math.inf,
        "max_struggle_score": 0.0,
        "max_sprint_score": 0.0,
        "max_weapon_score": 0.0,
        "max_convergence_approachers": 0,
        "max_convergence_spread_deg": 0.0,
        "max_convergence_score": 0.0,
    }

    try:
        for frame_idx, timestamp_s, frame in source:
            stats["frames_total"] += 1
            tracked = tracker.track(frame)
            persons = [o for o in tracked if o.cls == ObjectClass.PERSON]
            vehicles = [o for o in tracked if o.cls == ObjectClass.VEHICLE]
            if persons:
                stats["person_frames"] += 1
            if vehicles:
                stats["vehicle_frames"] += 1

            poses = pose_estimator.estimate(frame, persons, vehicles)

            weapon_hits = {}
            if weapon_detector is not None:
                weapon_hits = weapon_detector.detect(frame, persons, vehicles)

            h, w = frame.shape[:2]
            frame_diag = math.hypot(w, h)
            pair_records = feature_extractor.extract_pairs(
                frame_idx, timestamp_s, frame_diag, persons, vehicles, poses, calibration
            )
            for record in pair_records:
                if record.person_track_id in weapon_hits:
                    conf, bbox = weapon_hits[record.person_track_id]
                    record.weapon_detected = True
                    record.weapon_confidence = conf
                    record.weapon_bbox_xyxy = bbox

            if pair_records:
                stats["frames_with_pair"] += 1
            for r in pair_records:
                stats["min_distance_norm"] = min(stats["min_distance_norm"], r.person_vehicle_distance_norm)
                stats["max_dwell_s"] = max(stats["max_dwell_s"], r.dwell_time_s)
                stats["max_close_dwell_s"] = max(stats["max_close_dwell_s"], r.close_dwell_time_s)
                stats["max_approach_speed_px_s"] = max(stats["max_approach_speed_px_s"], r.approach_speed_px_s)
                stats["max_struggle_score"] = max(stats["max_struggle_score"], score_struggle(r, thresholds))
                stats["max_sprint_score"] = max(stats["max_sprint_score"], score_sprint(r, thresholds))
                stats["max_weapon_score"] = max(stats["max_weapon_score"], score_weapon(r, thresholds))

            vehicle_velocities = {
                v.track_id: feature_extractor.vehicle_velocity_px_s(v.track_id) for v in vehicles
            }
            convergence_records = compute_convergence(
                frame_idx, timestamp_s, pair_records, vehicles, vehicle_velocities, calibration
            )
            for conv in convergence_records:
                stats["max_convergence_approachers"] = max(
                    stats["max_convergence_approachers"], conv.num_simultaneous_approachers
                )
                stats["max_convergence_spread_deg"] = max(
                    stats["max_convergence_spread_deg"], conv.angular_spread_deg
                )
                stats["max_convergence_score"] = max(
                    stats["max_convergence_score"], score_convergence(conv, thresholds)
                )

            if frame_idx % 100 == 0:
                logger.info("processed frame %d (t=%.2fs)", frame_idx, timestamp_s)
    finally:
        source.close()

    if stats["min_distance_norm"] == math.inf:
        stats["min_distance_norm"] = None
    if stats["max_approach_speed_px_s"] == -math.inf:
        stats["max_approach_speed_px_s"] = None

    reference_thresholds = {
        "proximity_norm_threshold": thresholds.proximity_norm_threshold,
        "struggle_dwell_min_s": thresholds.struggle_dwell_min_s,
        "sprint_speed_px_s_threshold": thresholds.sprint_speed_px_s_threshold,
        "weapon_confidence_threshold": thresholds.weapon_confidence_threshold,
        "convergence_min_approachers": thresholds.convergence_min_approachers,
        "convergence_angle_threshold_deg": thresholds.convergence_angle_threshold_deg,
        "risk_score_flag_threshold": thresholds.risk_score_flag_threshold,
    }

    result = {"video": str(args.video), "stats": stats, "reference_thresholds": reference_thresholds}
    logger.info("Diagnostic result:\n%s", json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        logger.info("Wrote summary to %s", args.out)


if __name__ == "__main__":
    main()
