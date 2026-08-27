"""Print every raw weapon-detector hit on a clip, frame by frame, plus a
summary of whether any run of hits was long enough to have survived the
real pipeline's threshold+duration debounce.

Standalone utility, not part of the `driveway_guard` package. Narrower than
diagnose_pipeline.py: skips pose estimation and feature extraction entirely
and only runs Tracker + WeaponDetector, so you can see exactly what the
weapon model saw on a specific clip without anything else in the way --
useful when diagnose_pipeline.py's max_weapon_score clears
weapon_confidence_threshold but no weapon_at_window event ever fires,
which means the hit(s) never lasted weapon_min_duration_s in a row.

Usage:
    python scripts/inspect_weapon_hits.py --video /path/to/clip.mp4 --weapon-model weapon_model.pt --device cuda:0 [--conf 0.25]
"""

import argparse
import logging
from pathlib import Path

from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.scoring.weapon import WeaponThresholds
from driveway_guard.sources.video_file import VideoFileSource

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print raw weapon-detector hits frame by frame")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weapon-model", required=True, type=Path)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.4, help="Weapon-model YOLO conf cutoff (lower to see weaker signal)")
    parser.add_argument("--proximity-norm", type=float, default=0.15)
    parser.add_argument("--tracker-conf", type=float, default=0.35, help="Person/vehicle detector conf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(message)s")

    thresholds = WeaponThresholds()
    source = VideoFileSource(args.video, frame_stride=args.frame_stride)
    logger.info(
        "resolution: %dx%d | fps: %.3f | frame_count: %d | scoring uses "
        "weapon_confidence_threshold=%.2f, weapon_min_duration_s=%.2f",
        source.frame_width,
        source.frame_height,
        source.fps,
        source.frame_count,
        thresholds.weapon_confidence_threshold,
        thresholds.weapon_min_duration_s,
    )

    tracker = Tracker(args.detector_model, conf=args.tracker_conf, device=args.device)
    weapon_detector = WeaponDetector(
        str(args.weapon_model), device=args.device, conf=args.conf, proximity_norm=args.proximity_norm
    )

    hits: list[tuple[int, float, int, float]] = []  # frame_idx, timestamp_s, person_track_id, confidence
    frames_total = 0

    try:
        for frame_idx, timestamp_s, frame in source:
            frames_total += 1
            tracked = tracker.track(frame)
            persons = [o for o in tracked if o.cls == ObjectClass.PERSON]
            vehicles = [o for o in tracked if o.cls == ObjectClass.VEHICLE]
            weapon_hits = weapon_detector.detect(frame, persons, vehicles)
            for person_track_id, (conf, _bbox) in weapon_hits.items():
                hits.append((frame_idx, timestamp_s, person_track_id, conf))
                logger.info(
                    "frame %5d t=%6.2fs person=%d confidence=%.3f%s",
                    frame_idx,
                    timestamp_s,
                    person_track_id,
                    conf,
                    "  <-- clears weapon_confidence_threshold"
                    if conf >= thresholds.weapon_confidence_threshold
                    else "",
                )
    finally:
        source.close()

    above_threshold = [h for h in hits if h[3] >= thresholds.weapon_confidence_threshold]

    longest_run_s = 0.0
    if above_threshold:
        run_start_ts = above_threshold[0][1]
        prev_ts = above_threshold[0][1]
        for _frame_idx, ts, _pid, _conf in above_threshold[1:]:
            gap_s = ts - prev_ts
            if gap_s > 5.0 / max(source.fps, 1.0):
                longest_run_s = max(longest_run_s, prev_ts - run_start_ts)
                run_start_ts = ts
            prev_ts = ts
        longest_run_s = max(longest_run_s, prev_ts - run_start_ts)

    logger.info("")
    logger.info("=== Summary ===")
    logger.info("frames_total: %d", frames_total)
    logger.info("frames_with_any_weapon_hit (conf >= --conf %.2f): %d", args.conf, len(hits))
    logger.info(
        "frames_with_hit_above_score_threshold (conf >= %.2f): %d",
        thresholds.weapon_confidence_threshold,
        len(above_threshold),
    )
    logger.info(
        "longest_continuous_run_above_threshold_s: %.2f (needs >= %.2f to have fired weapon_at_window)",
        longest_run_s,
        thresholds.weapon_min_duration_s,
    )
    if hits:
        max_conf = max(h[3] for h in hits)
        logger.info("max_confidence_seen: %.3f", max_conf)
    else:
        logger.info(
            "No weapon hits at all at --conf %.2f -- try lowering --conf to see if there's "
            "weaker signal being filtered out before it ever reaches the confidence-threshold check.",
            args.conf,
        )


if __name__ == "__main__":
    main()
