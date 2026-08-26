"""Save annotated snapshot images of the weapon detector's strongest hits on
a clip, for visual spot-checking (is it really a gun, or a phone/dark
object it's confusing for one?).

Standalone utility, not part of the `driveway_guard` package. Picks the
--count highest-confidence hits at least --min-gap-s apart in time (so five
snapshots aren't just five near-duplicate consecutive frames from the same
instant), draws the weapon box (red) and the person box (yellow) on a
full-frame copy, and also saves a padded close-up crop around the weapon
box for each one.

Usage:
    python scripts/export_weapon_snapshots.py --video clip.mp4 --weapon-model weapon_model.pt --out snapshots/ --device cuda:0
"""

import argparse
import logging
from pathlib import Path

import cv2

from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.imaging import crop_with_padding
from driveway_guard.sources.video_file import VideoFileSource

logger = logging.getLogger(__name__)

_WEAPON_COLOR = (0, 0, 255)
_PERSON_COLOR = (0, 220, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotated snapshots of weapon-detector hits")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weapon-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-conf", type=float, default=0.5, help="Only consider hits at/above this confidence")
    parser.add_argument("--min-gap-s", type=float, default=0.3, help="Minimum time between selected snapshots")
    parser.add_argument("--conf", type=float, default=0.4, help="Weapon-model YOLO conf cutoff")
    parser.add_argument("--proximity-norm", type=float, default=0.15)
    parser.add_argument("--tracker-conf", type=float, default=0.35, help="Person/vehicle detector conf")
    parser.add_argument("--crop-pad-ratio", type=float, default=1.0, help="Padding around the weapon box for the close-up crop")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    source = VideoFileSource(args.video, frame_stride=args.frame_stride)
    tracker = Tracker(args.detector_model, conf=args.tracker_conf, device=args.device)
    weapon_detector = WeaponDetector(
        str(args.weapon_model), device=args.device, conf=args.conf, proximity_norm=args.proximity_norm
    )

    candidates = []  # (frame_idx, timestamp_s, person_track_id, confidence, weapon_bbox, person_bbox)
    frames_cache = {}

    try:
        for frame_idx, timestamp_s, frame in source:
            tracked = tracker.track(frame)
            persons = [o for o in tracked if o.cls == ObjectClass.PERSON]
            vehicles = [o for o in tracked if o.cls == ObjectClass.VEHICLE]
            weapon_hits = weapon_detector.detect(frame, persons, vehicles)
            for person_track_id, (conf, weapon_bbox) in weapon_hits.items():
                if conf < args.min_conf:
                    continue
                person_bbox = next(
                    (p.bbox_xyxy for p in persons if p.track_id == person_track_id), None
                )
                candidates.append(
                    (frame_idx, timestamp_s, person_track_id, conf, weapon_bbox, person_bbox)
                )
                frames_cache[frame_idx] = frame.copy()
    finally:
        source.close()

    if not candidates:
        logger.info(
            "No hits at/above confidence %.2f found -- nothing to export. Try lowering --min-conf.",
            args.min_conf,
        )
        return

    candidates.sort(key=lambda c: c[3], reverse=True)
    selected = []
    for cand in candidates:
        ts = cand[1]
        if all(abs(ts - s[1]) >= args.min_gap_s for s in selected):
            selected.append(cand)
        if len(selected) >= args.count:
            break
    selected.sort(key=lambda c: c[1])

    logger.info(
        "Selected %d snapshot(s) from %d hit(s) at/above confidence %.2f:",
        len(selected),
        len(candidates),
        args.min_conf,
    )
    h, w = next(iter(frames_cache.values())).shape[:2]
    for frame_idx, timestamp_s, person_track_id, conf, weapon_bbox, person_bbox in selected:
        annotated = frames_cache[frame_idx].copy()

        wx1, wy1, wx2, wy2 = (int(v) for v in weapon_bbox)
        cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), _WEAPON_COLOR, 2)
        cv2.putText(
            annotated,
            f"weapon {conf:.2f}",
            (wx1, max(0, wy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _WEAPON_COLOR,
            2,
        )
        if person_bbox is not None:
            px1, py1, px2, py2 = (int(v) for v in person_bbox)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), _PERSON_COLOR, 1)

        stem = f"frame{frame_idx:05d}_t{timestamp_s:.2f}s_conf{conf:.2f}"
        full_path = args.out / f"{stem}_full.png"
        cv2.imwrite(str(full_path), annotated)

        crop, _ox, _oy = crop_with_padding(annotated, weapon_bbox, args.crop_pad_ratio, w, h)
        crop_path = args.out / f"{stem}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        logger.info(
            "frame %d t=%.2fs person=%d confidence=%.3f -> %s, %s",
            frame_idx,
            timestamp_s,
            person_track_id,
            conf,
            full_path.name,
            crop_path.name,
        )


if __name__ == "__main__":
    main()
