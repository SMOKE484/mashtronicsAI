"""Deep-dive on one exact frame: every tracked person/vehicle, whether each
person was gated as "near a vehicle" for weapon detection, and every raw
weapon-model box found in each gated person's crop -- not just the single
highest-confidence one WeaponDetector.detect() keeps.

Built to diagnose a specific failure mode: two people both visibly holding
weapons in the same frame, but the pipeline only ever reports one. Three
different root causes look identical from the outside (both just show up
as "second gun never appears in events.json"), and this script tells them
apart:
  (a) the second person was never tracked as their own person at all
  (b) they were tracked and gated, but the weapon model found nothing (or
      nothing above threshold) in their individual crop
  (c) they were tracked and gated, the weapon model DID find something in
      their crop, but WeaponDetector.detect() only keeps the single
      highest-confidence threat box per crop -- so if both people's guns
      ended up visible in the same crop (e.g. they're standing close
      together and one person's padded crop includes the other), only one
      of the two ever survives.

Also writes an annotated PNG with every person/vehicle box, track ID, and
(if a weapon model is given) every raw weapon-model box found, so the
frame can be checked visually against what you saw in the actual clip.

Usage:
    python scripts/inspect_frame.py --video clip.mp4 --at-seconds 11.5 --weapon-model weapon_model.pt --out frame.png --device cuda:0
    python scripts/inspect_frame.py --video clip.mp4 --frame-idx 466 --weapon-model weapon_model.pt --out frame.png --device cuda:0
"""

import argparse
import logging
import math
from pathlib import Path

import cv2

from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.types import ObjectClass
from driveway_guard.imaging import crop_with_padding
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_COLOR = (0, 220, 0)
_VEHICLE_COLOR = (0, 140, 255)
_WEAPON_COLOR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive on every detection in one exact frame")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--at-seconds", type=float, default=None, help="Alternative to --frame-idx")
    parser.add_argument("--weapon-model", type=Path, default=None)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--out", type=Path, default=None, help="Path to write the annotated frame PNG")
    parser.add_argument("--proximity-norm", type=float, default=0.15, help="Weapon-gating proximity, same default as WeaponDetector")
    parser.add_argument("--pad-ratio", type=float, default=0.4, help="Crop padding, same default as WeaponDetector")
    parser.add_argument("--weapon-conf", type=float, default=0.1, help="Low YOLO conf cutoff so weak/discarded boxes are still shown")
    parser.add_argument("--tracker-conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.frame_idx is None and args.at_seconds is None:
        parser.error("Provide either --frame-idx or --at-seconds")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(message)s")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = args.frame_idx if args.frame_idx is not None else round(args.at_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {args.video}")
    timestamp_s = frame_idx / fps
    logger.info("frame %d (t=%.2fs, fps=%.2f)", frame_idx, timestamp_s, fps)

    tracker = Tracker(args.detector_model, conf=args.tracker_conf, device=args.device)
    tracked = tracker.track(frame)
    persons = [o for o in tracked if o.cls == ObjectClass.PERSON]
    vehicles = [o for o in tracked if o.cls == ObjectClass.VEHICLE]

    h, w = frame.shape[:2]
    frame_diag = math.hypot(w, h)
    annotated = frame.copy()

    logger.info("")
    logger.info("=== Tracked objects ===")
    for v in vehicles:
        x1, y1, x2, y2 = (int(c) for c in v.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _VEHICLE_COLOR, 2)
        cv2.putText(annotated, f"vehicle#{v.track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _VEHICLE_COLOR, 2)
        logger.info("vehicle#%d conf=%.2f bbox=%s", v.track_id, v.confidence, tuple(round(c) for c in v.bbox_xyxy))

    for p in persons:
        x1, y1, x2, y2 = (int(c) for c in p.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _PERSON_COLOR, 2)
        cv2.putText(annotated, f"person#{p.track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PERSON_COLOR, 2)
        nearest = min(
            (math.hypot(p.centroid[0] - v.centroid[0], p.centroid[1] - v.centroid[1]) / frame_diag for v in vehicles),
            default=None,
        )
        gated = nearest is not None and nearest <= args.proximity_norm
        logger.info(
            "person#%d conf=%.2f bbox=%s nearest_vehicle_dist_norm=%s gated_for_weapon_check=%s",
            p.track_id,
            p.confidence,
            tuple(round(c) for c in p.bbox_xyxy),
            f"{nearest:.3f}" if nearest is not None else "n/a (no vehicle tracked)",
            gated,
        )

    if args.weapon_model is not None:
        logger.info("")
        logger.info("=== Raw weapon-model output per gated person's crop (ALL boxes, not just top-1) ===")
        weapon_model = YOLO(str(args.weapon_model))
        for p in persons:
            nearest = min(
                (math.hypot(p.centroid[0] - v.centroid[0], p.centroid[1] - v.centroid[1]) / frame_diag for v in vehicles),
                default=None,
            )
            if nearest is None or nearest > args.proximity_norm:
                logger.info("person#%d: not gated, weapon model not run on their crop", p.track_id)
                continue
            crop, offset_x, offset_y = crop_with_padding(frame, p.bbox_xyxy, args.pad_ratio, w, h)
            if crop.size == 0:
                logger.info("person#%d: empty crop, skipped", p.track_id)
                continue
            pred = weapon_model(crop, device=args.device, conf=args.weapon_conf, verbose=False)
            boxes = pred[0].boxes
            if boxes is None or len(boxes) == 0:
                logger.info("person#%d: weapon model found nothing in their crop (conf>=%.2f)", p.track_id, args.weapon_conf)
                continue
            names = weapon_model.names
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for conf, cls_id, box in sorted(zip(confs, cls_ids, xyxy), key=lambda t: -t[0]):
                class_name = names.get(int(cls_id), str(cls_id))
                bx1, by1, bx2, by2 = box
                fx1, fy1, fx2, fy2 = bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y
                logger.info(
                    "person#%d: class=%s confidence=%.3f frame_bbox=%s",
                    p.track_id,
                    class_name,
                    conf,
                    (round(fx1), round(fy1), round(fx2), round(fy2)),
                )
                if class_name.lower() != "person":
                    cv2.rectangle(annotated, (int(fx1), int(fy1)), (int(fx2), int(fy2)), _WEAPON_COLOR, 2)
                    cv2.putText(
                        annotated,
                        f"{class_name} {conf:.2f}",
                        (int(fx1), max(0, int(fy1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        _WEAPON_COLOR,
                        2,
                    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), annotated)
        logger.info("")
        logger.info("Wrote annotated frame to %s", args.out)


if __name__ == "__main__":
    main()
