import argparse
import hashlib
import json
import logging
from pathlib import Path

from driveway_guard.config import RunConfig
from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.output.event_log import write_csv, write_json
from driveway_guard.output.video_writer import AnnotatedVideoWriter
from driveway_guard.pipeline import Pipeline
from driveway_guard.scoring.weapon import WeaponScorer, WeaponThresholds
from driveway_guard.sources.video_file import VideoFileSource


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Driveway weapon-at-window detection pipeline (weapon detection only -- "
        "see HANDOVER.md/plan for the other event types, currently retired)"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--weapon-model", required=True, type=Path, help="YOLO checkpoint fine-tuned for firearms"
    )
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video-output", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--weapon-conf", type=float, default=0.4)
    parser.add_argument("--weapon-proximity-norm", type=float, default=0.15)
    parser.add_argument("--weapon-pad-ratio", type=float, default=0.4)
    parser.add_argument("--weapon-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--weapon-min-duration-s", type=float, default=0.5)
    parser.add_argument("--event-cooldown-s", type=float, default=5.0)
    parser.add_argument(
        "--weapon-max-gap-s",
        type=float,
        default=0.15,
        help="Tolerate a below-threshold dip up to this long without resetting the duration debounce",
    )
    args = parser.parse_args()

    return RunConfig(
        video_path=args.video,
        out_dir=args.out,
        weapon_model=args.weapon_model,
        detector_model=args.detector_model,
        conf=args.conf,
        device=args.device,
        frame_stride=args.frame_stride,
        write_video=not args.no_video_output,
        log_level=args.log_level,
        weapon_conf=args.weapon_conf,
        weapon_proximity_norm=args.weapon_proximity_norm,
        weapon_pad_ratio=args.weapon_pad_ratio,
        weapon_confidence_threshold=args.weapon_confidence_threshold,
        weapon_min_duration_s=args.weapon_min_duration_s,
        event_cooldown_s=args.event_cooldown_s,
        weapon_max_gap_s=args.weapon_max_gap_s,
    )


def _file_hash(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> None:
    config = parse_args()
    logging.basicConfig(
        level=config.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    config.out_dir.mkdir(parents=True, exist_ok=True)

    source = VideoFileSource(config.video_path, frame_stride=config.frame_stride)

    tracker = Tracker(config.detector_model, conf=config.conf, device=config.device)
    weapon_detector = WeaponDetector(
        str(config.weapon_model),
        device=config.device,
        conf=config.weapon_conf,
        proximity_norm=config.weapon_proximity_norm,
        pad_ratio=config.weapon_pad_ratio,
    )
    thresholds = WeaponThresholds(
        weapon_confidence_threshold=config.weapon_confidence_threshold,
        weapon_min_duration_s=config.weapon_min_duration_s,
        event_cooldown_s=config.event_cooldown_s,
        weapon_max_gap_s=config.weapon_max_gap_s,
    )
    scorer = WeaponScorer(thresholds)

    video_writer = None
    if config.write_video:
        video_writer = AnnotatedVideoWriter(
            config.out_dir / "annotated.mp4",
            fps=source.fps,
            width=source.frame_width,
            height=source.frame_height,
        )

    pipeline = Pipeline(
        tracker=tracker,
        weapon_detector=weapon_detector,
        scorer=scorer,
        proximity_norm=config.weapon_proximity_norm,
        video_writer=video_writer,
    )
    try:
        pipeline.run(source)
    finally:
        source.close()
        if video_writer is not None:
            video_writer.close()

    write_json(pipeline.events, config.out_dir / "events.json")
    write_csv(pipeline.events, config.out_dir / "events.csv")

    if pipeline.events:
        logger.info("Flagged events (sorted by time):")
        for event in sorted(pipeline.events, key=lambda e: e.start_timestamp_s):
            logger.info(
                "  [%s] t=%.2fs-%.2fs  score=%.2f  track_ids=%s",
                event.event_type,
                event.start_timestamp_s,
                event.end_timestamp_s,
                event.peak_score,
                event.track_ids,
            )
    else:
        logger.info("No events flagged.")

    (config.out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "video_path": str(config.video_path),
                "detector_model": config.detector_model,
                "weapon_model": str(config.weapon_model),
                "weapon_model_hash": _file_hash(config.weapon_model),
                "conf": config.conf,
                "device": config.device,
                "frame_stride": config.frame_stride,
                "weapon_conf": config.weapon_conf,
                "weapon_proximity_norm": config.weapon_proximity_norm,
                "weapon_pad_ratio": config.weapon_pad_ratio,
                "weapon_confidence_threshold": config.weapon_confidence_threshold,
                "weapon_min_duration_s": config.weapon_min_duration_s,
                "event_cooldown_s": config.event_cooldown_s,
                "weapon_max_gap_s": config.weapon_max_gap_s,
                "num_events": len(pipeline.events),
            },
            indent=2,
        )
    )
    logger.info("Done. %d event(s) flagged. Output in %s", len(pipeline.events), config.out_dir)


if __name__ == "__main__":
    main()
