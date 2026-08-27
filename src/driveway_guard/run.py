import argparse
import hashlib
import json
import logging
from pathlib import Path

from driveway_guard.calibration.schema import CalibrationConfig
from driveway_guard.config import RunConfig
from driveway_guard.detection.tracker import Tracker
from driveway_guard.detection.weapon_detector import WeaponDetector
from driveway_guard.features.extractor import FeatureExtractor
from driveway_guard.output.event_log import write_csv, write_json
from driveway_guard.output.video_writer import AnnotatedVideoWriter
from driveway_guard.pipeline import Pipeline
from driveway_guard.pose.estimator import PoseEstimator
from driveway_guard.scoring.rules import RuleBasedScorer, RuleThresholds
from driveway_guard.sources.video_file import VideoFileSource


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Driveway anomaly detection pipeline")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--detector-model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--pose-proximity-norm", type=float, default=0.15)
    parser.add_argument(
        "--weapon-model",
        type=Path,
        default=None,
        help="Optional YOLO checkpoint fine-tuned for firearms; weapon-at-window "
        "detection is skipped entirely if not provided.",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video-output", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    return RunConfig(
        video_path=args.video,
        out_dir=args.out,
        calib_path=args.calib,
        detector_model=args.detector_model,
        pose_model=args.pose_model,
        pose_proximity_norm=args.pose_proximity_norm,
        weapon_model=args.weapon_model,
        conf=args.conf,
        device=args.device,
        frame_stride=args.frame_stride,
        write_video=not args.no_video_output,
        log_level=args.log_level,
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

    calibration = None
    if config.calib_path is not None:
        calibration = CalibrationConfig.load(
            config.calib_path,
            expected_width=source.frame_width,
            expected_height=source.frame_height,
        )

    tracker = Tracker(config.detector_model, conf=config.conf, device=config.device)
    pose_estimator = PoseEstimator(
        config.pose_model, device=config.device, proximity_norm=config.pose_proximity_norm
    )
    weapon_detector = None
    if config.weapon_model is not None:
        weapon_detector = WeaponDetector(str(config.weapon_model), device=config.device)
    else:
        logger.info("No --weapon-model provided; weapon-at-window detection is disabled.")

    thresholds = RuleThresholds()
    feature_extractor = FeatureExtractor(close_proximity_norm=thresholds.proximity_norm_threshold)
    scorer = RuleBasedScorer(thresholds)

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
        pose_estimator=pose_estimator,
        weapon_detector=weapon_detector,
        feature_extractor=feature_extractor,
        scorer=scorer,
        calibration=calibration,
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
                "calib_path": str(config.calib_path) if config.calib_path else None,
                "calib_hash": _file_hash(config.calib_path),
                "detector_model": config.detector_model,
                "pose_model": config.pose_model,
                "weapon_model": str(config.weapon_model) if config.weapon_model else None,
                "conf": config.conf,
                "device": config.device,
                "frame_stride": config.frame_stride,
                "num_events": len(pipeline.events),
            },
            indent=2,
        )
    )
    logger.info("Done. %d event(s) flagged. Output in %s", len(pipeline.events), config.out_dir)


if __name__ == "__main__":
    main()
