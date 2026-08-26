from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    video_path: Path
    out_dir: Path
    calib_path: Path | None = None
    detector_model: str = "yolo11n.pt"
    pose_model: str = "yolo11n-pose.pt"
    pose_proximity_norm: float = 0.15
    weapon_model: Path | None = None
    conf: float = 0.35
    device: str = "cpu"
    frame_stride: int = 1
    write_video: bool = True
    log_level: str = "INFO"
