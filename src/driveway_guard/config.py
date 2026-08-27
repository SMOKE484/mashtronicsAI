from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    video_path: Path
    out_dir: Path
    weapon_model: Path
    detector_model: str = "yolo11n.pt"
    conf: float = 0.35
    device: str = "cpu"
    frame_stride: int = 1
    write_video: bool = True
    log_level: str = "INFO"

    # WeaponDetector tunables (crop-level firearm detection near a person).
    weapon_conf: float = 0.4
    weapon_proximity_norm: float = 0.15
    weapon_pad_ratio: float = 0.4

    # WeaponThresholds tunables (event-level debounce). See HANDOVER.md/plan
    # "Validation against real footage" -- weapon_min_duration_s in
    # particular is expected to need tuning against real clips.
    weapon_confidence_threshold: float = 0.5
    weapon_min_duration_s: float = 0.5
    event_cooldown_s: float = 5.0
    weapon_max_gap_s: float = 0.15
