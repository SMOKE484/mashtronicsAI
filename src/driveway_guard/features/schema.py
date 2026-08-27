from dataclasses import dataclass, field

from driveway_guard.pose.estimator import Keypoint

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(slots=True)
class FrameFeatureVector:
    """A per-frame, per-(person-track, vehicle-track) snapshot. Normalized
    distance fields make this comparable across camera resolutions/positions
    — used directly by the v1 rule-based scorer, and designed to double as
    a training row for a future learned model."""

    # identity / timing
    frame_idx: int
    timestamp_s: float
    person_track_id: int
    vehicle_track_id: int

    # raw positional
    person_bbox_xyxy: BBox
    vehicle_bbox_xyxy: BBox
    person_centroid: Point
    vehicle_centroid: Point

    # proximity / relative motion
    person_vehicle_distance_px: float
    person_vehicle_distance_norm: float
    person_velocity_px_s: Point
    vehicle_velocity_px_s: Point
    approach_speed_px_s: float
    dwell_time_s: float

    # pose-derived (None if pose wasn't run / not gated this frame)
    pose_keypoints: list[Keypoint] | None = None
    pose_mean_confidence: float | None = None
    max_joint_velocity_px_s: float | None = None
    arms_raised_score: float | None = None
    torso_lean_angle_deg: float | None = None

    # second-person / contact
    nearest_other_person_track_id: int | None = None
    nearest_other_person_distance_norm: float | None = None
    person_person_contact: bool = False

    # weapon
    weapon_detected: bool = False
    weapon_confidence: float | None = None
    weapon_bbox_xyxy: BBox | None = None

    # driveway/zone context
    person_in_driveway_zone: bool = False
    vehicle_in_driveway_zone: bool = False

    # extensibility for v2 without a schema migration
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class VehicleConvergenceFeatureVector:
    """Per-vehicle aggregate across all currently-approaching person-tracks
    for that vehicle this frame — multi-directional convergence is a
    cross-person signal, not expressible in the per-pair record above."""

    vehicle_track_id: int
    frame_idx: int
    timestamp_s: float
    approaching_person_track_ids: list[int]
    approaching_person_bearings_deg: list[float]
    angular_spread_deg: float
    num_simultaneous_approachers: int

    # vehicle egress motion while surrounded — a resident trying to flee by
    # driving/backing out of the driveway while people are converging on the
    # car is a materially higher-risk situation than a stationary convergence.
    # None if no calibration (no egress_path to project onto) is available.
    vehicle_velocity_px_s: Point = (0.0, 0.0)
    vehicle_egress_speed_px_s: float | None = None


@dataclass(slots=True)
class BlockingObservation:
    """A vehicle-vehicle relationship: `blocking_vehicle_track_id` is
    obstructing `resident_vehicle_track_id`'s calibrated egress corridor.
    Kept separate from FrameFeatureVector since boxing-in doesn't involve
    a person at all."""

    frame_idx: int
    timestamp_s: float
    resident_vehicle_track_id: int
    blocking_vehicle_track_id: int
    blocking_overlap_ratio: float
    blocking_duration_s: float
    blocking_vehicle_stopped: bool
    person_exited_blocking_vehicle: bool = False
