import math

from driveway_guard.calibration.geometry import blocking_overlap_ratio, point_in_polygon
from driveway_guard.calibration.schema import CalibrationConfig
from driveway_guard.detection.types import TrackedObject
from driveway_guard.features.schema import BlockingObservation, FrameFeatureVector
from driveway_guard.features.track_state import ProximityDwellTracker, TrackHistory
from driveway_guard.pose.estimator import Keypoint

# How close (as a fraction of the frame diagonal) a person must be to a
# vehicle before a FrameFeatureVector is recorded for that pair at all.
# Looser than the pose-gating threshold so approach/sprint/convergence
# signals are captured before someone is already at the vehicle.
_RECORD_PROXIMITY_NORM = 0.5
_CONTACT_NORM = 0.03
_POSE_MIN_CONF = 0.3
_BLOCKING_STOPPED_SPEED_PX_S = 15.0
_EXIT_WINDOW_S = 3.0

# COCO-17 keypoint indices.
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12


class FeatureExtractor:
    def __init__(self, record_proximity_norm: float = _RECORD_PROXIMITY_NORM):
        self._record_proximity_norm = record_proximity_norm
        self._histories: dict[int, TrackHistory] = {}
        self._pair_dwell = ProximityDwellTracker()
        self._blocking_dwell = ProximityDwellTracker()

    def _history(self, track_id: int) -> TrackHistory:
        if track_id not in self._histories:
            self._histories[track_id] = TrackHistory()
        return self._histories[track_id]

    def vehicle_velocity_px_s(self, track_id: int) -> tuple[float, float]:
        """Current velocity for a tracked vehicle, independent of whether
        any person is currently near it — needed to detect a vehicle
        fleeing after everyone who was surrounding it has already gotten
        in (and so no longer shows up as an "approaching person")."""
        return self._history(track_id).velocity_px_s()

    def extract_pairs(
        self,
        frame_idx: int,
        timestamp_s: float,
        frame_diag: float,
        persons: list[TrackedObject],
        vehicles: list[TrackedObject],
        poses: dict[int, list[Keypoint]],
        calibration: CalibrationConfig | None,
    ) -> list[FrameFeatureVector]:
        for p in persons:
            hist = self._history(p.track_id)
            hist.update_position(timestamp_s, p.centroid)
            if p.track_id in poses:
                hist.update_pose(timestamp_s, poses[p.track_id])
        for v in vehicles:
            self._history(v.track_id).update_position(timestamp_s, v.centroid)

        records: list[FrameFeatureVector] = []
        for p in persons:
            for v in vehicles:
                dist_px = _dist(p.centroid, v.centroid)
                dist_norm = dist_px / frame_diag if frame_diag > 0 else dist_px
                if dist_norm > self._record_proximity_norm:
                    self._pair_dwell.update((p.track_id, v.track_id), False, timestamp_s)
                    continue

                p_hist = self._history(p.track_id)
                v_hist = self._history(v.track_id)
                p_vel = p_hist.velocity_px_s()
                v_vel = v_hist.velocity_px_s()
                approach_speed = _approach_speed(p.centroid, v.centroid, p_vel, v_vel)
                dwell = self._pair_dwell.update((p.track_id, v.track_id), True, timestamp_s)

                keypoints = poses.get(p.track_id)
                pose_mean_conf = None
                arms_raised = None
                torso_lean = None
                if keypoints:
                    pose_mean_conf = sum(k[2] for k in keypoints) / len(keypoints)
                    arms_raised = _arms_raised_score(keypoints)
                    torso_lean = _torso_lean_angle(keypoints)
                max_joint_vel = p_hist.max_joint_velocity_px_s()

                nearest_id, nearest_dist_norm = _nearest_other_person(p, persons, frame_diag)
                contact = nearest_dist_norm is not None and nearest_dist_norm <= _CONTACT_NORM

                person_in_zone = False
                vehicle_in_zone = False
                if calibration is not None:
                    person_in_zone = point_in_polygon(
                        p.centroid, calibration.driveway_zone.polygon
                    )
                    vehicle_in_zone = point_in_polygon(
                        v.centroid, calibration.driveway_zone.polygon
                    )

                records.append(
                    FrameFeatureVector(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        person_track_id=p.track_id,
                        vehicle_track_id=v.track_id,
                        person_bbox_xyxy=p.bbox_xyxy,
                        vehicle_bbox_xyxy=v.bbox_xyxy,
                        person_centroid=p.centroid,
                        vehicle_centroid=v.centroid,
                        person_vehicle_distance_px=dist_px,
                        person_vehicle_distance_norm=dist_norm,
                        person_velocity_px_s=p_vel,
                        vehicle_velocity_px_s=v_vel,
                        approach_speed_px_s=approach_speed,
                        dwell_time_s=dwell,
                        pose_keypoints=keypoints,
                        pose_mean_confidence=pose_mean_conf,
                        max_joint_velocity_px_s=max_joint_vel,
                        arms_raised_score=arms_raised,
                        torso_lean_angle_deg=torso_lean,
                        nearest_other_person_track_id=nearest_id,
                        nearest_other_person_distance_norm=nearest_dist_norm,
                        person_person_contact=contact,
                        person_in_driveway_zone=person_in_zone,
                        vehicle_in_driveway_zone=vehicle_in_zone,
                    )
                )
        return records

    def extract_blocking(
        self,
        frame_idx: int,
        timestamp_s: float,
        vehicles: list[TrackedObject],
        persons: list[TrackedObject],
        calibration: CalibrationConfig | None,
    ) -> list[BlockingObservation]:
        """Boxing-in detection. Requires calibration with a
        resident_vehicle_hint zone to identify which vehicle is the
        resident's (v1 simplification — without the hint, this is skipped
        entirely rather than guessing)."""
        if calibration is None or calibration.resident_vehicle_hint is None:
            return []

        resident = None
        for v in vehicles:
            if point_in_polygon(
                v.centroid, calibration.resident_vehicle_hint.typical_start_zone_polygon
            ):
                resident = v
                break
        if resident is None:
            return []

        observations: list[BlockingObservation] = []
        for v in vehicles:
            if v.track_id == resident.track_id:
                continue
            ratio = blocking_overlap_ratio(v.bbox_xyxy, calibration.egress_path)
            active = ratio > 0.0
            duration = self._blocking_dwell.update(("blocking", v.track_id), active, timestamp_s)
            if not active:
                continue

            speed = self._history(v.track_id).speed_px_s()
            stopped = speed < _BLOCKING_STOPPED_SPEED_PX_S

            exited = False
            start = self._blocking_dwell.start_time(("blocking", v.track_id))
            if start is not None:
                for p in persons:
                    first_seen = self._history(p.track_id).first_seen_timestamp
                    if first_seen is None:
                        continue
                    if start <= first_seen <= start + _EXIT_WINDOW_S:
                        if _dist(p.centroid, v.centroid) / max(v.bbox_xyxy[2] - v.bbox_xyxy[0], 1.0) < 3.0:
                            exited = True
                            break

            observations.append(
                BlockingObservation(
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    resident_vehicle_track_id=resident.track_id,
                    blocking_vehicle_track_id=v.track_id,
                    blocking_overlap_ratio=ratio,
                    blocking_duration_s=duration,
                    blocking_vehicle_stopped=stopped,
                    person_exited_blocking_vehicle=exited,
                )
            )
        return observations


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _approach_speed(
    person_centroid: tuple[float, float],
    vehicle_centroid: tuple[float, float],
    p_vel: tuple[float, float],
    v_vel: tuple[float, float],
) -> float:
    dx = vehicle_centroid[0] - person_centroid[0]
    dy = vehicle_centroid[1] - person_centroid[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    ux, uy = dx / dist, dy / dist
    rel_vx, rel_vy = p_vel[0] - v_vel[0], p_vel[1] - v_vel[1]
    return rel_vx * ux + rel_vy * uy


def _nearest_other_person(
    p: TrackedObject, persons: list[TrackedObject], frame_diag: float
) -> tuple[int | None, float | None]:
    nearest_id = None
    nearest_norm = None
    for other in persons:
        if other.track_id == p.track_id:
            continue
        d_norm = _dist(p.centroid, other.centroid) / frame_diag if frame_diag > 0 else 0.0
        if nearest_norm is None or d_norm < nearest_norm:
            nearest_norm = d_norm
            nearest_id = other.track_id
    return nearest_id, nearest_norm


def _torso_scale(keypoints: list[Keypoint]) -> float | None:
    try:
        l_sh, r_sh, l_hip, r_hip = (
            keypoints[_L_SHOULDER],
            keypoints[_R_SHOULDER],
            keypoints[_L_HIP],
            keypoints[_R_HIP],
        )
    except IndexError:
        return None
    if min(l_sh[2], r_sh[2], l_hip[2], r_hip[2]) < _POSE_MIN_CONF:
        return None
    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    scale = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    return scale if scale > 1e-3 else None


def _arms_raised_score(keypoints: list[Keypoint]) -> float | None:
    try:
        l_sh, r_sh, l_wr, r_wr = (
            keypoints[_L_SHOULDER],
            keypoints[_R_SHOULDER],
            keypoints[_L_WRIST],
            keypoints[_R_WRIST],
        )
    except IndexError:
        return None
    if min(l_sh[2], r_sh[2], l_wr[2], r_wr[2]) < _POSE_MIN_CONF:
        return None
    shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
    scale = _torso_scale(keypoints) or 100.0
    l_raise = max(0.0, (shoulder_y - l_wr[1]) / scale)
    r_raise = max(0.0, (shoulder_y - r_wr[1]) / scale)
    return min(1.0, max(l_raise, r_raise))


def _torso_lean_angle(keypoints: list[Keypoint]) -> float | None:
    try:
        l_sh, r_sh, l_hip, r_hip = (
            keypoints[_L_SHOULDER],
            keypoints[_R_SHOULDER],
            keypoints[_L_HIP],
            keypoints[_R_HIP],
        )
    except IndexError:
        return None
    if min(l_sh[2], r_sh[2], l_hip[2], r_hip[2]) < _POSE_MIN_CONF:
        return None
    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return math.degrees(math.atan2(dx, -dy))
