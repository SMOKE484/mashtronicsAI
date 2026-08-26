from dataclasses import dataclass

from driveway_guard.features.schema import (
    BlockingObservation,
    FrameFeatureVector,
    VehicleConvergenceFeatureVector,
)
from driveway_guard.scoring.base import RiskScorer
from driveway_guard.scoring.events import EventAggregator, FlaggedEvent


@dataclass
class RuleThresholds:
    """All tunable values in one place — these are starting guesses, meant
    to be tuned once real/sourced clips exist (see README)."""

    # struggle / aggressive-contact
    proximity_norm_threshold: float = 0.08
    struggle_dwell_min_s: float = 1.5
    joint_velocity_struggle_px_s: float = 900.0
    contact_struggle_min_s: float = 1.0
    arms_raised_score_threshold: float = 0.6

    # boxing-in / blocking egress
    blocking_overlap_ratio_threshold: float = 0.5
    blocking_duration_min_s: float = 4.0
    blocking_with_exit_bonus: float = 0.3

    # weapon at window
    weapon_confidence_threshold: float = 0.5

    # sprint approach
    sprint_speed_px_s_threshold: float = 650.0

    # multi-directional convergence
    convergence_angle_threshold_deg: float = 90.0
    convergence_min_approachers: int = 2

    # scoring / event debounce
    risk_score_flag_threshold: float = 0.65
    event_min_duration_s: float = 0.3
    weapon_min_duration_s: float = 0.5
    event_cooldown_s: float = 5.0


def _soft(value: float, threshold: float, scale: float) -> float:
    """0 at/below threshold, ramping to 1 by threshold + scale."""
    if scale <= 0:
        return 1.0 if value >= threshold else 0.0
    return max(0.0, min(1.0, (value - threshold) / scale))


def score_struggle(record: FrameFeatureVector, t: RuleThresholds) -> float:
    if record.person_vehicle_distance_norm > t.proximity_norm_threshold:
        return 0.0
    if record.dwell_time_s < t.struggle_dwell_min_s:
        return 0.0

    joint_term = 0.0
    if record.max_joint_velocity_px_s is not None:
        joint_term = _soft(
            record.max_joint_velocity_px_s,
            t.joint_velocity_struggle_px_s,
            t.joint_velocity_struggle_px_s,
        )

    arms_term = 0.0
    if record.arms_raised_score is not None:
        arms_term = _soft(record.arms_raised_score, t.arms_raised_score_threshold, 0.4)

    contact_term = 0.0
    if record.person_person_contact and record.dwell_time_s >= t.contact_struggle_min_s:
        contact_term = 1.0

    return min(1.0, 0.4 * joint_term + 0.3 * arms_term + 0.3 * contact_term)


def score_boxing_in(obs: BlockingObservation, t: RuleThresholds) -> float:
    if obs.blocking_overlap_ratio < t.blocking_overlap_ratio_threshold:
        return 0.0
    if not obs.blocking_vehicle_stopped:
        return 0.0
    if obs.blocking_duration_s < t.blocking_duration_min_s:
        return 0.0
    base = _soft(obs.blocking_duration_s, t.blocking_duration_min_s, t.blocking_duration_min_s)
    bonus = t.blocking_with_exit_bonus if obs.person_exited_blocking_vehicle else 0.0
    return min(1.0, base + bonus)


def score_weapon(record: FrameFeatureVector, t: RuleThresholds) -> float:
    if not record.weapon_detected or record.weapon_confidence is None:
        return 0.0
    if record.weapon_confidence < t.weapon_confidence_threshold:
        return 0.0
    return record.weapon_confidence


def score_sprint(record: FrameFeatureVector, t: RuleThresholds) -> float:
    if record.approach_speed_px_s < t.sprint_speed_px_s_threshold:
        return 0.0
    return max(
        0.5, _soft(record.approach_speed_px_s, t.sprint_speed_px_s_threshold, t.sprint_speed_px_s_threshold)
    )


def score_convergence(conv: VehicleConvergenceFeatureVector, t: RuleThresholds) -> float:
    if conv.num_simultaneous_approachers < t.convergence_min_approachers:
        return 0.0
    if conv.angular_spread_deg < t.convergence_angle_threshold_deg:
        return 0.0
    remaining = max(1.0, 180.0 - t.convergence_angle_threshold_deg)
    return max(0.5, _soft(conv.angular_spread_deg, t.convergence_angle_threshold_deg, remaining))


class RuleBasedScorer(RiskScorer):
    def __init__(self, thresholds: RuleThresholds | None = None):
        self._t = thresholds or RuleThresholds()
        self._events = EventAggregator()

    def process_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        pair_records: list[FrameFeatureVector],
        blocking_observations: list[BlockingObservation],
        convergence_records: list[VehicleConvergenceFeatureVector],
    ) -> list[FlaggedEvent]:
        t = self._t
        flagged: list[FlaggedEvent] = []

        for r in pair_records:
            struggle_score = score_struggle(r, t)
            event = self._events.update(
                "struggle",
                (r.person_track_id, r.vehicle_track_id),
                struggle_score,
                t.risk_score_flag_threshold,
                t.event_min_duration_s,
                t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [r.person_track_id, r.vehicle_track_id],
            )
            if event:
                flagged.append(event)

            weapon_score = score_weapon(r, t)
            event = self._events.update(
                "weapon_at_window",
                (r.person_track_id, r.vehicle_track_id),
                weapon_score,
                t.risk_score_flag_threshold,
                t.weapon_min_duration_s,
                t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [r.person_track_id, r.vehicle_track_id],
            )
            if event:
                flagged.append(event)

            sprint_score = score_sprint(r, t)
            event = self._events.update(
                "sprint_approach",
                (r.person_track_id, r.vehicle_track_id),
                sprint_score,
                t.risk_score_flag_threshold,
                t.event_min_duration_s,
                t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [r.person_track_id, r.vehicle_track_id],
            )
            if event:
                flagged.append(event)

        for obs in blocking_observations:
            score = score_boxing_in(obs, t)
            event = self._events.update(
                "boxing_in",
                (obs.resident_vehicle_track_id, obs.blocking_vehicle_track_id),
                score,
                t.risk_score_flag_threshold,
                t.event_min_duration_s,
                t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [obs.resident_vehicle_track_id, obs.blocking_vehicle_track_id],
            )
            if event:
                flagged.append(event)

        for conv in convergence_records:
            score = score_convergence(conv, t)
            event = self._events.update(
                "multi_directional_convergence",
                (conv.vehicle_track_id,),
                score,
                t.risk_score_flag_threshold,
                t.event_min_duration_s,
                t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [conv.vehicle_track_id, *conv.approaching_person_track_ids],
            )
            if event:
                flagged.append(event)

        return flagged
