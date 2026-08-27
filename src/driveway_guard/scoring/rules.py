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
    # bonus when the vehicle is also moving in the egress direction (i.e.
    # driving/backing out of the driveway) while surrounded — a resident
    # attempting to flee under duress, not just a stationary convergence.
    # Requires calibration (egress_path); no-op without it.
    convergence_egress_speed_px_s_threshold: float = 60.0
    convergence_fleeing_bonus: float = 0.35
    # If the vehicle was surrounded recently (within this window) and is
    # now moving along the egress direction, flag it even though nobody is
    # actively "approaching" it any more — covers the case where the people
    # who converged on it got in (as a driver/passenger/hostage) rather than
    # staying visible outside, which would otherwise never re-qualify under
    # the plain convergence gate above. Deliberately scored high on its own
    # (not just a small bonus): a vehicle driven off shortly after being
    # surrounded is a strong, largely self-explaining signal regardless of
    # whether the weapon/struggle detectors also caught something.
    convergence_recent_window_s: float = 20.0
    convergence_recent_fleeing_score: float = 0.85

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


def score_convergence(
    conv: VehicleConvergenceFeatureVector,
    t: RuleThresholds,
    recently_surrounded: bool = False,
) -> float:
    fleeing = (
        conv.vehicle_egress_speed_px_s is not None
        and conv.vehicle_egress_speed_px_s >= t.convergence_egress_speed_px_s_threshold
    )
    qualifies_now = (
        conv.num_simultaneous_approachers >= t.convergence_min_approachers
        and conv.angular_spread_deg >= t.convergence_angle_threshold_deg
    )

    if qualifies_now:
        remaining = max(1.0, 180.0 - t.convergence_angle_threshold_deg)
        base = max(0.5, _soft(conv.angular_spread_deg, t.convergence_angle_threshold_deg, remaining))
        bonus = t.convergence_fleeing_bonus if fleeing else 0.0
        return min(1.0, base + bonus)

    if recently_surrounded and fleeing:
        # Nobody's actively converging on the vehicle this frame — they may
        # have gotten in — but it was surrounded a moment ago and is now
        # driving off along the egress path. Flag it on that basis alone.
        return t.convergence_recent_fleeing_score

    return 0.0


class RuleBasedScorer(RiskScorer):
    def __init__(self, thresholds: RuleThresholds | None = None):
        self._t = thresholds or RuleThresholds()
        self._events = EventAggregator()
        self._last_qualifying_convergence: dict[int, float] = {}

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
            qualifies_now = (
                conv.num_simultaneous_approachers >= t.convergence_min_approachers
                and conv.angular_spread_deg >= t.convergence_angle_threshold_deg
            )
            if qualifies_now:
                self._last_qualifying_convergence[conv.vehicle_track_id] = timestamp_s
            last_qualifying_ts = self._last_qualifying_convergence.get(conv.vehicle_track_id)
            recently_surrounded = (
                last_qualifying_ts is not None
                and timestamp_s - last_qualifying_ts <= t.convergence_recent_window_s
            )
            score = score_convergence(conv, t, recently_surrounded)
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
