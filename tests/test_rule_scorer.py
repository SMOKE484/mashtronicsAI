from driveway_guard.features.schema import (
    BlockingObservation,
    FrameFeatureVector,
    VehicleConvergenceFeatureVector,
)
from driveway_guard.scoring.rules import (
    RuleBasedScorer,
    RuleThresholds,
    score_boxing_in,
    score_convergence,
    score_sprint,
    score_struggle,
    score_weapon,
)


def _make_record(**overrides) -> FrameFeatureVector:
    defaults = dict(
        frame_idx=0,
        timestamp_s=0.0,
        person_track_id=1,
        vehicle_track_id=2,
        person_bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        vehicle_bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        person_centroid=(5.0, 5.0),
        vehicle_centroid=(5.0, 5.0),
        person_vehicle_distance_px=0.0,
        person_vehicle_distance_norm=0.0,
        person_velocity_px_s=(0.0, 0.0),
        vehicle_velocity_px_s=(0.0, 0.0),
        approach_speed_px_s=0.0,
        dwell_time_s=0.0,
    )
    defaults.update(overrides)
    return FrameFeatureVector(**defaults)


def test_struggle_zero_when_far_from_vehicle():
    t = RuleThresholds()
    record = _make_record(person_vehicle_distance_norm=0.5, dwell_time_s=5.0)
    assert score_struggle(record, t) == 0.0


def test_struggle_zero_when_dwell_too_short():
    t = RuleThresholds()
    record = _make_record(
        person_vehicle_distance_norm=0.01, dwell_time_s=0.1, max_joint_velocity_px_s=5000.0
    )
    assert score_struggle(record, t) == 0.0


def test_struggle_high_when_all_signals_present():
    t = RuleThresholds()
    record = _make_record(
        person_vehicle_distance_norm=0.01,
        dwell_time_s=5.0,
        max_joint_velocity_px_s=2000.0,
        arms_raised_score=0.9,
        person_person_contact=True,
    )
    score = score_struggle(record, t)
    assert score > 0.9


def test_struggle_partial_from_joint_velocity_only():
    t = RuleThresholds()
    record = _make_record(
        person_vehicle_distance_norm=0.01,
        dwell_time_s=5.0,
        max_joint_velocity_px_s=t.joint_velocity_struggle_px_s * 2,
    )
    score = score_struggle(record, t)
    assert 0.0 < score <= 0.4


def _make_blocking(**overrides) -> BlockingObservation:
    defaults = dict(
        frame_idx=0,
        timestamp_s=0.0,
        resident_vehicle_track_id=1,
        blocking_vehicle_track_id=2,
        blocking_overlap_ratio=1.0,
        blocking_duration_s=10.0,
        blocking_vehicle_stopped=True,
    )
    defaults.update(overrides)
    return BlockingObservation(**defaults)


def test_boxing_in_zero_when_overlap_below_threshold():
    t = RuleThresholds()
    obs = _make_blocking(blocking_overlap_ratio=0.1)
    assert score_boxing_in(obs, t) == 0.0


def test_boxing_in_zero_when_still_moving():
    t = RuleThresholds()
    obs = _make_blocking(blocking_vehicle_stopped=False)
    assert score_boxing_in(obs, t) == 0.0


def test_boxing_in_zero_when_duration_too_short():
    t = RuleThresholds()
    obs = _make_blocking(blocking_duration_s=0.5)
    assert score_boxing_in(obs, t) == 0.0


def test_boxing_in_positive_and_boosted_by_exit():
    t = RuleThresholds()
    obs_no_exit = _make_blocking(blocking_duration_s=t.blocking_duration_min_s * 3)
    obs_exit = _make_blocking(
        blocking_duration_s=t.blocking_duration_min_s * 3, person_exited_blocking_vehicle=True
    )
    assert score_boxing_in(obs_no_exit, t) > 0.0
    assert score_boxing_in(obs_exit, t) >= score_boxing_in(obs_no_exit, t)


def test_weapon_zero_when_not_detected():
    t = RuleThresholds()
    record = _make_record(weapon_detected=False)
    assert score_weapon(record, t) == 0.0


def test_weapon_zero_when_below_confidence_threshold():
    t = RuleThresholds()
    record = _make_record(weapon_detected=True, weapon_confidence=0.1)
    assert score_weapon(record, t) == 0.0


def test_weapon_positive_when_confident():
    t = RuleThresholds()
    record = _make_record(weapon_detected=True, weapon_confidence=0.9)
    assert score_weapon(record, t) == 0.9


def test_sprint_zero_below_threshold():
    t = RuleThresholds()
    record = _make_record(approach_speed_px_s=10.0)
    assert score_sprint(record, t) == 0.0


def test_sprint_positive_above_threshold():
    t = RuleThresholds()
    record = _make_record(approach_speed_px_s=t.sprint_speed_px_s_threshold * 2)
    assert score_sprint(record, t) > 0.0


def _make_convergence(**overrides) -> VehicleConvergenceFeatureVector:
    defaults = dict(
        vehicle_track_id=1,
        frame_idx=0,
        timestamp_s=0.0,
        approaching_person_track_ids=[2, 3],
        approaching_person_bearings_deg=[0.0, 180.0],
        angular_spread_deg=180.0,
        num_simultaneous_approachers=2,
    )
    defaults.update(overrides)
    return VehicleConvergenceFeatureVector(**defaults)


def test_convergence_zero_with_single_approacher():
    t = RuleThresholds()
    conv = _make_convergence(num_simultaneous_approachers=1, approaching_person_track_ids=[2])
    assert score_convergence(conv, t) == 0.0


def test_convergence_zero_when_spread_too_narrow():
    t = RuleThresholds()
    conv = _make_convergence(angular_spread_deg=10.0)
    assert score_convergence(conv, t) == 0.0


def test_convergence_positive_when_wide_spread_multiple_people():
    t = RuleThresholds()
    conv = _make_convergence(angular_spread_deg=180.0, num_simultaneous_approachers=2)
    assert score_convergence(conv, t) > 0.0


def test_convergence_boosted_when_vehicle_fleeing_along_egress():
    t = RuleThresholds()
    stationary = _make_convergence(angular_spread_deg=100.0)
    fleeing = _make_convergence(
        angular_spread_deg=100.0,
        vehicle_egress_speed_px_s=t.convergence_egress_speed_px_s_threshold * 2,
    )
    assert score_convergence(fleeing, t) > score_convergence(stationary, t)


def test_convergence_no_fleeing_bonus_without_calibration():
    t = RuleThresholds()
    conv = _make_convergence(angular_spread_deg=100.0)
    assert conv.vehicle_egress_speed_px_s is None
    assert score_convergence(conv, t) == 0.5


def test_convergence_recently_surrounded_and_fleeing_scores_without_current_approachers():
    # Nobody's actively converging this frame (e.g. they got in the car),
    # but the vehicle was surrounded a moment ago and is now driving off.
    t = RuleThresholds()
    conv = _make_convergence(
        num_simultaneous_approachers=0,
        approaching_person_track_ids=[],
        approaching_person_bearings_deg=[],
        angular_spread_deg=0.0,
        vehicle_egress_speed_px_s=t.convergence_egress_speed_px_s_threshold * 2,
    )
    assert score_convergence(conv, t, recently_surrounded=True) == t.convergence_recent_fleeing_score


def test_convergence_zero_when_fleeing_but_not_recently_surrounded():
    t = RuleThresholds()
    conv = _make_convergence(
        num_simultaneous_approachers=0,
        approaching_person_track_ids=[],
        approaching_person_bearings_deg=[],
        angular_spread_deg=0.0,
        vehicle_egress_speed_px_s=t.convergence_egress_speed_px_s_threshold * 2,
    )
    assert score_convergence(conv, t, recently_surrounded=False) == 0.0


def test_scorer_flags_vehicle_driven_off_shortly_after_being_surrounded():
    """End-to-end regression for the "hijackers get in and reverse the car,
    driver possibly still inside" scenario: the convergence signal itself
    disappears once nobody is left "approaching" (they've gotten in), so
    this needs the scorer's cross-frame memory, not just a single-frame
    score, to still flag the vehicle once it drives off."""
    t = RuleThresholds()
    scorer = RuleBasedScorer(t)

    surrounded = _make_convergence(
        vehicle_track_id=1, angular_spread_deg=180.0, num_simultaneous_approachers=2
    )
    assert scorer.process_frame(0, 0.0, [], [], [surrounded]) == []

    # Everyone's gotten in — no approachers, vehicle not moving yet.
    got_in = _make_convergence(
        vehicle_track_id=1,
        approaching_person_track_ids=[],
        approaching_person_bearings_deg=[],
        angular_spread_deg=0.0,
        num_simultaneous_approachers=0,
    )
    assert scorer.process_frame(1, 0.5, [], [], [got_in]) == []

    # Now it's moving off along the egress path — two consecutive samples
    # needed past event_min_duration_s before it actually flags.
    fleeing = _make_convergence(
        vehicle_track_id=1,
        approaching_person_track_ids=[],
        approaching_person_bearings_deg=[],
        angular_spread_deg=0.0,
        num_simultaneous_approachers=0,
        vehicle_egress_speed_px_s=t.convergence_egress_speed_px_s_threshold * 2,
    )
    assert scorer.process_frame(2, 1.0, [], [], [fleeing]) == []
    events = scorer.process_frame(3, 1.5, [], [], [fleeing])
    assert len(events) == 1
    assert events[0].event_type == "multi_directional_convergence"
