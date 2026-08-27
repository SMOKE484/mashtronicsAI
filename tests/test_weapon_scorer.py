from driveway_guard.detection.types import ObjectClass, TrackedObject
from driveway_guard.scoring.weapon import WeaponScorer, WeaponThresholds, score_weapon_hit

_FRAME_DIAG = 1000.0
_PROXIMITY_NORM = 0.5  # generous; these tests aren't about the gating distance


def _person(track_id, bbox=(100, 100, 200, 300)):
    return TrackedObject(track_id=track_id, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=bbox)


def _vehicle(track_id=1, bbox=(50, 50, 400, 400)):
    return TrackedObject(track_id=track_id, cls=ObjectClass.VEHICLE, confidence=0.9, bbox_xyxy=bbox)


def test_score_weapon_hit_zero_when_none():
    assert score_weapon_hit(None, 0.5) == 0.0


def test_score_weapon_hit_zero_below_threshold():
    assert score_weapon_hit(0.3, 0.5) == 0.0


def test_score_weapon_hit_passthrough_above_threshold():
    assert score_weapon_hit(0.9, 0.5) == 0.9


def test_event_fires_only_once_duration_met():
    scorer = WeaponScorer(WeaponThresholds(weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5))
    vehicle = _vehicle()
    person = _person(track_id=10)
    hits = {10: (0.8, (0.0, 0.0, 1.0, 1.0))}

    events = scorer.process_frame(0, 0.0, [person], [vehicle], hits, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # Still under weapon_min_duration_s (0.5s).
    events = scorer.process_frame(1, 0.2, [person], [vehicle], hits, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # Duration (0.6s) now clears weapon_min_duration_s.
    events = scorer.process_frame(2, 0.6, [person], [vehicle], hits, _FRAME_DIAG, _PROXIMITY_NORM)
    assert len(events) == 1
    assert events[0].event_type == "weapon_at_window"
    assert events[0].track_ids[0] == vehicle.track_id


def test_no_event_when_hit_never_clears_confidence_threshold():
    scorer = WeaponScorer(WeaponThresholds(weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5))
    vehicle = _vehicle()
    person = _person(track_id=10)
    low_conf_hits = {10: (0.3, (0.0, 0.0, 1.0, 1.0))}

    for frame_idx, t in enumerate([0.0, 0.2, 0.6, 1.0]):
        events = scorer.process_frame(
            frame_idx, t, [person], [vehicle], low_conf_hits, _FRAME_DIAG, _PROXIMITY_NORM
        )
        assert events == []


def test_brief_dip_below_threshold_does_not_reset_duration():
    """A single below-threshold frame inside an otherwise-sustained
    detection must not wipe out the accumulated duration -- real per-frame
    confidence flickers (see HANDOVER.md's real-footage validation), and a
    strict zero-tolerance reset made genuine ~1s detections with one dipped
    frame never accumulate enough duration to fire."""
    scorer = WeaponScorer(
        WeaponThresholds(weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5, weapon_max_gap_s=0.15)
    )
    vehicle = _vehicle()
    person = _person(track_id=10)
    high_conf = {10: (0.8, (0.0, 0.0, 1.0, 1.0))}
    low_conf = {10: (0.3, (0.0, 0.0, 1.0, 1.0))}

    events = scorer.process_frame(0, 0.0, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # One dipped frame, gap (0.1s) within weapon_max_gap_s -- must not reset.
    events = scorer.process_frame(1, 0.1, [person], [vehicle], low_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    events = scorer.process_frame(2, 0.2, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # Duration since the original start (0.6s) now clears weapon_min_duration_s.
    events = scorer.process_frame(3, 0.6, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert len(events) == 1
    assert events[0].event_type == "weapon_at_window"


def test_gap_longer_than_max_gap_s_still_resets():
    scorer = WeaponScorer(
        WeaponThresholds(weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5, weapon_max_gap_s=0.15)
    )
    vehicle = _vehicle()
    person = _person(track_id=10)
    high_conf = {10: (0.8, (0.0, 0.0, 1.0, 1.0))}
    low_conf = {10: (0.3, (0.0, 0.0, 1.0, 1.0))}

    events = scorer.process_frame(0, 0.0, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # Gap (0.3s) exceeds weapon_max_gap_s (0.15s) -- must reset.
    events = scorer.process_frame(1, 0.3, [person], [vehicle], low_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    events = scorer.process_frame(2, 0.4, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []

    # Only 0.3s since the reset at frame 1 -- still short of weapon_min_duration_s (0.5s).
    events = scorer.process_frame(3, 0.7, [person], [vehicle], high_conf, _FRAME_DIAG, _PROXIMITY_NORM)
    assert events == []


def test_event_survives_person_track_id_churn_because_keyed_by_vehicle():
    """Regression test for the vehicle-only debounce keying decision. Real
    footage (video3.mp4) showed weapon hits landing on 5 different person
    track IDs within a ~3.5s span as the tracker lost and reacquired the
    same person -- switching person track IDs must not reset the debounce
    clock, since the key is the vehicle alone."""
    scorer = WeaponScorer(WeaponThresholds(weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5))
    vehicle = _vehicle()

    events = scorer.process_frame(
        0, 0.0, [_person(31)], [vehicle], {31: (0.8, (0.0, 0.0, 1.0, 1.0))}, _FRAME_DIAG, _PROXIMITY_NORM
    )
    assert events == []

    events = scorer.process_frame(
        1, 0.2, [_person(35)], [vehicle], {35: (0.8, (0.0, 0.0, 1.0, 1.0))}, _FRAME_DIAG, _PROXIMITY_NORM
    )
    assert events == []

    events = scorer.process_frame(
        2, 0.6, [_person(37)], [vehicle], {37: (0.8, (0.0, 0.0, 1.0, 1.0))}, _FRAME_DIAG, _PROXIMITY_NORM
    )
    assert len(events) == 1
    assert events[0].track_ids[0] == vehicle.track_id
