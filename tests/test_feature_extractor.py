import math

from driveway_guard.detection.types import ObjectClass, TrackedObject
from driveway_guard.features.extractor import (
    FeatureExtractor,
    _approach_speed,
    _arms_raised_score,
    _dist,
    _nearest_other_person,
    _torso_lean_angle,
)


def test_dist_basic():
    assert abs(_dist((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-9


def test_approach_speed_positive_when_closing():
    # Person at origin moving toward a stationary vehicle at (100, 0).
    speed = _approach_speed((0.0, 0.0), (100.0, 0.0), (50.0, 0.0), (0.0, 0.0))
    assert abs(speed - 50.0) < 1e-6


def test_approach_speed_negative_when_moving_away():
    speed = _approach_speed((0.0, 0.0), (100.0, 0.0), (-50.0, 0.0), (0.0, 0.0))
    assert speed < 0


def test_approach_speed_accounts_for_vehicle_motion():
    # Person and vehicle moving at the same velocity -> zero closing speed.
    speed = _approach_speed((0.0, 0.0), (100.0, 0.0), (30.0, 0.0), (30.0, 0.0))
    assert abs(speed) < 1e-6


# COCO indices used by the helpers: 5/6 shoulders, 9/10 wrists.
def _person_kpts(overrides: dict) -> list[tuple[float, float, float]]:
    kpts = [(0.0, 0.0, 0.9)] * 17
    kpts = list(kpts)
    for idx, val in overrides.items():
        kpts[idx] = val
    return kpts


def test_arms_raised_score_high_when_wrists_above_shoulders():
    kpts = _person_kpts(
        {
            5: (0.0, 100.0, 0.9),  # left shoulder
            6: (20.0, 100.0, 0.9),  # right shoulder
            9: (0.0, 20.0, 0.9),  # left wrist, well above shoulder
            10: (20.0, 100.0, 0.9),  # right wrist, at shoulder height
            11: (0.0, 150.0, 0.9),  # left hip
            12: (20.0, 150.0, 0.9),  # right hip
        }
    )
    score = _arms_raised_score(kpts)
    assert score is not None and score > 0.5


def test_arms_raised_score_low_when_arms_down():
    kpts = _person_kpts(
        {
            5: (0.0, 100.0, 0.9),
            6: (20.0, 100.0, 0.9),
            9: (0.0, 180.0, 0.9),  # wrists below shoulders
            10: (20.0, 180.0, 0.9),
            11: (0.0, 150.0, 0.9),
            12: (20.0, 150.0, 0.9),
        }
    )
    score = _arms_raised_score(kpts)
    assert score == 0.0


def test_arms_raised_score_none_on_low_confidence():
    kpts = _person_kpts({9: (0.0, 20.0, 0.05)})
    assert _arms_raised_score(kpts) is None


def test_torso_lean_angle_near_zero_when_upright():
    kpts = _person_kpts(
        {
            5: (0.0, 100.0, 0.9),
            6: (20.0, 100.0, 0.9),
            11: (0.0, 150.0, 0.9),
            12: (20.0, 150.0, 0.9),
        }
    )
    angle = _torso_lean_angle(kpts)
    assert angle is not None and abs(angle) < 1e-6


def test_nearest_other_person_finds_closest():
    p = TrackedObject(track_id=1, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(0, 0, 10, 10))
    close = TrackedObject(
        track_id=2, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(15, 5, 25, 15)
    )
    far = TrackedObject(
        track_id=3, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(200, 5, 210, 15)
    )
    nearest_id, nearest_norm = _nearest_other_person(p, [p, close, far], frame_diag=100.0)
    assert nearest_id == 2
    assert nearest_norm is not None


def test_extract_pairs_records_only_within_proximity():
    extractor = FeatureExtractor(record_proximity_norm=0.2)
    person_near = TrackedObject(
        track_id=1, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(0, 0, 10, 10)
    )
    person_far = TrackedObject(
        track_id=2, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(900, 900, 910, 910)
    )
    vehicle = TrackedObject(
        track_id=3, cls=ObjectClass.VEHICLE, confidence=0.9, bbox_xyxy=(0, 0, 20, 20)
    )
    records = extractor.extract_pairs(
        frame_idx=0,
        timestamp_s=0.0,
        frame_diag=1000.0,
        persons=[person_near, person_far],
        vehicles=[vehicle],
        poses={},
        calibration=None,
    )
    person_ids = {r.person_track_id for r in records}
    assert 1 in person_ids
    assert 2 not in person_ids


def test_extract_pairs_dwell_time_accumulates_across_frames():
    extractor = FeatureExtractor(record_proximity_norm=0.5)
    person = TrackedObject(
        track_id=1, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=(0, 0, 10, 10)
    )
    vehicle = TrackedObject(
        track_id=2, cls=ObjectClass.VEHICLE, confidence=0.9, bbox_xyxy=(5, 5, 15, 15)
    )
    r1 = extractor.extract_pairs(0, 0.0, 1000.0, [person], [vehicle], {}, None)[0]
    r2 = extractor.extract_pairs(1, 2.0, 1000.0, [person], [vehicle], {}, None)[0]
    assert r1.dwell_time_s == 0.0
    assert abs(r2.dwell_time_s - 2.0) < 1e-6
