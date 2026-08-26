from driveway_guard.features.track_state import ProximityDwellTracker, TrackHistory


def test_track_history_velocity_from_two_positions():
    hist = TrackHistory()
    hist.update_position(0.0, (0.0, 0.0))
    hist.update_position(1.0, (10.0, 0.0))
    vx, vy = hist.velocity_px_s()
    assert abs(vx - 10.0) < 1e-6
    assert abs(vy - 0.0) < 1e-6


def test_track_history_velocity_zero_with_single_sample():
    hist = TrackHistory()
    hist.update_position(0.0, (5.0, 5.0))
    assert hist.velocity_px_s() == (0.0, 0.0)


def test_track_history_first_seen_timestamp_set_once():
    hist = TrackHistory()
    assert hist.first_seen_timestamp is None
    hist.update_position(2.0, (0.0, 0.0))
    assert hist.first_seen_timestamp == 2.0
    hist.update_position(3.0, (1.0, 1.0))
    assert hist.first_seen_timestamp == 2.0


def test_track_history_max_joint_velocity_none_without_two_pose_samples():
    hist = TrackHistory()
    assert hist.max_joint_velocity_px_s() is None
    hist.update_pose(0.0, [(0.0, 0.0, 0.9)])
    assert hist.max_joint_velocity_px_s() is None


def test_track_history_max_joint_velocity_ignores_low_confidence_points():
    hist = TrackHistory()
    hist.update_pose(0.0, [(0.0, 0.0, 0.9), (0.0, 0.0, 0.1)])
    hist.update_pose(1.0, [(100.0, 0.0, 0.9), (500.0, 0.0, 0.1)])
    # second keypoint has low confidence both frames -> ignored, so max
    # velocity should come from the first keypoint (100 px/s), not the
    # (much larger) low-confidence jump.
    assert abs(hist.max_joint_velocity_px_s() - 100.0) < 1e-6


def test_proximity_dwell_tracker_accumulates_while_active():
    tracker = ProximityDwellTracker()
    assert tracker.update(("a", "b"), True, 0.0) == 0.0
    assert tracker.update(("a", "b"), True, 1.5) == 1.5
    assert tracker.update(("a", "b"), True, 3.0) == 3.0


def test_proximity_dwell_tracker_resets_when_inactive():
    tracker = ProximityDwellTracker()
    tracker.update(("a", "b"), True, 0.0)
    tracker.update(("a", "b"), True, 2.0)
    assert tracker.update(("a", "b"), False, 2.5) == 0.0
    # dwell restarts from zero on the next active sample
    assert tracker.update(("a", "b"), True, 3.0) == 0.0
