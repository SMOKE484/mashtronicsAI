import math
from collections import deque

from driveway_guard.pose.estimator import Keypoint

_HISTORY_SECONDS = 2.0
_MAX_HISTORY_SAMPLES = 120  # generous cap regardless of fps


class TrackHistory:
    """Rolling recent history for a single tracked object (person or
    vehicle), used to derive velocity, dwell time, and joint velocity."""

    def __init__(self):
        self._positions: deque[tuple[float, tuple[float, float]]] = deque(
            maxlen=_MAX_HISTORY_SAMPLES
        )
        self._poses: deque[tuple[float, list[Keypoint]]] = deque(maxlen=_MAX_HISTORY_SAMPLES)
        self.first_seen_timestamp: float | None = None

    def update_position(self, timestamp_s: float, centroid: tuple[float, float]) -> None:
        if self.first_seen_timestamp is None:
            self.first_seen_timestamp = timestamp_s
        self._positions.append((timestamp_s, centroid))
        self._prune(self._positions, timestamp_s)

    def update_pose(self, timestamp_s: float, keypoints: list[Keypoint]) -> None:
        self._poses.append((timestamp_s, keypoints))
        self._prune(self._poses, timestamp_s)

    @staticmethod
    def _prune(buf: deque, now: float) -> None:
        while buf and now - buf[0][0] > _HISTORY_SECONDS:
            buf.popleft()

    def velocity_px_s(self) -> tuple[float, float]:
        if len(self._positions) < 2:
            return (0.0, 0.0)
        t0, p0 = self._positions[-2]
        t1, p1 = self._positions[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    def speed_px_s(self) -> float:
        vx, vy = self.velocity_px_s()
        return math.hypot(vx, vy)

    def max_joint_velocity_px_s(self) -> float | None:
        if len(self._poses) < 2:
            return None
        t0, kpts0 = self._poses[-2]
        t1, kpts1 = self._poses[-1]
        dt = t1 - t0
        if dt <= 0 or len(kpts0) != len(kpts1):
            return None
        max_v = 0.0
        for (x0, y0, c0), (x1, y1, c1) in zip(kpts0, kpts1):
            if c0 < 0.3 or c1 < 0.3:
                continue
            v = math.hypot(x1 - x0, y1 - y0) / dt
            max_v = max(max_v, v)
        return max_v


class ProximityDwellTracker:
    """Tracks how long a (person, vehicle) pair has been continuously
    within proximity, keyed by pair id. Also usable for single-key
    dwell tracking (e.g. a blocking vehicle's stopped duration)."""

    def __init__(self):
        self._start_times: dict[tuple, float] = {}

    def update(self, key: tuple, active: bool, now: float) -> float:
        if not active:
            self._start_times.pop(key, None)
            return 0.0
        if key not in self._start_times:
            self._start_times[key] = now
        return now - self._start_times[key]

    def start_time(self, key: tuple) -> float | None:
        return self._start_times.get(key)
