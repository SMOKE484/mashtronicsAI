import math
from dataclasses import dataclass

from driveway_guard.detection.types import TrackedObject
from driveway_guard.scoring.events import EventAggregator, FlaggedEvent


@dataclass
class WeaponThresholds:
    """Starting guesses, meant to be tuned against real footage -- see
    HANDOVER.md/plan "Validation against real footage"."""

    weapon_confidence_threshold: float = 0.5
    weapon_min_duration_s: float = 0.5
    event_cooldown_s: float = 5.0
    weapon_max_gap_s: float = 0.15


def score_weapon_hit(confidence: float | None, threshold: float) -> float:
    if confidence is None or confidence < threshold:
        return 0.0
    return confidence


def _nearest_gated_vehicle(
    person: TrackedObject,
    vehicles: list[TrackedObject],
    frame_diag: float,
    proximity_norm: float,
) -> TrackedObject | None:
    if frame_diag <= 0:
        return None
    best: TrackedObject | None = None
    best_dist_norm = math.inf
    for vehicle in vehicles:
        dist = math.hypot(
            person.centroid[0] - vehicle.centroid[0], person.centroid[1] - vehicle.centroid[1]
        )
        dist_norm = dist / frame_diag
        if dist_norm <= proximity_norm and dist_norm < best_dist_norm:
            best = vehicle
            best_dist_norm = dist_norm
    return best


class WeaponScorer:
    """Debounces per-vehicle weapon detections into FlaggedEvents.

    Keyed by vehicle track ID alone, not (person, vehicle) -- real footage
    (video3.mp4) showed weapon hits landing on 5 different person track IDs
    within a ~3.5s span as the tracker lost and reacquired the same person,
    which would reset a (person, vehicle)-keyed debounce clock on every
    switch. A weapon threat is fundamentally about the vehicle being at
    risk, not which person track happens to be holding it, so keying by
    vehicle alone survives that churn. Contributing person track IDs are
    still recorded on the emitted event for review.
    """

    def __init__(self, thresholds: WeaponThresholds | None = None):
        self._t = thresholds or WeaponThresholds()
        self._events = EventAggregator()

    def process_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        persons: list[TrackedObject],
        vehicles: list[TrackedObject],
        weapon_hits: dict[int, tuple[float, tuple[float, float, float, float]]],
        frame_diag: float,
        proximity_norm: float,
    ) -> list[FlaggedEvent]:
        hits_by_vehicle: dict[int, list[tuple[float, int]]] = {}
        for person in persons:
            if person.track_id not in weapon_hits:
                continue
            vehicle = _nearest_gated_vehicle(person, vehicles, frame_diag, proximity_norm)
            if vehicle is None:
                continue
            confidence, _bbox = weapon_hits[person.track_id]
            hits_by_vehicle.setdefault(vehicle.track_id, []).append((confidence, person.track_id))

        flagged: list[FlaggedEvent] = []
        for vehicle in vehicles:
            hits = hits_by_vehicle.get(vehicle.track_id, [])
            confidence = max((c for c, _ in hits), default=None)
            score = score_weapon_hit(confidence, self._t.weapon_confidence_threshold)
            contributing_person_ids = [person_id for _, person_id in hits]
            event = self._events.update(
                "weapon_at_window",
                (vehicle.track_id,),
                score,
                self._t.weapon_confidence_threshold,
                self._t.weapon_min_duration_s,
                self._t.event_cooldown_s,
                frame_idx,
                timestamp_s,
                [vehicle.track_id, *contributing_person_ids],
                self._t.weapon_max_gap_s,
            )
            if event:
                flagged.append(event)
        return flagged
