import math

from driveway_guard.features.schema import FrameFeatureVector, VehicleConvergenceFeatureVector

# Minimum closing speed (px/s) to count someone as "approaching" rather than
# standing still or drifting past.
_WALKING_SPEED_FLOOR_PX_S = 80.0


def compute_convergence(
    frame_idx: int,
    timestamp_s: float,
    pair_records: list[FrameFeatureVector],
    walking_speed_floor_px_s: float = _WALKING_SPEED_FLOOR_PX_S,
) -> list[VehicleConvergenceFeatureVector]:
    """Reuses the already-extracted per-pair records (rather than
    recomputing velocities) to find, per vehicle, all people currently
    closing in on it, and how widely spread their approach directions are
    around the vehicle."""
    by_vehicle: dict[int, list[FrameFeatureVector]] = {}
    for r in pair_records:
        if r.approach_speed_px_s >= walking_speed_floor_px_s:
            by_vehicle.setdefault(r.vehicle_track_id, []).append(r)

    results: list[VehicleConvergenceFeatureVector] = []
    for vehicle_id, records in by_vehicle.items():
        vehicle_centroid = records[0].vehicle_centroid
        ids: list[int] = []
        bearings: list[float] = []
        for r in records:
            dx = r.person_centroid[0] - vehicle_centroid[0]
            dy = r.person_centroid[1] - vehicle_centroid[1]
            bearings.append(math.degrees(math.atan2(dy, dx)) % 360.0)
            ids.append(r.person_track_id)

        results.append(
            VehicleConvergenceFeatureVector(
                vehicle_track_id=vehicle_id,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                approaching_person_track_ids=ids,
                approaching_person_bearings_deg=bearings,
                angular_spread_deg=_angular_coverage(bearings),
                num_simultaneous_approachers=len(ids),
            )
        )
    return results


def _angular_coverage(bearings_deg: list[float]) -> float:
    """How much of the 360-degree circle around the vehicle is covered by
    approach directions — i.e. 360 minus the largest empty gap. Two people
    approaching from opposite sides yields ~180; people all approaching
    from the same side yields a small value even with several of them."""
    if len(bearings_deg) < 2:
        return 0.0
    sorted_b = sorted(bearings_deg)
    gaps = [sorted_b[i + 1] - sorted_b[i] for i in range(len(sorted_b) - 1)]
    gaps.append(360.0 - sorted_b[-1] + sorted_b[0])
    return 360.0 - max(gaps)
