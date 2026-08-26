import math

from driveway_guard.calibration.schema import EgressPath

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            else:
                x_intersect = x1
            if x1 == x2 or x <= x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def bbox_corners(bbox: BBox) -> list[Point]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _unit(vec: Point) -> Point:
    mag = math.hypot(vec[0], vec[1])
    return (vec[0] / mag, vec[1] / mag)


def _perp(vec: Point) -> Point:
    return (-vec[1], vec[0])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def blocking_overlap_ratio(bbox: BBox, egress_path: EgressPath) -> float:
    """Fraction (0-1) of the egress corridor's width that `bbox` obstructs.

    Zero if the bbox doesn't fall within the corridor's length range at all
    (e.g. a vehicle parked well away from the exit), even if its
    perpendicular projection would otherwise overlap.
    """
    direction = _unit(egress_path.direction_vector)
    perp = _perp(direction)
    exit_point = egress_path.exit_point

    corners = bbox_corners(bbox)
    along_coords = []
    perp_coords = []
    for corner in corners:
        rel = (corner[0] - exit_point[0], corner[1] - exit_point[1])
        along_coords.append(_dot(rel, direction))
        perp_coords.append(_dot(rel, perp))

    along_min, along_max = min(along_coords), max(along_coords)
    # Corridor extends from the exit point backward (negative along direction)
    # into the driveway.
    corridor_along_min, corridor_along_max = -egress_path.corridor_length_px, 0.0
    if along_max < corridor_along_min or along_min > corridor_along_max:
        return 0.0

    half_width = egress_path.corridor_width_px / 2.0
    perp_min, perp_max = min(perp_coords), max(perp_coords)
    overlap = min(perp_max, half_width) - max(perp_min, -half_width)
    overlap = max(0.0, overlap)

    ratio = overlap / egress_path.corridor_width_px
    return max(0.0, min(1.0, ratio))
