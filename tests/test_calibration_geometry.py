from driveway_guard.calibration.geometry import blocking_overlap_ratio, point_in_polygon
from driveway_guard.calibration.schema import EgressPath

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon_inside():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon((15, 5), SQUARE) is False


def test_point_in_polygon_edge_cases_do_not_crash():
    assert point_in_polygon((0, 0), SQUARE) in (True, False)
    assert point_in_polygon((-1, -1), SQUARE) is False


def _straight_down_egress(corridor_width=200.0, corridor_length=500.0) -> EgressPath:
    return EgressPath(
        exit_point=(500.0, 900.0),
        direction_vector=(0.0, 1.0),
        corridor_width_px=corridor_width,
        corridor_length_px=corridor_length,
    )


def test_blocking_overlap_full_when_vehicle_spans_corridor_width():
    egress = _straight_down_egress(corridor_width=200.0)
    # A vehicle bbox at least as wide as the 200px corridor, centered on
    # it and well within its length range -> full coverage.
    bbox = (350.0, 700.0, 650.0, 780.0)
    ratio = blocking_overlap_ratio(bbox, egress)
    assert ratio == 1.0


def test_blocking_overlap_half_when_vehicle_narrower_than_corridor():
    egress = _straight_down_egress(corridor_width=200.0)
    # A 100px-wide bbox centered on a 200px corridor can only ever cover
    # half of it, even though it's perfectly centered.
    bbox = (450.0, 700.0, 550.0, 780.0)
    ratio = blocking_overlap_ratio(bbox, egress)
    assert abs(ratio - 0.5) < 1e-6


def test_blocking_overlap_zero_when_far_outside_corridor_width():
    egress = _straight_down_egress(corridor_width=200.0)
    # Perpendicular offset way beyond the corridor half-width (100px).
    bbox = (1400.0, 700.0, 1500.0, 780.0)
    ratio = blocking_overlap_ratio(bbox, egress)
    assert ratio == 0.0


def test_blocking_overlap_partial_at_corridor_edge():
    egress = _straight_down_egress(corridor_width=200.0)
    # Corridor spans x in [400, 600]. bbox spans x in [580, 680]:
    # overlap is [580, 600] = 20px of a 200px-wide corridor -> ratio 0.1.
    bbox = (580.0, 700.0, 680.0, 780.0)
    ratio = blocking_overlap_ratio(bbox, egress)
    assert abs(ratio - 0.1) < 1e-6


def test_blocking_overlap_zero_when_outside_corridor_length():
    egress = _straight_down_egress(corridor_width=200.0, corridor_length=300.0)
    # Perpendicularly centered, but far up the driveway beyond corridor_length.
    bbox = (450.0, 0.0, 550.0, 50.0)
    ratio = blocking_overlap_ratio(bbox, egress)
    assert ratio == 0.0
