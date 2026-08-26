from unittest.mock import MagicMock, patch

import numpy as np

from driveway_guard.detection.types import ObjectClass, TrackedObject
from driveway_guard.detection.weapon_detector import WeaponDetector


def _person(track_id=1, bbox=(100, 100, 200, 300)):
    return TrackedObject(track_id=track_id, cls=ObjectClass.PERSON, confidence=0.9, bbox_xyxy=bbox)


def _vehicle(track_id=2, bbox=(50, 50, 400, 400)):
    return TrackedObject(track_id=track_id, cls=ObjectClass.VEHICLE, confidence=0.9, bbox_xyxy=bbox)


def _fake_yolo_result(confs, cls_ids, boxes):
    """Shaped like ultralytics' `model(...)` return value: a list with one
    element whose `.boxes` exposes `.conf`/`.cls`/`.xyxy` as
    `.cpu().numpy()`-chainable, matching what WeaponDetector.detect() reads."""
    result = MagicMock()
    boxes_mock = MagicMock()
    boxes_mock.__len__.return_value = len(confs)
    boxes_mock.conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=float)
    boxes_mock.cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=float)
    boxes_mock.xyxy.cpu.return_value.numpy.return_value = np.array(boxes, dtype=float)
    result.boxes = boxes_mock
    return [result]


@patch("driveway_guard.detection.weapon_detector.YOLO")
def test_detect_keeps_weapon_box_over_person_box_in_same_crop(mock_yolo_cls):
    """Regression test for the non-threat-class-denylist fix: a crop
    centered on a person reliably also contains a `person` box from the
    model itself, sometimes at higher confidence than the real weapon box.
    Plain confs.argmax() would pick the person box; the denylist must
    filter it out first."""
    mock_model = MagicMock()
    mock_model.names = {0: "person", 1: "weapon"}
    mock_model.return_value = _fake_yolo_result(
        confs=[0.95, 0.6], cls_ids=[0, 1], boxes=[[10, 10, 20, 20], [12, 12, 18, 18]]
    )
    mock_yolo_cls.return_value = mock_model

    detector = WeaponDetector("fake.pt", conf=0.1)
    frame = np.zeros((480, 816, 3), dtype=np.uint8)

    results = detector.detect(frame, [_person()], [_vehicle()])

    assert 1 in results
    confidence, _bbox = results[1]
    assert confidence == 0.6


@patch("driveway_guard.detection.weapon_detector.YOLO")
def test_detect_returns_nothing_when_crop_only_has_person_box(mock_yolo_cls):
    mock_model = MagicMock()
    mock_model.names = {0: "person"}
    mock_model.return_value = _fake_yolo_result(confs=[0.9], cls_ids=[0], boxes=[[10, 10, 20, 20]])
    mock_yolo_cls.return_value = mock_model

    detector = WeaponDetector("fake.pt", conf=0.1)
    frame = np.zeros((480, 816, 3), dtype=np.uint8)

    results = detector.detect(frame, [_person()], [_vehicle()])

    assert results == {}


@patch("driveway_guard.detection.weapon_detector.YOLO")
def test_detect_skips_persons_not_gated_near_a_vehicle(mock_yolo_cls):
    mock_model = MagicMock()
    mock_model.names = {0: "weapon"}
    mock_model.return_value = _fake_yolo_result(confs=[0.9], cls_ids=[0], boxes=[[10, 10, 20, 20]])
    mock_yolo_cls.return_value = mock_model

    detector = WeaponDetector("fake.pt", conf=0.1, proximity_norm=0.05)
    frame = np.zeros((480, 816, 3), dtype=np.uint8)
    far_person = _person(bbox=(700, 400, 780, 470))

    results = detector.detect(frame, [far_person], [_vehicle()])

    assert results == {}
    mock_model.assert_not_called()
