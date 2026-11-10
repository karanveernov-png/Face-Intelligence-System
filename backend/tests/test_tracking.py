from app.services.tracking import FaceTracker, _iou


def test_iou_identical_boxes_is_one():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap_between_zero_and_one():
    iou = _iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert 0.0 < iou < 1.0


def test_new_detection_creates_a_track():
    tracker = FaceTracker()
    tracks = tracker.update([(10, 10, 50, 50)])
    assert len(tracks) == 1


def test_same_face_across_frames_keeps_same_track_id():
    tracker = FaceTracker()
    tracks1 = tracker.update([(10, 10, 50, 50)])
    track_id = next(iter(tracks1.keys()))

    # Slight movement each frame, like a real video of a moving face.
    tracks2 = tracker.update([(12, 11, 50, 50)])
    tracks3 = tracker.update([(15, 13, 50, 50)])

    assert track_id in tracks2
    assert track_id in tracks3
    assert len(tracks3) == 1  # still only one face overall


def test_two_simultaneous_faces_get_different_track_ids():
    tracker = FaceTracker()
    tracks = tracker.update([(0, 0, 50, 50), (300, 300, 50, 50)])
    assert len(tracks) == 2


def test_track_expires_after_max_missed_frames():
    from app.core.config import settings
    tracker = FaceTracker()
    tracker.update([(10, 10, 50, 50)])

    # Feed frames with no detections at all until the track should expire.
    for _ in range(settings.tracker_max_missed_frames + 1):
        tracks = tracker.update([])

    assert len(tracks) == 0
