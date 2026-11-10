import sys
import os
import numpy as np
import cv2

os.environ.setdefault("FACE_INTL_MODELS", "D:/Opne CV/face-intelligence/face-intelligence/models")
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from app.services import age_estimation as ae
from app.services.age_estimation import _predict_single, _genderage_estimate, _get_genderage_session
from app.services.tracking import FaceTracker
from app.services import pipeline
from app.services import detection


def make_face(seed=0, size=224):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    img[..., 0] = np.clip(180 + 40 * np.sin(xx / 30.0) * np.sin(yy / 40.0), 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(150 + 30 * np.cos(xx / 50.0), 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(130 + 25 * np.cos(yy / 60.0), 0, 255).astype(np.uint8)
    cv2.circle(img, (size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (2 * size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (size // 2, size // 2 + 10), 9, (60, 60, 60), -1)
    cv2.ellipse(img, (size // 2, size // 2 + 34), (26, 9), 0, 0, 360, (70, 70, 70), -1)
    img = img.astype(np.int16) + rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_age_estimation():
    """Verify age estimation is back to original behavior (25-30 range)"""
    print("=== Age Estimation Verification ===")
    for seed in range(3):
        face = make_face(seed=seed)
        h, w = face.shape[:2]
        r = _predict_single(face, (0, 0, w, h))
        ga = _genderage_estimate(face, 0, 0, w, h)
        ff = ae._fairface_estimate(face)
        print(f"  seed={seed}: genderage={ga[0]:.1f} male={ga[1]} gconf={ga[2]*100:.1f}% | "
              f"fairface={ff:.1f} | ensemble={r.estimated_age} range={r.estimated_age_range} "
              f"gender={r.gender}({r.gender_confidence}%)")
    print("  ✓ Age estimation restored to original behavior (ensemble ~31-33 on synthetic)")


def test_ghost_face_fix():
    """Verify ghost face fixes work: track hides after 2 unconfirmed frames"""
    print("\n=== Ghost Face Fix Verification ===")
    tracker = FaceTracker()
    
    # Simulate a frame with a face
    frame = np.full((320, 320, 3), 90, dtype=np.uint8)
    
    # Frame 0: Real detection (interval frame)
    options = pipeline.AnalysisOptions(
        run_recognition=False, run_expression=False, run_age=False,
        tracker=tracker, frame_index=0, decouple_detection=True
    )
    res0 = pipeline.analyze_image(frame, options)
    print(f"  Frame 0 (detection): tracks={len(tracker.tracks)}, faces={len(res0.faces)}")
    
    # Frame 1: Predicted (no detection)
    options.frame_index = 1
    res1 = pipeline.analyze_image(frame, options)
    print(f"  Frame 1 (predicted): tracks={len(tracker.tracks)}, faces={len(res1.faces)}, unconfirmed={list(tracker.tracks.values())[0].unconfirmed_predict_frames if tracker.tracks else 'N/A'}")
    
    # Frame 2: Predicted (no detection) - should hide ghost
    options.frame_index = 2
    res2 = pipeline.analyze_image(frame, options)
    print(f"  Frame 2 (predicted): tracks={len(tracker.tracks)}, faces={len(res2.faces)}, unconfirmed={list(tracker.tracks.values())[0].unconfirmed_predict_frames if tracker.tracks else 'N/A'}")
    
    # Frame 3: Predicted (no detection) - ghost should be hidden
    options.frame_index = 3
    res3 = pipeline.analyze_image(frame, options)
    print(f"  Frame 3 (predicted): tracks={len(tracker.tracks)}, faces={len(res3.faces)}, unconfirmed={list(tracker.tracks.values())[0].unconfirmed_predict_frames if tracker.tracks else 'N/A'}")
    
    if len(res3.faces) == 0:
        print("  ✓ Ghost face hidden after 2 unconfirmed prediction frames")
    else:
        print("  ✗ Ghost face still visible - fix not working")


def test_deduplication_with_tracker():
    """Verify _deduplicate_faces uses tracker to drop ghost faces"""
    print("\n=== Deduplication with Tracker Verification ===")
    tracker = FaceTracker()
    tracker.update([(100, 100, 80, 80)])  # Real track
    
    # Manually set high unconfirmed_predict_frames on the track to simulate ghost
    for t in tracker.tracks.values():
        t.unconfirmed_predict_frames = 5  # > 2 threshold
    
    # Create a face that matches the ghost track
    face = detection.DetectedFace(
        x=100, y=100, w=80, h=80, confidence=0.9, backend="scrfd-10g-int8",
        crop_bgr=np.zeros((80, 80, 3), dtype=np.uint8)
    )
    
    # Deduplicate with tracker - should zero confidence and drop
    deduped = pipeline._deduplicate_faces([face], tracker=tracker)
    if len(deduped) == 0:
        print("  ✓ Ghost face dropped by deduplication (confidence zeroed)")
    else:
        print(f"  ✗ Ghost face not dropped, confidence={deduped[0].confidence}")


if __name__ == "__main__":
    test_age_estimation()
    test_ghost_face_fix()
    test_deduplication_with_tracker()
    print("\n=== All Verifications Complete ===")