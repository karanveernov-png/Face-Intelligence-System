"""
True-positive/false-positive evaluation against REAL, consented face photos.

This is deliberately separate from tests/test_detection.py, tests/
test_recognition.py etc. (which use only synthetic images — see
docs/TROUBLESHOOTING.md for why) per the master spec's explicit
requirement: "Clearly separate unit tests from evaluations that require a
consented real-image dataset."

Every test in this file is skipped automatically, with a clear reason, if
the expected folder structure under samples/eval/ doesn't exist — so the
main test suite stays fully green with zero real photos, and these tests
come alive the moment you (or whoever has the rights to the photos) adds
some. This file could not be completed further in this build: the author
did not have access to real, rights-cleared photos of multiple real
people to supply on your behalf, and sourcing them from the open internet
would repeat exactly the authorization problem docs/TROUBLESHOOTING.md and
MODEL_CARDS.md already raised about using a real, identifiable person's
photo without clear, traceable consent — "give me permission to source
them" doesn't resolve who's actually consenting to have their face in a
biometric test fixture. This harness is the part that *can* be finished
without that; the photos are the part that can't be, responsibly, here.

Expected layout (create this yourself, with photos you have the rights
to use — see samples/README.md):

    samples/eval/
      identities/
        <person-name>/
          enroll.jpg        # the photo used to enroll this identity
          query_same.jpg    # a DIFFERENT photo of the SAME person
      distractors/
        *.jpg                # photos of people who are NOT enrolled
      no_face/
        *.jpg                # images containing no face at all

Run with: `pytest tests/test_evaluation_real_faces.py -v -s` (the -s shows
the printed summary at the end of each test).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.services import detection, recognition

EVAL_ROOT = Path(__file__).resolve().parents[2] / "samples" / "eval"
IDENTITIES_DIR = EVAL_ROOT / "identities"
DISTRACTORS_DIR = EVAL_ROOT / "distractors"
NO_FACE_DIR = EVAL_ROOT / "no_face"

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _identity_dirs() -> list[Path]:
    if not IDENTITIES_DIR.is_dir():
        return []
    return [d for d in IDENTITIES_DIR.iterdir() if d.is_dir() and (d / "enroll.jpg").exists()]


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT]


skip_no_identities = pytest.mark.skipif(
    not _identity_dirs(),
    reason=(
        f"No consented evaluation photos found under {IDENTITIES_DIR}. "
        "This test is skipped by design, not broken -- see this file's module "
        "docstring and samples/README.md for how to add your own."
    ),
)

skip_no_distractors = pytest.mark.skipif(
    not _images_in(DISTRACTORS_DIR),
    reason=f"No distractor photos found under {DISTRACTORS_DIR} -- see samples/README.md.",
)

skip_no_no_face_images = pytest.mark.skipif(
    not _images_in(NO_FACE_DIR),
    reason=f"No no-face evaluation images found under {NO_FACE_DIR} -- see samples/README.md.",
)


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


@skip_no_identities
def test_true_positive_recognition_on_real_faces():
    """For each enrolled identity, a different photo of the SAME person
    must be recognized -- the core "does this actually work on a real
    face" question the synthetic-image unit tests can't answer."""
    results = []
    for identity_dir in _identity_dirs():
        name = identity_dir.name
        enroll_img = cv2.imread(str(identity_dir / "enroll.jpg"))
        query_path = identity_dir / "query_same.jpg"
        assert enroll_img is not None, f"{identity_dir / 'enroll.jpg'} could not be decoded"
        if not query_path.exists():
            results.append((name, "SKIPPED (no query_same.jpg)"))
            continue

        faces, _ = detection.detect_faces(enroll_img)
        assert len(faces) >= 1, f"no face detected in {identity_dir / 'enroll.jpg'}"
        aligned_enroll = recognition.align_face(enroll_img, faces[0])
        recognition.enroll_face(aligned_enroll, name=name, consent_confirmed=True)

        query_img = cv2.imread(str(query_path))
        assert query_img is not None, f"{query_path} could not be decoded"
        q_faces, _ = detection.detect_faces(query_img)
        assert len(q_faces) >= 1, f"no face detected in {query_path}"
        aligned_query = recognition.align_face(query_img, q_faces[0])
        match = recognition.match_face(aligned_query)

        results.append((name, f"matched={match.identity!r} distance={match.distance}"))
        assert match.identity == name, (
            f"expected {name} to match itself across two photos, got {match.identity!r} "
            f"at distance {match.distance} (threshold {match.threshold})"
        )

    print("\nTrue-positive recognition results:")
    for name, outcome in results:
        print(f"  {name}: {outcome}")


@skip_no_identities
@skip_no_distractors
def test_false_positive_rate_against_distractors():
    """Enrolled identities must NOT match unrelated people's photos."""
    for identity_dir in _identity_dirs():
        name = identity_dir.name
        enroll_img = cv2.imread(str(identity_dir / "enroll.jpg"))
        faces, _ = detection.detect_faces(enroll_img)
        assert faces, f"no face detected in {identity_dir / 'enroll.jpg'}"
        aligned = recognition.align_face(enroll_img, faces[0])
        recognition.enroll_face(aligned, name=name, consent_confirmed=True)

    false_positives = []
    for distractor_path in _images_in(DISTRACTORS_DIR):
        img = cv2.imread(str(distractor_path))
        if img is None:
            continue
        faces, _ = detection.detect_faces(img)
        if not faces:
            continue  # detection miss on this distractor is a detection-accuracy question, not a false-positive one
        aligned = recognition.align_face(img, faces[0])
        match = recognition.match_face(aligned)
        if match.identity is not None:
            false_positives.append((distractor_path.name, match.identity, match.distance))

    print(f"\nFalse positives: {len(false_positives)} / {len(_images_in(DISTRACTORS_DIR))} distractor photos")
    for fname, matched_as, dist in false_positives:
        print(f"  {fname} incorrectly matched as {matched_as!r} (distance {dist})")

    assert not false_positives, (
        f"{len(false_positives)} distractor photo(s) incorrectly matched an enrolled identity -- "
        f"see printed output above, and consider tightening settings.recognition_distance_threshold"
    )


@skip_no_no_face_images
def test_detection_true_negative_on_no_face_images():
    """Real (not synthetic) images that genuinely contain no face must not
    produce a false detection."""
    false_detections = []
    for img_path in _images_in(NO_FACE_DIR):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces, _ = detection.detect_faces(img)
        if faces:
            false_detections.append((img_path.name, len(faces)))

    print(f"\nFalse detections: {len(false_detections)} / {len(_images_in(NO_FACE_DIR))} no-face images")
    assert not false_detections, f"detected a face where there wasn't one: {false_detections}"
