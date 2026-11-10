"""
Age-estimation evaluation against REAL, consented photos with a known age.

Same spirit as tests/test_evaluation_real_faces.py: the synthetic-image unit
tests in test_age_buckets.py and test_balanced_stack.py can only prove the
pipeline runs without crashing and returns the right shape -- they say
nothing about whether the age ensemble is actually close to correct on a
real face, or (the specific thing this suite exists to catch) whether it has
quietly drifted back toward defaulting ambiguous/young faces into an adult
bracket, which is the exact bug this ensemble replaced. See
app/services/age_estimation.py's module docstring for that history.

Every test here is skipped automatically, with a clear reason, if
samples/eval/ages/ doesn't exist -- so the main suite stays green with zero
real photos, and these come alive the moment you add your own (consented)
labeled photos. See samples/README.md for the folder layout.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.services import age_estimation, detection

EVAL_ROOT = Path(__file__).resolve().parents[2] / "samples" / "eval"
AGES_DIR = EVAL_ROOT / "ages"

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _age_entries() -> list[tuple[Path, int]]:
    """Each subfolder of ages/ must contain exactly one image plus a
    true_age.txt holding the person's real age (an integer) when that photo
    was taken. Malformed entries are skipped with a printed warning rather
    than failing collection, so one bad folder doesn't hide the rest."""
    if not AGES_DIR.is_dir():
        return []
    entries: list[tuple[Path, int]] = []
    for sub in sorted(AGES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        age_file = sub / "true_age.txt"
        images = [p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXT]
        if not age_file.exists() or not images:
            print(f"  [skip] {sub.name}: needs one image + true_age.txt")
            continue
        try:
            true_age = int(age_file.read_text().strip())
        except ValueError:
            print(f"  [skip] {sub.name}: true_age.txt isn't a plain integer")
            continue
        entries.append((images[0], true_age))
    return entries


skip_no_age_entries = pytest.mark.skipif(
    not _age_entries(),
    reason=(
        f"No consented, labeled age-evaluation photos found under {AGES_DIR}. "
        "This test is skipped by design, not broken -- see this file's module "
        "docstring and samples/README.md for how to add your own."
    ),
)


@skip_no_age_entries
def test_age_estimates_track_real_photos_within_a_loose_tolerance():
    """Sanity check on actual accuracy -- loose on purpose. This is not a
    benchmark run; with only a handful of photos a tight MAE threshold would
    just be noise. It exists to catch the model being badly, systematically
    wrong, not to certify a precise error rate."""
    entries = _age_entries()
    results = []
    errors = []

    for img_path, true_age in entries:
        img = cv2.imread(str(img_path))
        assert img is not None, f"{img_path} could not be decoded"

        faces, _ = detection.detect_faces(img)
        assert faces, f"no face detected in {img_path}"
        face = faces[0]

        result = age_estimation.estimate_age(face.crop_bgr).to_dict()
        results.append((img_path.parent.name, true_age, result))

        if result["estimated_age"] is not None:
            errors.append(result["estimated_age"] - true_age)

    print("\nAge evaluation results:")
    for name, true_age, result in results:
        est = result["estimated_age"]
        unc = result.get("estimated_age_uncertainty_years")
        print(f"  {name}: true={true_age}  est={est} (+/-{unc})  "
              f"range={result['estimated_age_range']}  gender={result.get('gender')}")

    usable = [r for r in results if r[2]["estimated_age"] is not None]
    assert usable, "every photo failed to produce an estimate -- check that the models load and faces are detected"

    if len(errors) >= 3:
        mean_abs_error = sum(abs(e) for e in errors) / len(errors)
        mean_signed_error = sum(errors) / len(errors)
        print(f"  mean absolute error: {mean_abs_error:.1f} years")
        print(f"  mean signed error:   {mean_signed_error:+.1f} years "
              f"(positive = skewing older, negative = skewing younger)")
        # Generous on purpose -- flags "badly wrong," not "imperfect."
        assert mean_abs_error < 15.0, (
            f"mean absolute error of {mean_abs_error:.1f} years across {len(errors)} real photos "
            "is far outside what an ensemble of two reasonable open models should produce -- "
            "check that both genderage.onnx and fairface.onnx are actually loading."
        )


@skip_no_age_entries
def test_estimates_are_not_a_constant_hardcoded_value():
    """Regression guard: earlier fallback paths in this module returned a
    fixed, confident-looking number (e.g. always 22, or always 12) whenever
    something failed internally, rather than surfacing the failure. If every
    distinct real photo here produces the exact same estimate, something is
    silently falling back instead of actually running inference."""
    entries = _age_entries()
    if len(entries) < 2:
        pytest.skip("need at least 2 labeled photos to check for a constant/hardcoded output")

    estimates = []
    for img_path, _true_age in entries:
        img = cv2.imread(str(img_path))
        faces, _ = detection.detect_faces(img)
        if not faces:
            continue
        result = age_estimation.estimate_age(faces[0].crop_bgr).to_dict()
        if result["estimated_age"] is not None:
            estimates.append(result["estimated_age"])

    if len(estimates) < 2:
        pytest.skip("fewer than 2 usable estimates to compare")

    assert len(set(estimates)) > 1, (
        f"every photo produced the identical estimate ({estimates[0]}) -- this is exactly the "
        "failure mode where a broken model silently falls back to one fixed placeholder number"
    )


@skip_no_age_entries
def test_clearly_young_faces_are_not_pushed_into_an_adult_bracket():
    """The specific regression this ensemble was built to prevent: a face
    labeled as clearly pre-teen (true_age < 13, chosen to be generous and
    unambiguous) must not come back estimated as 20+. This threshold is
    deliberately loose -- far looser than the model's own honest
    uncertainty -- because this test exists to catch a reintroduced
    adult-default bias, not to certify fine-grained accuracy."""
    young_entries = [(p, a) for p, a in _age_entries() if a < 13]
    if not young_entries:
        pytest.skip("no photos labeled with true_age < 13 to check this regression against")

    failures = []
    for img_path, true_age in young_entries:
        img = cv2.imread(str(img_path))
        faces, _ = detection.detect_faces(img)
        if not faces:
            continue
        result = age_estimation.estimate_age(faces[0].crop_bgr).to_dict()
        est = result["estimated_age"]
        if est is not None and est >= 20:
            failures.append((img_path.parent.name, true_age, est))

    print(f"\nYoung-face check: {len(failures)} / {len(young_entries)} photos pushed into an adult estimate")
    assert not failures, (
        f"{len(failures)} photo(s) with a stated true_age under 13 were estimated at 20+: {failures} -- "
        "this is the exact adult-default regression age_estimation.py was rewritten to eliminate"
    )
