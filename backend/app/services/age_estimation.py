"""
Age and gender estimation using an honest two-model ensemble:

  * InsightFace `genderage.onnx` (96x96 input): a direct continuous
    age regression (0-100) plus a 2-way gender logit, trained on a large
    face-recognition-adjacent dataset.
  * FairFace `fairface.onnx` / `fairface_int8.onnx` (224x224 input):
    a 9-bucket age classifier ("0-2" ... "70+") trained on a dataset
    explicitly balanced across age, race, and gender. Its expected
    value over the 9 buckets gives a second, independently-trained
    continuous estimate.

Design intent
--------------
This module previously contained logic that special-cased "is this a
child?" with narrow, easily-missed heuristics (skin texture, forehead
smoothness) and then defaulted anything that didn't clearly trip those
heuristics to a fixed adult bracket ("20-25", forcing gender to male
whenever facial hair was detected). That is backwards: a system that
reports a person's age should never be *more* willing to call someone
an adult than a child when the evidence is ambiguous, and it should
never substitute a confident-looking fixed number for "the model
wasn't sure."

This rewrite removes all of that. There is no child/adult branch and
no case that overrides the model's output toward a particular bracket.
Every face gets the same treatment: average the two independent
models' continuous estimates, report how much they disagree as an
honest uncertainty, and bucket the *result* (not a guess) using the
existing symmetric `_bucket_for_age` ranges. When the face crop can't
be analyzed at all, the result says so explicitly instead of
returning a confident-looking placeholder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.core.onnx_runtime import blob_from_mobilenet_crops, load_onnx_session

logger = logging.getLogger("face_intelligence.age_estimation")

AGE_LIMITATIONS_NOTE = (
    "Ensemble estimate from two independently-trained open models "
    "(InsightFace genderage + FairFace); the reported range reflects how much "
    "they agree, not a calibrated statistical interval. Visual age estimation "
    "from a single image is inherently approximate, especially near category "
    "boundaries. This is for descriptive/analytics purposes only -- it must "
    "not be used to determine whether a specific person is a minor, to gate "
    "access, or for any decision with legal or safety consequences for the "
    "person photographed."
)

# FairFace's 9 age buckets, in the model's output order, with a representative
# midpoint used only to compute an expected value from the softmax -- this is
# a symmetric convenience, not a judgment call about any one bucket.
_FAIRFACE_AGE_MIDPOINTS = np.array(
    [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 78.0], dtype=np.float64
)

_genderage_session = None
_fairface_session = None


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def _get_genderage_session():
    global _genderage_session
    if _genderage_session is not None:
        return _genderage_session

    root_dir = Path(__file__).resolve().parents[3]
    backend_dir = Path(__file__).resolve().parents[2]
    candidates = [
        root_dir / "models" / "genderage.onnx",
        backend_dir / "models" / "genderage.onnx",
        Path("models/genderage.onnx"),
        Path("../models/genderage.onnx"),
    ]

    for path in candidates:
        if path.is_file() and path.stat().st_size > 50_000:
            try:
                _genderage_session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                return _genderage_session
            except Exception as exc:
                logger.warning("Direct FP32 load failed for %s: %s", path, exc)

    # No fallback to a differently-shaped model here on purpose: genderage.onnx's
    # output is a 3-element [gender_logit_0, gender_logit_1, age/100] vector, and
    # age_resnet10_v1_int8.onnx (an 81-way per-year softmax) used to sit in this
    # candidate list -- if it ever loaded instead, _genderage_estimate() would
    # silently read probabilities[2] (a near-zero softmax mass) as "age/100,"
    # producing a confident, wrong, near-0 age for every face. If genderage.onnx
    # is missing, _predict_single still has fairface.onnx as the other ensemble
    # member and reports wider honest uncertainty rather than guessing.
    _genderage_session, _, _ = load_onnx_session(
        "models/genderage.onnx",
        candidate_names=("genderage.onnx",),
    )
    return _genderage_session


def _get_fairface_session():
    """Lazily load the FairFace multi-task (age/gender/race) ONNX model.

    Optional: if it can't be loaded, callers fall back to the genderage
    model alone with a wider reported uncertainty, rather than failing.
    """
    global _fairface_session
    if _fairface_session is not None:
        return _fairface_session
    _fairface_session, _, _ = load_onnx_session(
        "models/fairface.onnx",
        candidate_names=("fairface.onnx", "fairface_int8.onnx"),
    )
    return _fairface_session


def _align_insightface_96(image_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Center-and-scale crop to the 96x96 frame genderage.onnx expects."""
    scale = 96.0 / (max(w, h) * 1.5)
    cx = x + w * 0.5
    cy = y + h * 0.5
    M = np.array(
        [
            [scale, 0.0, 48.0 - cx * scale],
            [0.0, scale, 48.0 - cy * scale],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(image_bgr, M, (96, 96), borderValue=(0, 0, 0))


def _bucket_for_age(age: int) -> str:
    """Map a continuous age estimate to a symmetric demographic interval.

    Bucket boundaries:
      0-2   -> "0-2"
      3-6   -> "4-6"
      7-12  -> "8-12"
      13-20 -> "15-20"
      21-32 -> "25-32"
      33-43 -> "38-43"
      44-53 -> "48-53"
      54+   -> "60-100"

    These bucket widths are symmetric and applied identically to every
    age -- nothing here (or in the callers of this function) special-cases
    "adult" vs. "child" or defaults ambiguous cases toward either end.
    """
    if age <= 2:
        return "0-2"
    if age <= 6:
        return "4-6"
    if age <= 12:
        return "8-12"
    if age <= 20:
        return "15-20"
    if age <= 32:
        return "25-32"
    if age <= 43:
        return "38-43"
    if age <= 53:
        return "48-53"
    return "60-100"


def _genderage_estimate(image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[float, bool, float] | None:
    """Run genderage.onnx. Returns (continuous_age, is_male, gender_confidence) or None."""
    session = _get_genderage_session()
    if session is None:
        return None
    try:
        aligned_96 = _align_insightface_96(image_bgr, x1, y1, x2 - x1, y2 - y1)
        blob = cv2.dnn.blobFromImage(
            aligned_96, scalefactor=1.0, size=(96, 96), mean=(0.0, 0.0, 0.0), swapRB=True,
        )
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        preds = np.squeeze(outputs[0])

        gender_logits = preds[0:2]
        is_male = bool(gender_logits[1] > gender_logits[0])
        gender_probs = _softmax(gender_logits)
        gender_confidence = float(np.max(gender_probs))

        continuous_age = float(preds[2]) * 100.0
        return continuous_age, is_male, gender_confidence
    except Exception as exc:
        logger.error("genderage inference error: %s", exc)
        return None


def _fairface_estimate(face_crop_bgr: np.ndarray) -> float | None:
    """Run fairface.onnx. Returns an expected-value continuous age, or None."""
    session = _get_fairface_session()
    if session is None:
        return None
    try:
        blob = blob_from_mobilenet_crops([face_crop_bgr], size=(224, 224))
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})
        output_names = [o.name for o in session.get_outputs()]
        age_logits = np.squeeze(outputs[output_names.index("age_output")])
        age_probs = _softmax(age_logits.astype(np.float64))
        return float(np.dot(age_probs, _FAIRFACE_AGE_MIDPOINTS))
    except Exception as exc:
        logger.error("fairface inference error: %s", exc)
        return None


def _predict_single(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> "AgeResult":
    x, y, w, h = box
    if w < 10 or h < 10 or image_bgr is None or image_bgr.size == 0:
        return AgeResult(
            estimated_age=None, enabled=True, estimated_age_range=None,
            gender=None, gender_confidence=None, is_placeholder=True,
            unusable_reason="Face crop too small to analyze.",
        )

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return AgeResult(
            estimated_age=None, enabled=True, estimated_age_range=None,
            gender=None, gender_confidence=None, is_placeholder=True,
            unusable_reason="Face crop was empty.",
        )

    genderage_result = _genderage_estimate(image_bgr, x1, y1, x2, y2)
    fairface_age = _fairface_estimate(face_crop)

    estimates: list[float] = []
    if genderage_result is not None:
        estimates.append(genderage_result[0])
    if fairface_age is not None:
        estimates.append(fairface_age)

    if not estimates:
        # Neither model produced a result -- say so honestly rather than
        # returning a confident-looking number.
        return AgeResult(
            estimated_age=None, enabled=True, estimated_age_range=None,
            gender=None, gender_confidence=None, is_placeholder=True,
            unusable_reason="Age models unavailable or inference failed for this frame.",
        )

    estimated_age = float(np.mean(estimates))

    # Honest uncertainty: a baseline reflecting that single-image visual age
    # estimation is approximate even for one model, widened by how much the
    # two independent models actually disagree on this specific face. This
    # is a heuristic band, not a calibrated confidence interval -- see
    # AGE_LIMITATIONS_NOTE.
    base_uncertainty = 6.0
    if len(estimates) == 2:
        disagreement = abs(estimates[0] - estimates[1])
        uncertainty_years = base_uncertainty + 0.5 * disagreement
    else:
        # Only one model produced a result -- report wider uncertainty to
        # reflect the missing cross-check, not a different bias.
        uncertainty_years = base_uncertainty * 1.5

    gender: str | None = None
    gender_confidence: float | None = None
    if genderage_result is not None:
        _, is_male, gconf = genderage_result
        gender = "male" if is_male else "female"
        gender_confidence = round(gconf * 100.0, 1)

    age_int = int(round(max(0.0, estimated_age)))
    return AgeResult(
        estimated_age=round(estimated_age, 1),
        enabled=True,
        estimated_age_range=_bucket_for_age(age_int),
        estimated_age_uncertainty_years=round(uncertainty_years, 1),
        gender=gender,
        gender_confidence=gender_confidence,
        is_placeholder=False,
    )


@dataclass
class AgeResult:
    estimated_age: float | None
    enabled: bool
    estimated_age_range: str | None = None
    estimated_age_uncertainty_years: float | None = None
    gender: str | None = None
    gender_confidence: float | None = None
    is_placeholder: bool = False
    unusable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "estimated_age": None,
                "estimated_age_range": None,
                "note": "Age estimation disabled in settings.",
            }
        if self.is_placeholder:
            return {
                "enabled": True,
                "estimated_age": None,
                "estimated_age_range": None,
                "estimated_age_uncertainty_years": None,
                "gender": None,
                "gender_confidence": None,
                "note": self.unusable_reason or "Could not produce an age estimate for this face.",
            }
        return {
            "enabled": True,
            "estimated_age": self.estimated_age,
            "estimated_age_range": self.estimated_age_range,
            "estimated_age_uncertainty_years": self.estimated_age_uncertainty_years,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence,
            "note": AGE_LIMITATIONS_NOTE,
        }


def check_model() -> None:
    dummy = np.zeros((96, 96, 3), dtype=np.uint8)
    result = estimate_ages_batch([dummy])
    if not result:
        raise RuntimeError("Age estimation pipeline returned no result for the status probe.")


def estimate_ages_batch_in_frame(
    image_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]]
) -> list[AgeResult]:
    if not boxes or not settings.age_estimation_enabled:
        return [AgeResult(estimated_age=None, enabled=False) for _ in boxes]
    return [_predict_single(image_bgr, box) for box in boxes]


def estimate_ages_batch(faces_bgr: Sequence[np.ndarray]) -> list[AgeResult]:
    if not faces_bgr:
        return []
    if not settings.age_estimation_enabled:
        return [AgeResult(estimated_age=None, enabled=False) for _ in faces_bgr]
    return [
        _predict_single(face, (0, 0, face.shape[1], face.shape[0]))
        for face in faces_bgr
    ]


def estimate_age(face_bgr: np.ndarray) -> AgeResult:
    res = estimate_ages_batch([face_bgr])
    return res[0] if res else AgeResult(estimated_age=None, enabled=True, is_placeholder=True)
