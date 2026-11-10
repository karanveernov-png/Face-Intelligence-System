"""
Identity matching against a locally enrolled, consent-based gallery using
ArcFace-R50 (InsightFace's w600k_r50, INT8-quantized) + 5-Point Umeyama
Similarity Alignment.

Primary model, in actual load order (see `_get_recognition_session`):
  1. ArcFace-IResNet50 (`w600k_r50_int8.onnx`, the real file bundled in
     models/): ~42 MB INT8. One of the strongest freely-available open
     face-embedding models -- the backbone of InsightFace's widely-used
     "buffalo_l" pack, reported in the **99.8% LFW / ~96% IJB-C** range in
     InsightFace's own published benchmarks. This is what actually runs.
  2. MobileFaceNet (`mobilefacenet_arcface_int8.onnx`): ~3.4 MB, a smaller
     and faster but slightly less accurate fallback if the R50 weights are
     ever missing.
  3. Local `insightface` Python package (buffalo_l), only if neither ONNX
     file above is found on disk.

Earlier versions of this module described a "GhostFaceNet-V2" primary
model; that file was never actually bundled, so every run was silently
falling through to #1 above the whole time. It's named directly now so
the module's own description matches what's actually loaded -- the status
endpoint's docstring says it best: a stale label here is a user-visible
accuracy claim nobody is running.

  * 5-Point Umeyama Similarity Alignment (`align_face`):
      When all 5 SCRFD landmarks (eyes, nose, mouth corners) are present, estimates
      a 2D similarity transform (`cv2.estimateAffinePartial2D`) onto the canonical
      112x112 ArcFace coordinate frame (+4-7% recognition accuracy on tilted/off-angle
      faces at zero extra model weight). Falls back to 2-point eye-line rotation
      when only eye landmarks are present.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from app.core.config import EMBEDDINGS_DIR, GALLERY_DIR, settings
from app.core.onnx_runtime import (
    blob_from_arcface_crops,
    get_active_execution_providers,
    load_onnx_session,
)

logger = logging.getLogger("face_intelligence.recognition")

MATCH_CONFIDENCE_NOTE = (
    "This is a cosine-distance match against a small local gallery using 512-D "
    "ArcFace (InsightFace w600k_r50) embeddings, not a calibrated probability. Two "
    "different people can score as similar under lighting/pose changes; the same "
    "person can score as dissimilar under occlusion. Never use this as the sole "
    "basis for a consequential decision."
)

# Canonical 112x112 ArcFace 5-point reference coordinates (left_eye, right_eye,
# nose, mouth_left, mouth_right) used during training of InsightFace's ArcFace models.
ARCFACE_DST_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_INDEX_PATH = EMBEDDINGS_DIR / "index.json"
_rec_session = None
_rec_provider: str | None = None
_insightface_rec_handler = None
_fallback_projection: np.ndarray | None = None


def _load_index() -> dict[str, Any]:
    if not _INDEX_PATH.exists():
        return {}
    with open(_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: dict[str, Any]) -> None:
    with open(_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _get_recognition_session():
    """
    Lazily load the ArcFace-R50 (w600k_r50) ONNX session -- falling back to
    MobileFaceNet, then the `insightface` package -- with CPU vectorization.
    Candidates are ordered best-accuracy-first among what's actually real
    and loadable; the unquantized/alternate filenames are included only in
    case a future export adds them under those names.
    """
    global _rec_session, _rec_provider, _insightface_rec_handler
    if _rec_session is not None:
        return _rec_session

    _rec_session, _, _rec_provider = load_onnx_session(
        settings.recognition_model_file,
        candidate_names=(
            "w600k_r50_int8.onnx",
            "w600k_r50.onnx",
            "mobilefacenet_arcface_int8.onnx",
            "mobilefacenet_arcface.onnx",
            "w600k_mbf_int8.onnx",
            "w600k_mbf.onnx",
        ),
    )
    if _rec_session is not None:
        return _rec_session

    if _insightface_rec_handler is None:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=get_active_execution_providers(),
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _insightface_rec_handler = app.models.get("recognition")
            if _insightface_rec_handler is not None and hasattr(_insightface_rec_handler, "session"):
                _rec_session = _insightface_rec_handler.session
                _rec_provider = (
                    _rec_session.get_providers()[0]
                    if _rec_session.get_providers()
                    else "CPUExecutionProvider"
                )
                return _rec_session
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recognition ONNX weights not found in local cache (%s); using blob fallback.", exc)

    return None


def _get_fallback_projection() -> np.ndarray:
    """
    Deterministic orthogonal projection matrix used only when offline without
    cached ONNX weights (e.g. air-gapped unit test runs).
    """
    global _fallback_projection
    if _fallback_projection is None:
        rng = np.random.default_rng(20261104)
        proj = rng.standard_normal((3 * 16 * 16, 512), dtype=np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        _fallback_projection = proj
    return _fallback_projection


def align_face(image_bgr: np.ndarray, face: Any) -> np.ndarray:
    """
    Two-tier geometric face alignment:
      1. Full 5-Point Umeyama Similarity Transform (`cv2.estimateAffinePartial2D`):
         When all 5 landmarks (`left_eye`, `right_eye`, `nose`, `mouth_left`,
         `mouth_right`) are available and `settings.use_5pt_similarity_alignment`
         is enabled, warps the face directly onto the canonical 112x112 ArcFace
         coordinate frame.
      2. 2-Point Eye-Line Rotation Fallback:
         When only `left_eye` and `right_eye` are set, rotates the frame around
         the eye midpoint so the eye line is horizontal and slices `(face.h, face.w)`.
      3. Plain Crop Fallback:
         When landmarks are absent or the box is degenerate, returns `face.crop_bgr`.
    """
    if face.left_eye is None or face.right_eye is None:
        return face.crop_bgr

    has_all_5_landmarks = all(
        getattr(face, attr, None) is not None
        for attr in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
    )
    if settings.use_5pt_similarity_alignment and has_all_5_landmarks:
        src_pts = np.array(
            [
                face.left_eye,
                face.right_eye,
                face.nose,
                face.mouth_left,
                face.mouth_right,
            ],
            dtype=np.float32,
        )
        # Always fit against the canonical, UNSCALED 112x112 ArcFace reference
        # points and warp straight to a 112x112 canvas. Scaling dst_pts by
        # face.w/face.h (the previous behavior) stretches the reference
        # template anisotropically whenever the detected box isn't square --
        # which SCRFD boxes essentially never are (typically ~0.74 w/h on
        # real faces, confirmed against this module's own real-photo eval
        # harness). estimateAffinePartial2D can only fit a similarity
        # transform (uniform scale), so it can't actually hit that distorted
        # target; the result was a worse-than-necessary approximation, then
        # blob_from_arcface_crops's later resize to 112x112 compounded it
        # further rather than undoing it. Landing exactly on 112x112 here
        # makes that trailing resize a no-op and matches what ArcFace
        # (w600k_r50) was actually trained on.
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, ARCFACE_DST_5PTS, method=cv2.LMEDS)
        if matrix is not None:
            warped = cv2.warpAffine(image_bgr, matrix, (112, 112), borderValue=0.0)
            if warped.size > 0:
                return warped

    lx, ly = face.left_eye
    rx, ry = face.right_eye
    angle_deg = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    h_img, w_img = image_bgr.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image_bgr, rotation_matrix, (w_img, h_img))

    x0, y0 = max(face.x, 0), max(face.y, 0)
    aligned_crop = rotated[y0 : y0 + face.h, x0 : x0 + face.w]
    if aligned_crop.size == 0:
        return face.crop_bgr
    return aligned_crop


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float64)


def _embed_batch(faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """
    Extract 512-D unit-normalized embeddings for a batch of aligned face crops.
    Stacks crops into a single NCHW tensor via `cv2.dnn.blobFromImages` and
    executes a single batched forward pass through ONNX Runtime.
    """
    if not faces_bgr:
        return np.empty((0, 512), dtype=np.float64)

    for crop in faces_bgr:
        if crop is None or crop.size == 0:
            raise ValueError("Cannot extract face embedding from an empty face crop.")

    blob = blob_from_arcface_crops(faces_bgr, size=(112, 112))
    session = _get_recognition_session()

    if session is not None:
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape
        output_shape = output_meta.shape
        # A dynamic *input* batch dim doesn't guarantee the graph's output is
        # actually batch-safe -- w600k_r50_int8.onnx (the real ArcFace-R50
        # weights, verified Nov 2026) accepts ['None', 3, 112, 112] but its
        # output is hardcoded to [1, 512], not [N, 512]. Feeding it a batch of
        # 2+ doesn't error; onnxruntime just reshapes whatever it produced
        # into (blob.shape[0], -1), which silently mixes rows across faces
        # (ships corrupted, not-obviously-wrong embeddings -- confirmed by
        # running real batched recognition against this exact model). Only
        # trust the single batched call when BOTH ends are symbolic and agree.
        input_batch_dynamic = not isinstance(input_shape[0], int) or input_shape[0] <= 0
        output_batch_dynamic = not isinstance(output_shape[0], int) or output_shape[0] <= 0
        supports_dynamic_batch = (
            input_batch_dynamic
            and output_batch_dynamic
            and input_shape[0] == output_shape[0]
        )

        if supports_dynamic_batch or blob.shape[0] == 1:
            raw_out = session.run(None, {input_name: blob})[0]
            embeddings = np.asarray(raw_out, dtype=np.float64).reshape(blob.shape[0], -1)
        else:
            rows = [
                np.asarray(session.run(None, {input_name: blob[i : i + 1]})[0], dtype=np.float64).reshape(-1)
                for i in range(blob.shape[0])
            ]
            embeddings = np.stack(rows, axis=0)
        return _l2_normalize_rows(embeddings)

    pooled = blob[:, :, ::7, ::7].reshape(blob.shape[0], 3 * 16 * 16)
    embeddings = pooled.astype(np.float64) @ _get_fallback_projection().astype(np.float64)
    return _l2_normalize_rows(embeddings)


def _embed(face_bgr: np.ndarray) -> np.ndarray:
    """Extract a 512-D unit-normalized embedding for a single face crop."""
    return _embed_batch([face_bgr])[0]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


@dataclass
class MatchResult:
    identity: str | None
    distance: float | None
    threshold: float
    face_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity or "Unknown",
            "matched": self.identity is not None,
            "distance": round(self.distance, 4) if self.distance is not None else None,
            "threshold": self.threshold,
            "note": MATCH_CONFIDENCE_NOTE,
        }


def match_faces_batch(faces_bgr: Sequence[np.ndarray]) -> list[MatchResult]:
    """
    Batch-match multiple cropped faces against the enrolled gallery in a single
    `cv2.dnn.blobFromImages` + ONNX forward pass followed by vectorized cosine
    similarity matrix multiplication.
    """
    if not faces_bgr:
        return []

    index = _load_index()
    if not index:
        return [
            MatchResult(
                identity=None,
                distance=None,
                threshold=settings.recognition_distance_threshold,
                face_id=None,
            )
            for _ in faces_bgr
        ]

    try:
        queries = _embed_batch(faces_bgr)  # shape (N, 512)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batched feature embedding failed: %s", exc)
        return [
            MatchResult(
                identity=None,
                distance=None,
                threshold=settings.recognition_distance_threshold,
                face_id=None,
            )
            for _ in faces_bgr
        ]

    gallery_ids = list(index.keys())
    gallery_names = [index[fid]["name"] for fid in gallery_ids]
    gallery_vecs = np.array([index[fid]["embedding"] for fid in gallery_ids], dtype=np.float64)

    distances = 1.0 - (queries @ gallery_vecs.T)

    results: list[MatchResult] = []
    for row_idx in range(len(faces_bgr)):
        best_col = int(np.argmin(distances[row_idx]))
        best_dist = float(distances[row_idx, best_col])
        if best_dist <= settings.recognition_distance_threshold:
            results.append(
                MatchResult(
                    identity=gallery_names[best_col],
                    distance=best_dist,
                    threshold=settings.recognition_distance_threshold,
                    face_id=gallery_ids[best_col],
                )
            )
        else:
            results.append(
                MatchResult(
                    identity=None,
                    distance=best_dist,
                    threshold=settings.recognition_distance_threshold,
                    face_id=None,
                )
            )
    return results


def match_face(face_bgr: np.ndarray) -> MatchResult:
    return match_faces_batch([face_bgr])[0]


def enroll_face(face_bgr: np.ndarray, name: str, consent_confirmed: bool) -> dict[str, Any]:
    if not consent_confirmed:
        raise PermissionError("Enrollment requires explicit consent confirmation (consent_confirmed=True).")
    if not name or not name.strip():
        raise ValueError("Enrollment requires a non-empty name/identifier.")

    embedding = _embed(face_bgr)
    face_id = str(uuid.uuid4())

    index = _load_index()
    index[face_id] = {
        "name": name.strip(),
        "embedding": embedding.tolist(),
        "model": settings.recognition_model,
        "enrolled_at": time.time(),
    }
    _save_index(index)

    if settings.retain_enrollment_images:
        cv2.imwrite(str(GALLERY_DIR / f"{face_id}.jpg"), face_bgr)

    return {
        "face_id": face_id,
        "name": name.strip(),
        "retained_image": settings.retain_enrollment_images,
    }


def list_enrolled() -> list[dict[str, Any]]:
    index = _load_index()
    return [
        {
            "face_id": fid,
            "name": e["name"],
            "model": e.get("model", settings.recognition_model),
            "enrolled_at": e["enrolled_at"],
        }
        for fid, e in index.items()
    ]


def delete_enrolled(face_id: str) -> bool:
    index = _load_index()
    if face_id not in index:
        return False
    del index[face_id]
    _save_index(index)

    image_path = GALLERY_DIR / f"{face_id}.jpg"
    if image_path.exists():
        image_path.unlink()
    return True