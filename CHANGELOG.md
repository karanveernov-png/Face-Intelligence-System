# Changelog

## Iteration 7 — Full black/white/grey-box test pass; decompression-bomb fix

Added 84 new tests across 4 new files (`test_security_blackbox.py`,
`test_onnx_runtime_whitebox.py`, `test_expressions_whitebox.py`,
`test_pipeline_greybox.py`) plus one new WS regression test, taking the suite
from 67 to 151 passing tests. Organized by method:

- **Black-box** (`test_security_blackbox.py`): every HTTP/WS attack surface
  exercised as an external client would -- upload validation, path-traversal
  attempts in filenames and `face_id`, injection-style strings in free-text
  fields, malformed/missing form fields, and the decompression-bomb fix below.
- **White-box** (`test_onnx_runtime_whitebox.py`, `test_expressions_whitebox.py`):
  internal functions in two modules that had zero dedicated tests before --
  `core/onnx_runtime.py`'s model-loading and blob-normalization internals, and
  `expressions.py`'s confidence-threshold boundary and 8-class/7-class
  label selection.
- **Grey-box** (`test_pipeline_greybox.py`): forced failure injection into
  `pipeline.py`'s documented per-face fallback paths (recognition,
  expression, age) to confirm they actually activate and report errors
  correctly, plus the `decouple_detection`/tracker boolean logic's edge
  cases (empty tracker, interval vs. non-interval frames).

**Bugs found and fixed:**

1. **Decompression-bomb vulnerability (the significant one).** No code
   path checked a decoded image's pixel dimensions anywhere -- only the
   compressed upload size was capped (`max_upload_size_mb`). A highly
   compressible image (e.g. a large solid-color PNG) could be under 1 MB on
   disk and decode to over a gigabyte in memory; demonstrated a 0.78 MB file
   decoding to 768 MB before the fix. Added `max_decoded_image_pixels` /
   `max_decoded_image_dimension_px` to `core/config.py` and
   `reject_oversized_decoded_image` / `reject_oversized_dimensions` to
   `security/validation.py` (the latter checks container-reported
   width/height *without allocating a buffer of that size itself* -- an
   earlier draft of this fix used `np.empty((h, w, 3))` for the video
   pre-check, which would have allocated the very memory it was meant to
   prevent; caught and fixed before it shipped). Wired into
   `/api/analyze/image`, `/api/faces/enroll`, `/api/analyze/video` (checked
   against container metadata before decoding any frame), and `/ws/analyze`.
2. **A misleading test, not a code bug**: `test_api.py`'s
   `test_enroll_requires_consent_field` asserted the right status code (422)
   for the wrong reason -- its fixture image has no face, so the route's
   "no face detected" check fired before the consent check was ever reached.
   The actual 403-on-no-consent path had zero regression coverage. Renamed
   to document what it actually tests, and added a properly-targeted test
   (`test_security_blackbox.py::test_enroll_without_consent_on_a_real_detected_face_is_403_forbidden`)
   that mocks detection to reach the real consent-enforcement branch.
3. Confirmed-safe, not a bug: `delete_enrolled`'s `face_id` path parameter
   reaches a filesystem path (`GALLERY_DIR / f"{face_id}.jpg"`), but only
   after `face_id` is confirmed to be a key in the in-memory index --
   impossible for an attacker-supplied path-traversal string to be, since
   index keys are always server-generated UUIDs. Added regression tests
   rather than treating it as closed-by-inspection.

**Checked and found already correct, no change made**: `expressions.py`'s
`result_from_scores_dict` uses a deliberately low 0.22 confidence threshold
(documented in its own comment as an intentional "catch micro-expressions"
tradeoff). Verified the threshold's exact boundary behavior with tests but
left the calibration itself alone -- it's a documented product choice, not
an unintended defect, and distinct in kind from the age-estimation bias
fixed in Iteration 6.

---

## Iteration 6 — Honest age estimation; recognition config/docs corrected to match reality

- **Age Estimation (`backend/app/services/age_estimation.py`)**:
  Removed heuristic "is this a child?" branching (skin-texture smoothness,
  forehead smoothness, facial-hair detection) that defaulted ambiguous faces
  toward a fixed adult bracket and forced gender to male whenever facial hair
  was detected. Replaced with an honest two-model ensemble: InsightFace
  `genderage.onnx` (continuous regression) and FairFace `fairface.onnx`
  (9-bucket classifier, expected value over its softmax), averaged, with their
  disagreement reported as `estimated_age_uncertainty_years` instead of
  collapsed into a single confident-looking number. A face crop that can't be
  analyzed now returns an explicit "could not estimate" result rather than a
  placeholder that looked like a real prediction. `_bucket_for_age` (already
  symmetric, unchanged) is still used for the presentation bucket. Frontend
  (`SettingsPanel.tsx`, `ResultsPanel.tsx`, `types/api.ts`) updated to match
  and to surface the new uncertainty value.
- **Face Recognition (`backend/app/services/recognition.py`, `core/config.py`)**:
  `recognition_model_file` previously pointed at `ghostfacenet_v2_arcface_int8.onnx`,
  a filename that was never actually bundled — every run was silently falling
  through the candidate list to `w600k_r50_int8.onnx` (ArcFace-R50) instead.
  Config, docstrings, the `/api/models/status` label, and the Settings-panel
  hint now name that real model directly. No behavior changed (the same model
  was already running); this just makes the documentation match what's
  actually loaded. `docs/MODEL_CARDS.md` and `README.md` corrected for the
  same reason, including an unverified "~3.6 years MAE" age-model benchmark
  that was never measured against anything running in this repo.

---

## Iteration 5 — Upgrade to the 4-Model "Balanced" Production Stack & Real-Time Pipeline Strategy

Replaced the heavyweight / mixed-framework models (RetinaFace, InsightFace `antelopev2` ResNet-100, DeepFace Mini-Xception, and HuggingFace PyTorch ViT age classifier totaling >700 MB) with the **4-Model Balanced Production Stack** (~27.2 MB combined ONNX footprint, 30+ FPS on CPU):

### 1. The 4-Model Balanced Roster
- **Detection & Alignment (`backend/app/services/detection.py`)**:
  Upgraded to **SCRFD-2.5G** (`scrfd_2.5g_bnkps.onnx`, ~3.2 MB, 93.8% WiderFace Hard, ~8–12 ms CPU latency) with full multi-stride anchor-free ONNX decoding for bounding boxes and 5-point facial landmarks (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`), retaining OpenCV Haar Cascade as a zero-dependency offline fallback.
- **Face Recognition (`backend/app/services/recognition.py`)**:
  Upgraded from `antelopev2` to **MobileFaceNet (ArcFace loss)** (`mobilefacenet_arcface.onnx` / `w600k_mbf.onnx`, ~8.0 MB, 99.65% LFW / 93.2% IJB-C, ~10–15 ms CPU latency), outputting 512-D unit-normalized embeddings matched via vectorized cosine distance (`1.0 - queries @ gallery.T`).
- **Emotion Analysis (`backend/app/services/expressions.py`)**:
  Upgraded from FER2013 Mini-Xception to **HSEmotion (MobileNet-V3 / EfficientNet-B0 8-class ONNX)** (`hsemotion_mobilenetv3.onnx`, ~8.5 MB, 85.5% RAF-DB across 8 classes: `angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
- **Age Estimation (`backend/app/services/age_estimation.py`)**:
  Upgraded from the 350 MB+ PyTorch ViT (`nateraw/vit-age-classifier`) to **MobileNetV3 (Trained on FairFace)** (`fairface_age_mobilenetv3.onnx`, ~7.5 MB, ~4.4 years MAE), computing expected age from the FairFace distribution and mapping into presentation buckets (`_bucket_for_age`).

### 2. Pipeline Performance Strategy
- **Decoupled Detection from Recognition (`backend/app/services/tracking.py`, `pipeline.py`, `routes.py`, `ws.py`)**:
  Runs SCRFD-2.5G detection on every $N$-th frame (`settings.detection_interval = 3`), propagating bounding boxes and landmarks on intermediate frames via pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) and centroid velocity tracking (`FaceTracker.predict_inter_frame`).
- **Batched Cropped Faces (`backend/app/core/onnx_runtime.py`, `pipeline.py`)**:
  Stacks multiple face crops into a single NCHW tensor via `cv2.dnn.blobFromImages` (`blob_from_arcface_crops` at `112×112` and `blob_from_mobilenet_crops` at `224×224`) for single-pass batched inference across recognition, emotion, and age models.
- **ONNX Runtime with CPU Vectorization (`backend/app/core/onnx_runtime.py`)**:
  Configures `onnxruntime.InferenceSession` with prioritized `OpenVINOExecutionProvider` and `CPUExecutionProvider`, `ORT_ENABLE_ALL` graph optimizations, CPU memory arena, and intra-op thread pooling for AVX2/AVX-512 acceleration.

---

## Iteration 4 — age-range presentation (investigated an alternative model, didn't adopt it)

`age_estimation.py`'s `AgeResult` includes `estimated_age_range` (e.g. "25-32"), computed by bucketing the point estimate into standard age ranges (`tests/test_age_buckets.py`).

## Iteration 3 — frontend, webcam mode, evaluation harness

Added React + Vite + TypeScript dashboard (`frontend/`), live WebSocket webcam analysis (`backend/app/api/ws.py`), and consented real-face evaluation harness (`backend/tests/test_evaluation_real_faces.py`).

## Iteration 2 — bug fixes + full landmarks/alignment

Fixed float landmark coordinates in `cv2.circle()`, isolated per-instance `FaceTracker` ID counters, guaranteed temp-upload cleanup via FastAPI `Depends()`, wrapped `cv2.VideoCapture` in `try/finally`, and added 5-point landmark eye-line alignment (`recognition.align_face()`).

## Iteration 1 — initial build

Initial FastAPI backend, detection + Haar fallback, consent-gated recognition, expression classification, optional age estimation, IoU video tracking, automated test suite, and documentation.
