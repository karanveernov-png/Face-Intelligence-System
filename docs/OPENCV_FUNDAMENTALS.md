# OpenCV & Computer Vision Fundamentals — Learn As We Build

This is written so you can debug and extend this project yourself, not just
run it. Each section points at the actual file and line of code that uses
the concept, and ends with one small experiment you can run to see it for
yourself. Run experiments from `backend/` with `python3` after
`pip install -r requirements.txt`.

---

## 1. What an image is in computer memory

A digital image is just a grid of numbers. A 640×480 color photo is a 3-D
array: 480 rows, 640 columns, 3 numbers per pixel (one per color channel).
OpenCV represents this as a NumPy array.

```python
import cv2
img = cv2.imread("some_photo.jpg")
print(img.shape)   # (480, 640, 3) -> (height, width, channels)
print(img.dtype)   # uint8 -> each number is 0-255
```

**Experiment**: `print(img[0, 0])` — that's the blue/green/red values of
the single pixel in the top-left corner.

## 2. Pixels, dimensions, channels, and NumPy arrays

`img.shape[0]` is height (rows), `img.shape[1]` is width (columns) — easy
to mix up, and a classic source of bugs (`detection.py`'s
`image_bgr.shape[:2]` unpacks it as `h_img, w_img` specifically to keep
this straight). `img.shape[2]` is the channel count: 3 for color, 1 (or
just a 2-D array with no third dimension) for grayscale.

**Experiment**: `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` then compare
`gray.shape` (2-D) to `img.shape` (3-D).

## 3. BGR vs. RGB

OpenCV reads and stores color images as **Blue-Green-Red**, not the more
common Red-Green-Blue. This is a historical quirk of the library, and it is
*the* single most common bug source when mixing OpenCV with other tools
(matplotlib, PIL, most deep learning frameworks all expect RGB).

In this project: `detection.py`'s Haar-cascade path explicitly converts
`cv2.COLOR_BGR2GRAY` before detection (grayscale has no channel-order
issue). `expressions.py`'s emotion preprocessing also converts
`BGR2GRAY`. `recognition.py`'s ArcFace embedding deliberately does **not**
convert to RGB — this was verified by reading the actual preprocessing
code in the underlying library rather than assumed, and is documented in
`docs/MODEL_CARDS.md`, because guessing wrong here silently produces
embeddings that are internally consistent (so tests can still pass!) but
don't match the published accuracy of the original model.

**Experiment**: `cv2.imwrite("swapped.jpg", img[:, :, ::-1])` — this
reverses the channel order. Open the result; the colors will look wrong
(skin tones go blue) because you've turned BGR into "accidentally RGB
interpreted as BGR".

## 4. Reading and displaying images

`cv2.imread(path)` returns `None` — not an exception — if the file can't be
decoded. This project checks for that explicitly rather than letting a
`None` silently propagate and crash somewhere confusing later:

```python
# backend/app/api/routes.py
img = cv2.imread(str(path))
if img is None:
    raise HTTPException(status_code=400, detail="File could not be decoded as an image...")
```

**Experiment**: `cv2.imread("/tmp/nonexistent.jpg")` — confirm it returns
`None`, not an error.

## 5. Resizing, cropping, and normalization

*Resizing* changes pixel dimensions (`cv2.resize`). *Cropping* is just
NumPy array slicing — `image_bgr[y:y+h, x:x+w]` in `detection.py` is how a
face region is cut out of the full frame; no OpenCV function is even
needed for this, because a cropped region is just a sub-array.
*Normalization* rescales pixel values, usually into a range a neural
network was trained on. `recognition.py`'s `(x - 127.5) / 128` is a
normalization formula; see section 9 below for why it matters.

**Experiment**: `small = cv2.resize(img, (100, 100))` — note this ignores
aspect ratio and will distort a non-square image; `pipeline.py`'s
`DeepFace.analyze` internally uses a pad-to-preserve-aspect-ratio resize
for exactly this reason.

## 6. How a face detector works at a high level

Both detectors in this project share the same basic idea even though their
internals are very different:

- **Haar Cascade** (`detection._detect_haar`): slides a window of simple
  rectangular "is this region brighter/darker than that region" tests
  across the image at many scales. Cheap per-test, needs thousands of
  tests, but each one is fast — a 2001-era technique that still works
  today because it needs no GPU and no large model file.
- **CenterFace** (`detection._detect_centerface`): a small convolutional
  neural network that was trained on thousands of labeled face images to
  directly output "here's a heatmap of face centers, and here's the
  box/landmark offsets at each center" in a single forward pass. More
  accurate, especially off-angle, but needs the trained weight file.

## 7. Bounding boxes and confidence thresholds

A bounding box is `(x, y, w, h)` — top-left corner plus width/height (this
project's convention throughout; some libraries instead use two corner
points, `(x1, y1, x2, y2)` — always check which one you're working with).
A confidence threshold filters out low-certainty detections:
`settings.detection_confidence_threshold` (default `0.5`) in
`core/config.py`. Haar Cascade doesn't produce a real confidence score at
all (`detection.py` hardcodes `1.0` for it, with a comment explaining why)
— this is a real limitation of that algorithm, not an oversight.

**Experiment**: lower `detection_confidence_threshold` to `0.1` and
re-run analysis on a hard image (poor lighting, side angle) — you'll
likely see more boxes, including more false positives. Raise it to `0.9`
and watch real faces start getting dropped too. There is no threshold that
eliminates both false positives and false negatives simultaneously — that
trade-off is fundamental, not a bug to fix.

## 8. Facial landmarks and face alignment

Landmarks are specific, named points on a face (eye centers, nose tip,
mouth corners). *Alignment* uses landmarks to rotate/scale a face crop so
the eyes are level and at a known position before feeding it to a
recognition model — because ArcFace was trained on aligned faces, an
unaligned/tilted crop measurably hurts its accuracy.

`detection.py` extracts all 5 points CenterFace produces (both eyes, nose,
both mouth corners). `recognition.align_face()` uses the two eye points
to do the actual alignment:

1. Compute the angle between the eyes: `atan2(right_eye.y - left_eye.y,
   right_eye.x - left_eye.x)`. Level eyes → angle ≈ 0.
2. Build a rotation matrix around the eye-center point
   (`cv2.getRotationMatrix2D`) and warp the *whole frame* with it
   (`cv2.warpAffine`) — not just the crop, because the crop's pixels
   alone don't carry enough context to rotate cleanly without introducing
   black corners inside the face region.
3. Crop the same bounding box out of the *rotated* frame.

Step 3 is an approximation worth understanding, not just accepting: the
bounding box was computed on the *original*, unrotated image, so cutting
that exact box out of the rotated image is only exactly correct when the
rotation angle is small. For a near-frontal face (the common case) this
holds well; for a heavily tilted head it's a rougher approximation. This
is standard practice in lightweight alignment pipelines, and is documented
here rather than silently assumed to be pixel-perfect.

Both recognition endpoints (`/api/analyze/image`'s recognition task, and
`/api/faces/enroll`) apply the *same* alignment before embedding — this
matters because an embedding computed from an aligned crop is not directly
comparable to one from an unaligned crop; mismatched preprocessing between
enrollment and matching would silently degrade accuracy without raising
any error.

**Experiment**: run `tests/test_alignment.py` — in particular,
`test_align_with_level_eyes_is_a_near_noop_rotation` demonstrates that an
already-level pair of eyes produces a near-identical crop to a plain,
unaligned crop (as it should — there's nothing to correct), while
`test_align_with_tilted_eyes_produces_correctly_shaped_output` confirms a
genuinely tilted pair still produces a valid, correctly-shaped output.

## 9. Face embeddings and similarity comparison

An embedding is a neural network's compressed numeric summary of a face —
in this project, 512 floating-point numbers from ArcFace. Two photos of
the same person produce embeddings that are close together in that
512-dimensional space; two different people produce embeddings that are
farther apart. "Close" and "far" are measured with **cosine distance**
(`recognition._cosine_distance`): `1 - dot_product` of two unit-length
vectors. Dot product of unit vectors equals the cosine of the angle
between them — hence the name. Identical direction -> dot product 1 ->
distance 0. Perpendicular -> dot product 0 -> distance 1.

**Experiment**: run `test_cosine_distance_of_identical_unit_vectors_is_zero`
and `test_cosine_distance_of_orthogonal_unit_vectors_is_one` in
`tests/test_recognition.py` and read the two-line proof in the test itself.

## 10. Why recognition needs a threshold and an "Unknown" class

An embedding model always outputs *some* vector for *any* face, even one
it's never seen before — there's no built-in "I don't recognize this"
signal. A threshold on the distance to the *closest* enrolled identity is
what turns "closest match, whatever it is" into "a genuine match, or
nobody I know". `recognition.match_face()` implements exactly this: it
finds the closest enrolled identity, then only reports it if the distance
clears `settings.recognition_distance_threshold` — otherwise it returns
`None` (serialized as `"Unknown"`), never forcing the closest guess.

## 11. Facial-expression classification

See `docs/MODEL_CARDS.md`'s Emotion section for the model and its training
data. Mechanically, it's a small CNN that takes a 48×48 grayscale crop and
outputs 7 probabilities summing to 1 (a "softmax" output) — the same shape
of problem as the age/gender models, just with 7 categories of output
instead of 1 continuous number or 2 categories.

## 12. How pretrained models differ from training your own

Every model in this project is **pretrained** — someone else already ran
the (expensive, slow, data-hungry) training process and published the
resulting weight file. This project only ever *runs* those weights
(inference), never retrains them. Training would mean: gather thousands of
labeled examples, define a loss function, run gradient descent for hours/
days, and iterate. Inference, what this project does, means: load the
already-learned weights once (see `onnx_utils.build_session` inside the
`deepface` package, wrapped by this project's service modules) and reuse
that loaded model for every request — `services/detection.py`'s module-
level `_haar_detector` cache, and DeepFace's own internal model cache, both
exist specifically so a model is loaded from disk **once**, not on every
request (see section 15, performance).

## 13. Video capture, frames, and webcam handling

A video file (or webcam stream) is just a sequence of still images —
"frames" — read one at a time: `cv2.VideoCapture(path)`, then
`cap.read()` in a loop, which returns `(success_bool, frame_array)`.
`api/routes.py`'s `/api/analyze/video` does exactly this, with two
practical additions: `settings.frame_skip` (process every Nth frame,
because processing all 30 isn't necessary or fast enough — see section 15)
and `settings.max_video_frames_processed` (a hard cap so a huge or corrupt
video can't hang the request indefinitely).

## 14. Face tracking and why detection is not tracking

A detector has no memory — it looks at one frame and finds faces in
isolation, with no idea that "box at (10,10)" in frame 5 is the same
physical face as "box at (12,11)" in frame 6. **Tracking** is the separate
job of stitching per-frame detections into continuous identities over
time. `services/tracking.py`'s `FaceTracker` does this with Intersection-
over-Union (IoU): if a new frame's detection box overlaps enough with an
existing track's last-known box, it's considered the same track; if not,
a new track starts. See the module's docstring for the full algorithm and
why this simple approach (rather than a motion-predicting Kalman filter)
was the right first choice here.

**Experiment**: run `tests/test_tracking.py::test_same_face_across_frames_keeps_same_track_id`
and step through it — a face that moves slightly each frame keeps the
same `track_id`; see section "Next Phase" in the README for what breaks
this (fast motion, or a face that briefly leaves frame for too many
frames — `settings.tracker_max_missed_frames`).

## 15. CPU performance and optimization

This project runs on CPU only — no GPU required, per the master spec. Three
concrete techniques are used, all visible in the code rather than just
asserted here:

1. **Load once, reuse** (section 12): avoids the dominant cost of running
   a neural network, which is usually loading/compiling the model graph,
   not the actual arithmetic of one forward pass.
2. **Frame skipping** (section 13): analyzing every single video frame is
   rarely necessary — a face doesn't change expression or identity between
   two frames 33ms apart. `settings.frame_skip` is a direct, configurable
   knob for the latency/thoroughness trade-off.
3. **Measured, not claimed, latency**: every analysis result includes
   `total_latency_ms` (`pipeline.py`) and detection's own `latency_ms`
   (`detection.py`) — honest numbers from `time.time()` calls around the
   actual work, not an invented FPS figure. On the 1-CPU-core sandbox this
   was built and verified in, a single image with all three analysis
   tasks enabled took **roughly 1-1.2 seconds** (measured, cached-model,
   repeated runs) — i.e. this configuration is **not** real-time video
   analysis on modest hardware; it's "fast enough for interactive still-
   image analysis, and configurable frame-skipped video analysis".

## 16. Common errors and how to debug them

| Symptom | Likely cause | Where to look |
|---|---|---|
| `cv2.imread` returns `None` | Bad path, or truly corrupt/unsupported file | section 4 above |
| Detector finds nothing on a real photo | Wrong color order fed in, or threshold too high | sections 3, 7 |
| Recognition always says "Unknown" even for the enrolled person | Threshold too strict for your lighting/camera, or face wasn't aligned the same way at enroll vs. query time | section 8, `MODEL_CARDS.md` |
| `ModuleNotFoundError: deepface` | `pip install -r backend/requirements.txt` wasn't run, or wrong virtual environment active | `README.md` setup steps |
| First request is very slow, later ones are fast | First run is downloading model weights (hundreds of MB) from GitHub; cached afterward | section 12, 15 |
| Model status endpoint shows `"loaded": false` with an error | Read the `error` field directly — it's the real exception message, not a generic failure string | `api/routes.py`'s `/api/models/status` |
