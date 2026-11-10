# Troubleshooting

## Setup issues

**`ModuleNotFoundError: No module named 'app'` when running tests**
Run pytest from `backend/` with `PYTHONPATH` set to that directory:
```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests/ -v
```

**`ModuleNotFoundError: No module named 'deepface'` (or `fastapi`, `cv2`, ...)**
`pip install -r requirements.txt` wasn't run in the environment Python is
actually using. If you're using a virtual environment, confirm it's
activated (`(venv)` should show in your prompt) before installing.

**First request after a fresh install takes 30-90+ seconds**
This is expected and not a bug: the first call to each model downloads its
weight file (CenterFace ~7.5MB, Emotion ~6MB, ArcFace ~137MB, Age/Gender
~540MB each) from GitHub release assets into `~/.deepface/weights/` (or
`%USERPROFILE%\.deepface\weights\` on Windows). Every call after that uses
the cached file and is fast — see `docs/OPENCV_FUNDAMENTALS.md` section 15
for measured cached latency. If you want to pre-download everything before
a demo, run `GET /api/models/status` once ahead of time.

**`RuntimeError: The starlette.testclient module requires the httpx2 package`**
This project's pinned FastAPI/Starlette version needs both `httpx` and
`httpx2` for its test client, not just `httpx`. Both are in
`requirements.txt` — if you installed dependencies before this was added,
rerun `pip install -r requirements.txt`.

**Webcam not available / camera permission errors**
Webcam capture is not implemented yet in this build — see the top-level
`README.md`, "Next Phase". `cv2.VideoCapture(0)` is the function you'd call
to add it; the honest reason it isn't here yet is explained in the README
rather than silently stubbed out.

## Why there are no real face images in `samples/`

This was a deliberate decision, not an oversight. The master spec is
explicit that this project must never "use a public person's face or
private photographs without appropriate authorization", and separately
asks for tests to "clearly separate unit tests from evaluations that
require a consented real-image dataset". During the build, several
candidate sources of free, redistributable face photos were investigated:

- `opencv/opencv_zoo`'s own test fixtures — blocked by Git LFS in this
  build's sandboxed network environment (see `MODEL_CARDS.md`).
- `deepface`'s own `tests/dataset/` folder — the documented path
  (`tests/dataset/img1.jpg` etc., referenced in the project's own README
  and several published tutorials) returned 404 at build time, suggesting
  the repository's test fixtures have since moved or been reorganized.
- Well-known CV tutorial images of real, named people (e.g. the image
  commonly used in OpenCV's own official Haar Cascade tutorial) were
  deliberately **not** used, out of the same caution about using a real,
  identifiable person's photo without clear, traceable consent — even
  though OpenCV's own tutorials use it.

**What this means practically**: this project's automated test suite
(`backend/tests/`) validates every code path that doesn't require a real
face — error handling, the tracker, the API contract, the consent/
threshold/deletion logic in recognition (using a deterministic synthetic
image to prove "same input -> same embedding -> matches itself"), and
model-loading. It does **not** and cannot validate true-positive detection
accuracy end-to-end, because that requires an actual human face.

**To close this gap yourself**: add a photo you have the rights to use
(your own, or one with clear permission) to `samples/`, point
`POST /api/analyze/image` at it, and confirm a bounding box comes back with
reasonable coordinates. A good first test: a clear, front-facing, well-lit
photo should be detected by CenterFace with confidence well above the
default `0.5` threshold.

## Known limitations (honest list)

- No 68-point landmark output (only the 5 points CenterFace itself
  produces: both eyes, nose, both mouth corners — see `MODEL_CARDS.md`).
  A 68-point model (e.g. dlib's shape predictor) would need its own
  weights and wasn't pursued — see `CHANGELOG.md` for what *was* closed.
- No webcam streaming endpoint yet.
- No frontend yet.
- Tracker is IoU-only — no motion prediction, so a fast-moving face or a
  long occlusion can lose its track ID and start a new one.
- `/api/faces/enroll` requires exactly one face in the image; it doesn't
  yet support picking which face to enroll from a multi-face photo.
- True-positive detection/recognition accuracy has not been measured on a
  real face dataset in this build — see the section above.
