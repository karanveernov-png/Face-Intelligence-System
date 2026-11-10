# Architecture — Balanced Production Stack & Pipeline Strategy

## End-to-End Data Flow

```
Upload (image/video) or Live Webcam Frame (/ws/analyze)
        |
        v
security/validation.py   -- extension + size check, safe temp filename, guaranteed cleanup
        |
        v
services/pipeline.py     -- Decoupled Detection Orchestrator:
        |
        +-- [Frame % N == 0] --> services/detection.py (SCRFD-2.5G ONNX, ~3.2 MB, 5-pt landmarks)
        |                        (with automatic fallback to bundled OpenCV Haar Cascade)
        |
        +-- [Frame % N != 0] --> services/tracking.py  (Lucas-Kanade Optical Flow + Centroid Tracker)
        |                        propagates bounding boxes & landmarks in <1 ms between SCRFD passes
        v
core/onnx_runtime.py     -- Batched Crop Vectorization (`cv2.dnn.blobFromImages`):
        |                   stacks all N detected/tracked face crops into contiguous NCHW tensors
        |                   and executes on ONNX Runtime (`OpenVINOExecutionProvider` / `CPUExecutionProvider`)
        |
        +--> services/recognition.py    -- MobileFaceNet-ArcFace (~8.0 MB) -> (N, 512) embeddings -> cosine match
        +--> services/expressions.py    -- HSEmotion MobileNet-V3 (~8.5 MB) -> (N, 8) RAF-DB probabilities
        +--> services/age_estimation.py -- FairFace MobileNet-V3 (~7.5 MB) -> (N,) expected age + range bucket
        |
        v
pipeline.PipelineResult  -- structured dict: per-face results + detector/tracker telemetry + task_errors
```

## 3 Pillars of Pipeline Performance

### 1. Decouple Detection from Recognition (`services/tracking.py` & `services/pipeline.py`)
Running a full-frame detector on every frame of a 30 FPS stream wastes CPU cycles when faces move only a few pixels between consecutive frames. In video (`/api/analyze/video`) and live webcam (`/ws/analyze`) modes:
- **Detection frames** (`frame_index % settings.detection_interval == 0`, or whenever no active tracks exist) execute **SCRFD-2.5G** to discover new faces and re-anchor existing tracks via IoU + centroid matching.
- **Intermediate frames** (`frame_index % settings.detection_interval != 0`) bypass SCRFD and call `FaceTracker.predict_inter_frame()`, which tracks interior face keypoints via pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) with smoothed centroid velocity fallback.

### 2. Batch Cropped Faces (`core/onnx_runtime.py`)
When multiple people are in view, invoking ONNX Runtime in a Python loop incurs per-call session overhead and prevents SIMD batch utilization. Instead:
- `blob_from_arcface_crops()` calls `cv2.dnn.blobFromImages` in C++ to resize, swap BGR→RGB, and normalize all aligned `112×112` crops into a single `(N, 3, 112, 112)` float32 tensor for **MobileFaceNet-ArcFace**.
- `blob_from_mobilenet_crops()` calls `cv2.dnn.blobFromImages` to stack all `224×224` face crops into a single `(N, 3, 224, 224)` float32 tensor with ImageNet normalization for **HSEmotion** and **FairFace Age**.

### 3. ONNX Runtime with CPU Vectorization (`core/onnx_runtime.py`)
All four models run through `onnxruntime.InferenceSession` configured with:
- Prioritized execution providers: `("OpenVINOExecutionProvider", "CPUExecutionProvider")`.
- `GraphOptimizationLevel.ORT_ENABLE_ALL`, CPU memory arena, memory pattern optimization, and multi-core intra-op thread pooling for AVX2 / AVX-512 hardware acceleration.
