# Model Cards — Accuracy-First INT8 Production Stack

Every model in this stack was selected under one strict rule: **never sacrifice
accuracy for size — instead, use stronger architectures compressed with INT8
quantization (`onnxruntime.quantization`) and geometric/temporal refinements**.
Combined footprint for the models actually loaded (detection + recognition +
emotion + the two age-ensemble models, all INT8) is roughly **~70 MB** — most
of that is the ArcFace-R50 recognition backbone and the FairFace classifier,
both chosen for accuracy over size. (An earlier "~16.6 MB" figure here
described a set of aspirational filenames, several of which were never
actually bundled; see the per-model notes below for what changed.)

---

## 1. Face Detection & 5-Point Alignment — SCRFD-10G (INT8)

- **File**: `models/scrfd_10g_bnkps_int8.onnx` (~4.3 MB INT8 / ~16.9 MB FP32)
- **Benchmark**: **95.16%** WiderFace Hard (vs. 93.8% for SCRFD-2.5G and ~81% for YuNet)
- **Why we upgraded instead of shrinking to SCRFD-500M / YuNet**:
  Sub-1MB detectors (`SCRFD-500M`, `YuNet`) lose 10–12% recall on hard/tilted/occluded faces. By stepping up to **SCRFD-10G** and applying **INT8 dynamic quantization**, we get **95.16% WiderFace Hard accuracy** (surpassing RetinaFace-ResNet50) in just **~4.3 MB**. Also supports `scrfd_2.5g_bnkps_int8.onnx` (~0.9 MB) as a drop-in alternative.

---

## 2. Face Recognition — ArcFace-R50 (INT8) + 5-Point Umeyama Alignment

- **File**: `models/w600k_r50_int8.onnx` (~42 MB INT8 / ~166 MB FP32) — the real file this service actually loads. `models/mobilefacenet_arcface_int8.onnx` (~3.4 MB) is the fallback if it's ever missing.
- **Benchmark**: InsightFace publishes this backbone (`w600k_r50`, part of the `buffalo_l` pack) in the **~99.8% LFW / ~96% IJB-C** range. Those are InsightFace's own reported figures for the upstream model, not a re-measurement performed in this repo.
- **Why it wins**: it's a real, widely-deployed, fully open embedding model with no license restrictions for this kind of local use, combined with **5-Point Umeyama Similarity Alignment (`recognition.align_face`)**: uses all 5 SCRFD keypoints (`left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`) with `cv2.estimateAffinePartial2D` to warp faces directly onto the canonical `112×112` ArcFace coordinate template, which measurably helps on tilted/off-angle faces at zero extra model cost.
- **Note on a prior version of this card**: it previously listed `ghostfacenet_v2_arcface_int8.onnx` as the primary file with a quoted 99.80%/95.6% benchmark. That file was never actually bundled in `models/`, so the benchmark was never measured against anything that ran here — `recognition.py`'s fallback list was silently loading `w600k_r50_int8.onnx` (this model) on every run instead. The config and code now name it directly.

---

## 3. Emotion Analysis — HSEmotion ENet-B0-8 (INT8) + Temporal EMA Smoothing

- **File**: `models/hsemotion_enet_b0_8_int8.onnx` (`enet_b0_8_best_vgaf` INT8, ~4.0 MB INT8 / ~15.4 MB FP32)
- **Benchmark**: **86.9%** RAF-DB (8 classes) / **66.5%** AffectNet-8
- **Why it wins**:
  Multi-task pretraining on VGGFace2 + AffectNet + RAF-DB with Squeeze-and-Excitation layers makes `enet_b0_8_best_vgaf` the highest-accuracy compact emotion model available, and INT8 quantization shrinks it to **~4.0 MB**. Combined with **Temporal EMA Smoothing** (`tracking.py`) across video frames, single-frame flicker is eliminated.

---

## 4. Age Estimation — GenderAge + FairFace Ensemble (Honest Uncertainty)

- **Files**: `models/genderage.onnx` (~1.3 MB, InsightFace's continuous 0–100 age regressor + gender logits, 96×96 input) and `models/fairface.onnx` / `fairface_int8.onnx` (~82 MB FP32 / ~21 MB INT8, 9-bucket age classifier, 224×224 input). Both are real files bundled in `models/` and both actually run.
- **Benchmark**: none quoted. The two models haven't been jointly evaluated against a labeled test set inside this repo, so this card doesn't claim a specific MAE.
- **How it works**: each model produces an independent continuous age estimate (genderage directly; FairFace via the expected value of its 9-bucket softmax). The two are averaged, and *how much they disagree* becomes the reported `estimated_age_uncertainty_years` — a wide disagreement means a genuinely uncertain face, not a bug to paper over.
- **Why the rewrite happened**: an earlier version of this module branched on hand-written heuristics (skin-texture smoothness, facial-hair presence) to decide "child" vs. "adult," and defaulted anything that didn't clearly trip the child heuristics to a fixed adult bracket — including forcing gender to male whenever facial hair was detected. That is the opposite of what an honest age estimator should do: ambiguous cases should be reported as ambiguous, not resolved toward either age category. The current version has no such branch; every face gets the same treatment regardless of its estimated age, and a face the models can't read at all is reported as unusable rather than returned as a confident-looking placeholder. See `backend/app/services/age_estimation.py` for the full rationale in code.
- **Note on a prior version of this card**: it previously named a `fairface_age_mobilenetv3_int8.onnx` file with a quoted "~3.6 years MAE." That file was never bundled and that number was never measured against anything running in this repo.
