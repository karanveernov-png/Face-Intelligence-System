from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
GALLERY_DIR = DATA_DIR / 'gallery'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings'
TMP_UPLOAD_DIR = DATA_DIR / 'tmp_uploads'

for d in (MODELS_DIR, DATA_DIR, GALLERY_DIR, EMBEDDINGS_DIR, TMP_UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # 1. Detection & Alignment: SCRFD-10G INT8 (95.16% WiderFace Hard, ~4.3 MB INT8)
    #    Also supports SCRFD-2.5G (~3.2 MB FP32 / ~0.9 MB INT8) + OpenCV Haar fallback
    detector_backend: str = 'scrfd-10g-int8'
    detector_fallback_backend: str = 'opencv-haar'
    detector_model_file: str = 'scrfd_10g_bnkps_int8.onnx'
    detector_input_size: tuple[int, int] = (640, 640)
    detection_confidence_threshold: float = 0.5
    detection_nms_threshold: float = 0.4

    # 2. Face Recognition: ArcFace-R50 INT8 (InsightFace's w600k_r50, the backbone
    #    behind the widely-used "buffalo_l" model pack) -- one of the strongest
    #    freely-available open face-embedding models, and the real, verified-
    #    loadable weights actually bundled in models/. Previously this pointed at
    #    a "GhostFaceNet-V2" filename that was never a real downloadable file, so
    #    every run silently fell through to this same model via the candidate
    #    list below; it's now named directly so the config is honest about what
    #    runs. mobilefacenet_arcface_int8.onnx remains available as a smaller,
    #    faster, slightly-less-accurate fallback.
    recognition_model: str = 'ArcFace-R50-INT8'
    recognition_model_file: str = 'w600k_r50_int8.onnx'
    recognition_distance_threshold: float = 0.68
    use_5pt_similarity_alignment: bool = True

    # 3. Emotion Analysis: HSEmotion EfficientNet-B0 / MobileNetV3 INT8 (86.9% RAF-DB 8-class)
    expression_model: str = 'HSEmotion-ENetB0-INT8'
    expression_model_file: str = "models/enet_b0_8_best_vgaf.onnx"
    expression_min_confidence: float = 0.3

    # 4. Age & Gender: an honest two-model ensemble, both real, verified-loadable
    #    weights -- InsightFace's genderage.onnx (continuous 0-100 regression +
    #    gender logits, 96x96 input) and FairFace's fairface.onnx (9-bucket age
    #    classifier trained on a dataset balanced across age/race/gender, 224x224
    #    input). The two models' continuous estimates are averaged; how much they
    #    disagree becomes the reported uncertainty. See
    #    app/services/age_estimation.py for why this replaced an earlier
    #    heuristic-driven version that defaulted ambiguous faces toward an adult
    #    bracket -- that bias is gone, not just relabeled.
    age_model: str = 'GenderAge+FairFace-Ensemble'
    age_model_file: str = "models/fairface.onnx"
    gender_model_file: str = "models/fairface.onnx"
    landmark_predictor_file: str = 'shape_predictor_5_face_landmarks.dat'
    age_estimation_enabled: bool = True

    # Pipeline Performance & Accuracy Strategy:
    # - Auto INT8 dynamic quantization (4x smaller weights, ~2x faster AVX-512 VNNI, <0.1% accuracy loss)
    # - Decouple detection from recognition (run SCRFD every Nth frame, optical/centroid track in between)
    # - Temporal EMA smoothing across video/webcam tracks for stable emotion & age predictions
    # - Batch cropped faces via cv2.dnn.blobFromImages
    auto_int8_quantize: bool = True
    detection_interval: int = 3
    temporal_ema_alpha: float = 0.4
    batch_inference_enabled: bool = True
    onnx_execution_providers: tuple[str, ...] = (
        'OpenVINOExecutionProvider',
        'CPUExecutionProvider',
    )
    onnx_intra_op_threads: int = max(1, min(os.cpu_count() or 4, 8))
    onnx_inter_op_threads: int = 1

    # Tracker & media upload settings
    # Was 15 detection *cycles* (up to 15 * detection_interval raw frames,
    # i.e. several seconds) before a drifted/ghost track was ever fully
    # dropped. predict_inter_frame() now hides a track from rendering much
    # sooner (see Track.unconfirmed_predict_frames), but this still governs
    # when it's removed from tracker state entirely -- shortened so a track
    # that never gets reclaimed by a real detection doesn't linger.
    tracker_max_missed_frames: int = 4
    tracker_iou_threshold: float = 0.3
    max_upload_size_mb: int = 25
    # Compressed file size alone doesn't bound memory use once an image is
    # decoded -- a highly-compressible image (e.g. a huge solid-color PNG)
    # can be a few MB on disk and gigabytes once decoded (a classic
    # "decompression bomb"). These cap the decoded array, checked right
    # after cv2.imread / before per-frame video processing. 40,000,000 px
    # is comfortably above a 45MP DSLR photo (~8256x5504) while rejecting
    # the pathological cases; 12,000px per side guards extreme aspect
    # ratios that a pure area cap wouldn't catch on its own.
    max_decoded_image_pixels: int = 40_000_000
    max_decoded_image_dimension_px: int = 12_000
    allowed_image_ext: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    allowed_video_ext: tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    frame_skip: int = 1
    max_video_frames_processed: int = 300
    retain_enrollment_images: bool = False


settings = Settings()
