#!/usr/bin/env python3
"""
Reproducible Performance Benchmark Suite
Face Intelligence System

Run: python benchmark.py --output reports/benchmarks/benchmark_$(date +%Y%m%d_%H%M%S).json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Dict

import cv2
import numpy as np

# Ensure we can import the app
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.onnx_runtime import get_active_execution_providers
from app.services import detection, recognition, expressions, age_estimation, pipeline, tracking


@dataclass
class BenchmarkEnvironment:
    """Capture full environment for reproducibility."""
    timestamp: str
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    cpu_model: str
    opencv_version: str
    onnxruntime_version: str
    numpy_version: str
    execution_providers: List[str]
    model_files: Dict[str, str]  # name -> sha256
    settings_snapshot: Dict[str, Any]


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""
    count: int
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def get_cpu_model() -> str:
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get('brand_raw', 'unknown')
    except Exception:
        return platform.processor() or 'unknown'


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]  # Short hash for readability


def collect_environment() -> BenchmarkEnvironment:
    import onnxruntime as ort
    import numpy as np

    model_dir = Path(__file__).parent.parent / 'models'
    model_files = {}
    for model_file in model_dir.glob('*.onnx'):
        model_files[model_file.name] = file_sha256(model_file)

    return BenchmarkEnvironment(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        cpu_model=get_cpu_model(),
        opencv_version=cv2.__version__,
        onnxruntime_version=ort.__version__,
        numpy_version=np.__version__,
        execution_providers=get_active_execution_providers(),
        model_files=model_files,
        settings_snapshot={
            'detection_interval': settings.detection_interval,
            'temporal_ema_alpha': settings.temporal_ema_alpha,
            'batch_inference_enabled': settings.batch_inference_enabled,
            'onnx_intra_op_threads': settings.onnx_intra_op_threads,
            'recognition_distance_threshold': settings.recognition_distance_threshold,
            'detection_confidence_threshold': settings.detection_confidence_threshold,
        }
    )


def make_test_image(width: int = 640, height: int = 480, face_count: int = 1) -> np.ndarray:
    """Create a synthetic test image with face-like regions."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add some face-like oval regions
    for i in range(face_count):
        cx = width // (face_count + 1) * (i + 1)
        cy = height // 2
        w = min(120, width // 4)
        h = min(160, height // 3)
        cv2.ellipse(img, (cx, cy), (w//2, h//2), 0, 0, 360, (180, 160, 140), -1)
        # Add eye-like dots
        cv2.circle(img, (cx - w//4, cy - h//6), 5, (50, 50, 50), -1)
        cv2.circle(img, (cx + w//4, cy - h//6), 5, (50, 50, 50), -1)
    return img


def benchmark_cold_start(iterations: int = 3) -> List[float]:
    """Measure cold start time (import + model loading)."""
    times = []
    for _ in range(iterations):
        # Fresh import each time (simulate cold start)
        start = time.perf_counter()
        import importlib
        import app.services.detection
        import app.services.recognition
        import app.services.expressions
        import app.services.age_estimation
        importlib.reload(app.services.detection)
        importlib.reload(app.services.recognition)
        importlib.reload(app.services.expressions)
        importlib.reload(app.services.age_estimation)
        # Trigger model loading
        test_img = make_test_image()
        detection.detect_faces(test_img)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_detection(image: np.ndarray, iterations: int = 50, warmup: int = 5) -> List[float]:
    """Benchmark face detection latency."""
    # Warmup
    for _ in range(warmup):
        detection.detect_faces(image)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detection.detect_faces(image)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_recognition(image: np.ndarray, iterations: int = 50, warmup: int = 5) -> List[float]:
    """Benchmark recognition (detect + align + embed)."""
    # Warmup
    for _ in range(warmup):
        faces, _ = detection.detect_faces(image)
        if faces:
            aligned = recognition.align_face(image, faces[0])
            recognition._embed(aligned)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        faces, _ = detection.detect_faces(image)
        if faces:
            aligned = recognition.align_face(image, faces[0])
            recognition._embed(aligned)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_expressions(image: np.ndarray, iterations: int = 50, warmup: int = 5) -> List[float]:
    """Benchmark expression classification."""
    faces, _ = detection.detect_faces(image)
    if not faces:
        return [0.0]

    crops = [f.crop_bgr for f in faces]

    # Warmup
    for _ in range(warmup):
        expressions.classify_expressions_batch(crops)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        expressions.classify_expressions_batch(crops)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_age_estimation(image: np.ndarray, iterations: int = 30, warmup: int = 3) -> List[float]:
    """Benchmark age estimation."""
    faces, _ = detection.detect_faces(image)
    if not faces:
        return [0.0]

    boxes = [(f.x, f.y, f.w, f.h) for f in faces]

    # Warmup
    for _ in range(warmup):
        age_estimation.estimate_ages_batch_in_frame(image, boxes)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        age_estimation.estimate_ages_batch_in_frame(image, boxes)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_full_pipeline(image: np.ndarray, iterations: int = 30, warmup: int = 3) -> List[float]:
    """Benchmark complete analysis pipeline."""
    options = pipeline.AnalysisOptions(
        run_recognition=True,
        run_expression=True,
        run_age=True,
    )

    # Warmup
    for _ in range(warmup):
        pipeline.analyze_image(image, options)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pipeline.analyze_image(image, options)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_tracking(image: np.ndarray, iterations: int = 100, warmup: int = 10) -> List[float]:
    """Benchmark tracking (optical flow) vs full detection."""
    tracker = tracking.FaceTracker()
    options = pipeline.AnalysisOptions(
        run_recognition=False,
        run_expression=False,
        run_age=False,
        tracker=tracker,
        frame_index=0,
        decouple_detection=True,
    )

    # Warmup - establish tracks
    for i in range(warmup):
        opts = pipeline.AnalysisOptions(
            run_recognition=False, run_expression=False, run_age=False,
            tracker=tracker, frame_index=i, decouple_detection=True
        )
        pipeline.analyze_image(image, opts)

    # Benchmark inter-frame tracking (no detection)
    times = []
    for i in range(warmup, warmup + iterations):
        opts = pipeline.AnalysisOptions(
            run_recognition=False, run_expression=False, run_age=False,
            tracker=tracker, frame_index=i, decouple_detection=True
        )
        start = time.perf_counter()
        pipeline.analyze_image(image, opts)
        times.append((time.perf_counter() - start) * 1000)
    return times


def benchmark_concurrent(requests: int = 20, concurrency: int = 4) -> List[float]:
    """Benchmark concurrent request handling."""
    import concurrent.futures

    image = make_test_image()
    options = pipeline.AnalysisOptions(run_recognition=True, run_expression=True, run_age=False)

    def single_request():
        start = time.perf_counter()
        pipeline.analyze_image(image, options)
        return (time.perf_counter() - start) * 1000

    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(requests)]
        for f in concurrent.futures.as_completed(futures):
            times.append(f.result())
    return times


def benchmark_batch_scaling(image: np.ndarray, max_faces: int = 5) -> Dict[int, float]:
    """Benchmark how latency scales with number of faces in batch."""
    results = {}
    for n in range(1, max_faces + 1):
        img = make_test_image(face_count=n)
        options = pipeline.AnalysisOptions(run_recognition=True, run_expression=True, run_age=True)

        # Warmup
        for _ in range(3):
            pipeline.analyze_image(img, options)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            pipeline.analyze_image(img, options)
            times.append((time.perf_counter() - start) * 1000)

        results[n] = statistics.median(times)
    return results


def compute_stats(times: List[float]) -> LatencyStats:
    """Compute statistical summary."""
    if not times:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    sorted_times = sorted(times)
    count = len(sorted_times)

    return LatencyStats(
        count=count,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        stdev_ms=statistics.stdev(times) if count > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
        p50_ms=sorted_times[int(count * 0.50)],
        p95_ms=sorted_times[int(count * 0.95)] if count > 20 else max(times),
        p99_ms=sorted_times[int(count * 0.99)] if count > 100 else max(times),
    )


def run_all_benchmarks() -> Dict[str, Any]:
    """Execute complete benchmark suite."""
    print("=" * 60)
    print("FACE INTELLIGENCE SYSTEM - PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Collect environment
    print("\n[1/10] Collecting environment information...")
    env = collect_environment()
    print(f"       CPU: {env.cpu_model} ({env.cpu_count} cores)")
    print(f"       Providers: {env.execution_providers}")
    print(f"       Models: {len(env.model_files)} files")

    # Test images
    single_face = make_test_image(640, 480, face_count=1)
    multi_face = make_test_image(1280, 720, face_count=3)

    results = {
        'environment': asdict(env),
        'benchmarks': {}
    }

    # Cold start
    print("\n[2/10] Benchmarking cold start...")
    times = benchmark_cold_start(3)
    stats = compute_stats(times)
    results['benchmarks']['cold_start_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms")

    # Detection
    print("\n[3/10] Benchmarking face detection...")
    times = benchmark_detection(single_face, 50, 5)
    stats = compute_stats(times)
    results['benchmarks']['detection_single_face_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Recognition
    print("\n[4/10] Benchmarking recognition (detect+align+embed)...")
    times = benchmark_recognition(single_face, 50, 5)
    stats = compute_stats(times)
    results['benchmarks']['recognition_single_face_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Expressions
    print("\n[5/10] Benchmarking expression classification...")
    times = benchmark_expressions(single_face, 50, 5)
    stats = compute_stats(times)
    results['benchmarks']['expressions_single_face_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Age estimation
    print("\n[6/10] Benchmarking age estimation...")
    times = benchmark_age_estimation(single_face, 30, 3)
    stats = compute_stats(times)
    results['benchmarks']['age_estimation_single_face_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Full pipeline
    print("\n[7/10] Benchmarking full pipeline (all tasks)...")
    times = benchmark_full_pipeline(single_face, 30, 3)
    stats = compute_stats(times)
    results['benchmarks']['full_pipeline_single_face_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Tracking (inter-frame)
    print("\n[8/10] Benchmarking tracking (optical flow, no detection)...")
    times = benchmark_tracking(single_face, 100, 10)
    stats = compute_stats(times)
    results['benchmarks']['tracking_inter_frame_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Concurrent
    print("\n[9/10] Benchmarking concurrent requests...")
    times = benchmark_concurrent(20, 4)
    stats = compute_stats(times)
    results['benchmarks']['concurrent_20req_4workers_ms'] = asdict(stats)
    print(f"       Median: {stats.median_ms:.1f}ms (p95: {stats.p95_ms:.1f}ms)")

    # Batch scaling
    print("\n[10/10] Benchmarking batch scaling (1-5 faces)...")
    scaling = benchmark_batch_scaling(multi_face, 5)
    results['benchmarks']['batch_scaling_ms_per_face_count'] = scaling
    for n, ms in scaling.items():
        print(f"       {n} face(s): {ms:.1f}ms median")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Face Intelligence Benchmark Suite')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    parser.add_argument('--quick', '-q', action='store_true', help='Run reduced benchmark set')
    args = parser.parse_args()

    results = run_all_benchmarks()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()