"""
WHITE-BOX testing of app/core/onnx_runtime.py: every test here calls an
internal function directly and asserts on its exact return shape/type,
using knowledge of the implementation (not just the public API contract).
This file exists because onnx_runtime.py had zero dedicated tests despite
being the shared foundation every model-loading service depends on --
test_balanced_stack.py covers the blob-stacking happy path, but not the
failure/edge branches exercised here.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.onnx_runtime import (
    blob_from_arcface_crops,
    blob_from_mobilenet_crops,
    create_session_options,
    get_active_execution_providers,
    load_onnx_session,
    maybe_quantize_to_int8,
    prepare_valid_crops,
    resolve_model_path,
)


# --- resolve_model_path / load_onnx_session (missing-file branches) -------

def test_resolve_model_path_returns_none_when_nothing_matches():
    result = resolve_model_path(
        "definitely_does_not_exist.onnx",
        candidate_names=("also_missing.onnx", "still_missing.onnx"),
    )
    assert result is None


def test_load_onnx_session_returns_all_none_when_model_missing():
    session, path, provider = load_onnx_session(
        "nonexistent_model_xyz.onnx", candidate_names=("also_nowhere.onnx",)
    )
    assert session is None
    assert path is None
    assert provider is None


def test_resolve_model_path_finds_a_real_bundled_model():
    """Positive control for the two tests above: confirms the search
    mechanism itself works, not just that it fails closed."""
    result = resolve_model_path("scrfd_10g_bnkps_int8.onnx")
    assert result is not None
    assert result.is_file()


def test_maybe_quantize_to_int8_is_a_noop_for_already_int8_names():
    from pathlib import Path

    already_int8 = Path("models/whatever_int8.onnx")
    assert maybe_quantize_to_int8(already_int8) == already_int8


# --- prepare_valid_crops (the sanitization layer every blob fn relies on) -

def test_prepare_valid_crops_replaces_none_with_zero_fallback():
    cleaned = prepare_valid_crops([None], fallback_size=(50, 60))
    assert len(cleaned) == 1
    assert cleaned[0].shape == (60, 50, 3)  # (h, w, 3) from (w, h) fallback_size
    assert cleaned[0].dtype == np.uint8
    assert np.all(cleaned[0] == 0)


def test_prepare_valid_crops_replaces_empty_array_with_zero_fallback():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    cleaned = prepare_valid_crops([empty], fallback_size=(40, 40))
    assert cleaned[0].shape == (40, 40, 3)


def test_prepare_valid_crops_converts_grayscale_to_bgr():
    gray = np.full((80, 80), 100, dtype=np.uint8)
    cleaned = prepare_valid_crops([gray])
    assert cleaned[0].ndim == 3
    assert cleaned[0].shape[2] == 3


def test_prepare_valid_crops_passes_through_valid_bgr_unchanged_in_shape():
    valid = np.full((100, 90, 3), 77, dtype=np.uint8)
    cleaned = prepare_valid_crops([valid])
    assert cleaned[0].shape == valid.shape
    assert np.all(cleaned[0] == 77)


def test_prepare_valid_crops_handles_mixed_batch():
    """A batch combining every edge case at once -- the realistic scenario
    when one face in a multi-face frame has a bad crop and the others
    don't; the batch must not fail as a whole."""
    batch = [
        None,
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.full((70, 70), 50, dtype=np.uint8),  # grayscale
        np.full((70, 70, 3), 90, dtype=np.uint8),  # normal
    ]
    cleaned = prepare_valid_crops(batch, fallback_size=(70, 70))
    assert len(cleaned) == 4
    for c in cleaned:
        assert c.shape == (70, 70, 3)
        assert c.dtype == np.uint8


# --- blob_from_arcface_crops / blob_from_mobilenet_crops ------------------

def test_blob_from_arcface_crops_empty_list_returns_correctly_shaped_empty_array():
    blob = blob_from_arcface_crops([], size=(112, 112))
    assert blob.shape == (0, 3, 112, 112)
    assert blob.dtype == np.float32


def test_blob_from_mobilenet_crops_empty_list_returns_correctly_shaped_empty_array():
    blob = blob_from_mobilenet_crops([], size=(224, 224))
    assert blob.shape == (0, 3, 224, 224)
    assert blob.dtype == np.float32


def test_blob_from_arcface_crops_normalizes_to_expected_range():
    """ArcFace normalization is (pixel - 127.5) / 127.5 -- a mid-gray input
    (127 or 128) should land very close to 0, and pure white/black should
    land close to +1/-1."""
    mid_gray = np.full((50, 50, 3), 127, dtype=np.uint8)
    blob = blob_from_arcface_crops([mid_gray], size=(112, 112))
    assert blob.shape == (1, 3, 112, 112)
    assert abs(float(blob.mean())) < 0.05

    white = np.full((50, 50, 3), 255, dtype=np.uint8)
    blob_white = blob_from_arcface_crops([white], size=(112, 112))
    assert float(blob_white.mean()) > 0.9


def test_blob_from_arcface_crops_handles_none_in_batch_without_raising():
    blob = blob_from_arcface_crops([None, np.full((60, 60, 3), 50, dtype=np.uint8)], size=(112, 112))
    assert blob.shape == (2, 3, 112, 112)


def test_blob_from_arcface_and_mobilenet_use_different_normalizations():
    """Regression guard: these two must NOT share normalization constants --
    ArcFace expects [-1, 1]-ish, FairFace/HSEmotion expect ImageNet-normalized
    [0,1]-then-standardized. If someone accidentally merged these helpers,
    this is what would silently break."""
    crop = np.full((80, 80, 3), 200, dtype=np.uint8)
    arcface_blob = blob_from_arcface_crops([crop], size=(112, 112))
    mobilenet_blob = blob_from_mobilenet_crops([crop], size=(224, 224))
    # Same input, meaningfully different output distributions.
    assert abs(float(arcface_blob.mean()) - float(mobilenet_blob.mean())) > 0.1


# --- provider / session-options plumbing -----------------------------------

def test_get_active_execution_providers_never_returns_empty():
    providers = get_active_execution_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1
    assert "CPUExecutionProvider" in providers or len(providers) >= 1


def test_create_session_options_returns_configured_object():
    import onnxruntime as ort

    opts = create_session_options()
    assert isinstance(opts, ort.SessionOptions)
    assert opts.intra_op_num_threads >= 1
