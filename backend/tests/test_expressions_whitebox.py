"""
WHITE-BOX testing of app/services/expressions.py -- zero dedicated test file
existed for this module before. Tests the confidence-threshold branching,
8-class vs 7-class label selection, and batch mechanics directly, using
knowledge of result_from_scores_dict's internal threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.expressions import (
    HSEMOTION_7_CLASSES,
    HSEMOTION_8_CLASSES,
    ExpressionResult,
    _result_from_probs,
    _softmax,
    classify_expression,
    classify_expressions_batch,
    result_from_scores_dict,
)


# --- result_from_scores_dict: the certain/uncertain threshold boundary ----

def test_score_just_above_threshold_is_certain():
    scores = {"happy": 0.23, "neutral": 0.11, "sad": 0.10, "angry": 0.56 / 10}
    # normalize so it's a believable distribution but keep happy as max > 0.22
    scores = {"happy": 0.23, "sad": 0.20, "angry": 0.19, "neutral": 0.18, "fear": 0.20}
    result = result_from_scores_dict(scores)
    assert result.certain is True
    assert result.label == "happy"


def test_score_just_below_threshold_is_uncertain():
    scores = {"happy": 0.21, "sad": 0.20, "angry": 0.19, "neutral": 0.20, "fear": 0.20}
    result = result_from_scores_dict(scores)
    assert result.certain is False
    assert result.label is None


def test_score_exactly_at_threshold_is_certain():
    """min_conf comparison is `< min_conf`, so exactly 0.22 should NOT be
    rejected -- pins down the boundary's exact inclusivity."""
    scores = {"happy": 0.22, "sad": 0.20, "angry": 0.19, "neutral": 0.19, "fear": 0.20}
    result = result_from_scores_dict(scores)
    assert result.certain is True


def test_uniform_distribution_is_uncertain():
    """8-way uniform chance (0.125 each) is well below the 0.22 threshold --
    the model genuinely doesn't know, and that must come through as
    'uncertain', not an arbitrarily-picked label."""
    scores = {label: 0.125 for label in HSEMOTION_8_CLASSES}
    result = result_from_scores_dict(scores)
    assert result.certain is False
    assert result.label is None
    assert result.to_dict()["label"] == "uncertain"


def test_result_to_dict_scores_are_rounded_and_labeled():
    scores = {"happy": 0.9, "sad": 0.1}
    result = result_from_scores_dict(scores)
    d = result.to_dict()
    assert d["label"] == "happy"
    assert d["certain"] is True
    assert d["scores"]["happy"] == 0.9
    assert "note" in d and isinstance(d["note"], str) and len(d["note"]) > 0


# --- _result_from_probs: 8-class vs 7-class label-set selection -----------

def test_8_element_probs_use_8_class_labels_including_contempt():
    probs = np.array([0.03, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02])  # index 1 = contempt
    result = _result_from_probs(probs)
    assert set(result.scores.keys()) == set(HSEMOTION_8_CLASSES)
    assert result.label == "contempt"


def test_7_element_probs_use_7_class_labels_without_contempt():
    probs = np.array([0.02, 0.9, 0.02, 0.02, 0.02, 0.01, 0.01])  # index 1 = disgust in 7-class order
    result = _result_from_probs(probs)
    assert set(result.scores.keys()) == set(HSEMOTION_7_CLASSES)
    assert "contempt" not in result.scores
    assert result.label == "disgust"


# --- _softmax -----------------------------------------------------------

def test_softmax_sums_to_one():
    logits = np.array([2.0, 1.0, 0.1, -1.0, 0.5, 3.0, -0.5, 0.0])
    probs = _softmax(logits)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert np.all(probs >= 0)


def test_softmax_is_numerically_stable_for_large_logits():
    """Without the max-subtraction trick this would overflow to inf/nan."""
    logits = np.array([1000.0, 999.0, 998.0, 1, 1, 1, 1, 1])
    probs = _softmax(logits)
    assert np.all(np.isfinite(probs))
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_softmax_batched_rows_are_independent():
    batch = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    probs = _softmax(batch)
    assert np.argmax(probs[0]) == 0
    assert np.argmax(probs[1]) == 2


# --- classify_expression(s)_batch: batching mechanics ----------------------

def test_classify_expressions_batch_empty_list():
    assert classify_expressions_batch([]) == []


def test_classify_expression_returns_well_formed_result():
    crop = np.full((100, 100, 3), 130, dtype=np.uint8)
    result = classify_expression(crop)
    assert isinstance(result, ExpressionResult)
    d = result.to_dict()
    assert set(d.keys()) == {"label", "certain", "scores", "note"}
    assert abs(sum(d["scores"].values()) - 1.0) < 0.01


def test_classify_expressions_batch_produces_independent_results_not_hardcoded():
    """Same regression shape as the age-ensemble 'not a constant' check:
    meaningfully different crops should (usually) not all collapse to the
    exact same scores dict, which would indicate the batch dimension is
    being silently collapsed/broadcast somewhere."""
    rng = np.random.default_rng(99)
    crops = [rng.integers(0, 255, (90, 90, 3), dtype=np.uint8) for _ in range(4)]
    results = classify_expressions_batch(crops)
    assert len(results) == 4
    score_tuples = [tuple(sorted(r.scores.items())) for r in results]
    assert len(set(score_tuples)) > 1, "every crop produced identical scores -- batch dim may be collapsed"


def test_classify_expressions_batch_matches_single_classify_for_one_crop():
    """Batch-of-1 and the single-image convenience wrapper must agree --
    they share the same underlying _embed-style batched path."""
    crop = np.full((100, 100, 3), 160, dtype=np.uint8)
    single = classify_expression(crop)
    batched = classify_expressions_batch([crop])[0]
    assert single.label == batched.label
    assert single.scores == pytest.approx(batched.scores)
