import pytest

from app.services.age_estimation import _bucket_for_age, AgeResult


@pytest.mark.parametrize(
    "age,expected_bucket",
    [
        (0, "0-2"), (2, "0-2"),
        (3, "4-6"), (6, "4-6"),
        (7, "8-12"), (12, "8-12"),
        (13, "15-20"), (20, "15-20"),
        (21, "25-32"), (32, "25-32"),
        (33, "38-43"), (43, "38-43"),
        (44, "48-53"), (53, "48-53"),
        (54, "60-100"), (100, "60-100"), (150, "60-100"),
    ],
)
def test_bucket_boundaries(age, expected_bucket):
    assert _bucket_for_age(age) == expected_bucket


def test_buckets_cover_every_age_with_no_gaps():
    # Every integer age from 0 to 120 must land in exactly one bucket —
    # guards against an off-by-one leaving a gap between boundaries.
    for age in range(0, 121):
        assert _bucket_for_age(age) is not None


def test_age_result_includes_range_when_enabled():
    result = AgeResult(estimated_age=27, enabled=True, estimated_age_range="25-32")
    d = result.to_dict()
    assert d["estimated_age"] == 27
    assert d["estimated_age_range"] == "25-32"


def test_age_result_range_is_none_when_disabled():
    result = AgeResult(estimated_age=None, enabled=False)
    d = result.to_dict()
    assert d["estimated_age_range"] is None
