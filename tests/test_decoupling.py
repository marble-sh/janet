from __future__ import annotations

import pytest

from janet.loop.decoupling import Decoupling, _half_gain, _ranks, detect, spearman


def test_ranks_treat_float_noise_as_tied() -> None:
    ranks = _ranks([1.0, 1.0 + 1e-16, 1.0, 1.0])
    assert ranks == [2.5, 2.5, 2.5, 2.5]


def test_ranks_handles_ties_and_order() -> None:
    assert _ranks([3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]


def test_spearman_known_and_degenerate() -> None:
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([1.0, 2.0], [1.0, 2.0]) is None
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None


def test_half_gain_splits_the_series() -> None:
    assert _half_gain([0.0, 0.0, 1.0, 1.0]) == 1.0
    assert _half_gain([1.0, 1.0, 0.5, 0.5]) == -0.5


def test_detect_needs_the_minimum_points() -> None:
    result = detect([(0.5, 0.5), (0.6, 0.5), (0.7, 0.5)])
    assert not result.flagged
    assert "need 4 points" in result.reason


def test_detect_flags_a_rising_dev_with_a_flat_proxy() -> None:
    result = detect([(0.4, 0.5), (0.4, 0.5), (0.6, 0.5), (0.7, 0.5)])
    assert result.flagged
    assert "dev rose" in result.reason
    assert result.dev_gain > 0


def test_detect_accepts_a_rising_pair() -> None:
    result = detect([(0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7)])
    assert not result.flagged
    assert result.rank_corr == pytest.approx(1.0)


def test_detect_flags_anti_correlation() -> None:
    # Dev has no net gain, so only the rank-correlation branch can fire.
    result = detect([(0.4, 0.8), (0.7, 0.5), (0.4, 0.8), (0.7, 0.5)])
    assert result.flagged
    assert "anti-correlated" in result.reason


def test_detect_ignores_a_constant_dev_proxy() -> None:
    # This is the real P4 pattern: dev is saturated at the same score to within
    # float noise while held-out moves. That is not evidence of decoupling; the
    # detector must not turn on strict mode for it.
    points = [
        (0.9333333333333333, 0.75),
        (0.9333333333333332, 0.7444444444444445),
        (0.9333333333333333, 0.5944444444444444),
        (0.9333333333333333, 0.6222222222222222),
    ]
    result = detect(points)
    assert not result.flagged
    assert result.dev_gain == 0.0
    assert result.rank_corr is None


def test_decoupling_payload_is_json_safe() -> None:
    result = Decoupling(4, 0.1, 0.2, 0.5, False, "ok")
    payload = result.to_dict()
    assert payload["dev_gain"] == 0.1
    assert payload["flagged"] is False
    assert payload["rank_corr"] == 0.5
