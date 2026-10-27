from __future__ import annotations

import pytest

from janet.loop.decoupling import (
    CusumResult,
    Decoupling,
    DecouplingError,
    _half_gain,
    _ranks,
    assess,
    cusum,
    detect,
    spearman,
    stale_objective_probe,
)


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


def test_cusum_detects_a_step_rise() -> None:
    points = [(0.5, 0.5)] * 3 + [(0.56, 0.5)] * 3
    result = cusum(points)
    assert result.flagged
    assert result.first_flag == 4
    assert result.peak >= result.threshold


def test_cusum_ignores_a_stable_high_gap() -> None:
    # Dev is normally easier than the hidden anchor; a stable gap is not drift.
    result = cusum([(0.9, 0.7)] * 8)
    assert not result.flagged
    assert result.statistic == 0.0


def test_cusum_needs_the_minimum_points() -> None:
    result = cusum([(0.5, 0.5)] * 4)
    assert not result.flagged
    assert "need 5 points" in result.reason


def test_cusum_detects_a_gradual_shift() -> None:
    points = [(0.5, 0.5)] * 3 + [(0.53, 0.5)] * 5
    result = cusum(points)
    assert result.flagged
    assert result.first_flag == 6


def test_cusum_validates_parameters() -> None:
    with pytest.raises(DecouplingError, match="drift"):
        cusum([], drift=-0.1)
    with pytest.raises(DecouplingError, match="threshold"):
        cusum([], threshold=0.0)
    with pytest.raises(DecouplingError, match="baseline_points"):
        cusum([], baseline_points=0)
    with pytest.raises(DecouplingError, match="min_points"):
        cusum([], baseline_points=3, min_points=3)


def test_assess_runs_both_detectors() -> None:
    decoupling, drift = assess([(0.4, 0.5), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5)])
    assert decoupling.flagged
    assert drift.flagged
    assert isinstance(drift, CusumResult)


def test_stale_probe_detects_within_the_window() -> None:
    probe = stale_objective_probe(shift=0.06, noise=0.0)
    assert probe.detection_latency == 1
    assert probe.detection_index == probe.pre_points + 1
    assert probe.reframed


def test_stale_probe_reports_a_missing_shift() -> None:
    probe = stale_objective_probe(shift=0.0, noise=0.0)
    assert probe.detection_latency is None
    assert probe.detection_index is None
    assert not probe.reframed


def test_stale_probe_is_deterministic_with_noise() -> None:
    first = stale_objective_probe(shift=0.08, noise=0.01, seed=3)
    second = stale_objective_probe(shift=0.08, noise=0.01, seed=3)
    assert first == second
    assert first.reframed
    assert first.to_dict()["detection_latency"] == 1


def test_stale_probe_validates_inputs() -> None:
    with pytest.raises(DecouplingError, match="pre_points"):
        stale_objective_probe(pre_points=1)
    with pytest.raises(DecouplingError, match="noise"):
        stale_objective_probe(noise=-1.0)
