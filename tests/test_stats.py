from __future__ import annotations

import pytest

from janet.eval.stats import (
    AlphaSpending,
    StatsError,
    bootstrap_ci,
    certify,
    e_value,
    paired_test,
    sign_test_p,
)


def test_sign_test_known_values() -> None:
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert sign_test_p(4, 0) == pytest.approx(1 / 16)
    assert sign_test_p(5, 5) == pytest.approx(638 / 1024)
    assert sign_test_p(0, 5) == 1.0


def test_e_value_is_one_without_evidence_and_grows_with_wins() -> None:
    assert e_value(0, 0) == 1.0
    values = [e_value(wins, 0) for wins in range(0, 8)]
    assert values == sorted(values)
    assert e_value(3, 1) == pytest.approx((1.3**3) * (0.7**1))


def test_bootstrap_ci_brackets_constant_and_positive_effects() -> None:
    assert bootstrap_ci([0.1] * 5, alpha=0.05, resamples=100, seed=0) == pytest.approx((0.1, 0.1))
    low, high = bootstrap_ci([0.1, 0.2, 0.3, 0.4], alpha=0.05, resamples=500, seed=0)
    assert low > 0.0
    assert low <= high


def test_alpha_spending_sums_to_the_budget_and_decays() -> None:
    spending = AlphaSpending(horizon=4, alpha_total=0.05, decay=0.5)
    alphas = tuple(spending.alpha(generation) for generation in range(1, 5))
    assert sum(alphas) == pytest.approx(0.05)
    assert alphas == tuple(sorted(alphas, reverse=True))
    assert sum(spending.weights()) == pytest.approx(1.0)
    with pytest.raises(StatsError, match="generation"):
        spending.alpha(0)


def test_alpha_spending_rejects_bad_parameters() -> None:
    with pytest.raises(StatsError, match="horizon"):
        AlphaSpending(horizon=0)
    with pytest.raises(StatsError, match="alpha_total"):
        AlphaSpending(alpha_total=1.0)
    with pytest.raises(StatsError, match="decay"):
        AlphaSpending(decay=0.0)


def test_null_candidate_is_rejected() -> None:
    test = paired_test([1.0, 0.5, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], alpha=0.05, resamples=100)
    assert (test.wins, test.losses, test.ties) == (0, 0, 4)
    assert test.p_value == 1.0
    assert test.e_value == 1.0
    assert test.decision == "reject"
    assert test.mean_delta == 0.0


def test_strong_paired_improvement_is_accepted() -> None:
    test = paired_test([0.0] * 5, [1.0] * 5, alpha=0.05, resamples=200, seed=1)
    assert test.wins == 5
    assert test.p_value == pytest.approx(1 / 32)
    assert test.ci_low > 0.0
    assert test.decision == "accept"
    assert test.accepted


def test_one_win_is_not_enough_evidence() -> None:
    test = paired_test([0.0], [1.0], alpha=0.05, resamples=10)
    assert test.p_value == 0.5
    assert test.decision == "reject"


def test_paired_inputs_are_validated() -> None:
    with pytest.raises(StatsError, match="length"):
        paired_test([1.0], [1.0, 2.0], alpha=0.05)
    with pytest.raises(StatsError, match="empty"):
        paired_test([], [], alpha=0.05)
    with pytest.raises(StatsError, match="alpha"):
        paired_test([1.0], [1.0], alpha=1.0)


def test_certificate_is_bound_to_target_and_verifier() -> None:
    test = paired_test([0.0] * 5, [1.0] * 5, alpha=0.05, resamples=50)
    certificate = certify(test, target="dev", verifier_version="suite-v1", generation=3)
    assert certificate.valid_for("dev", "suite-v1")
    assert not certificate.valid_for("dev", "suite-v2")
    assert not certificate.valid_for("fresh", "suite-v1")
    payload = certificate.to_dict()
    assert payload["test"]["decision"] == "accept"  # type: ignore[index]
    assert payload["target"] == "dev"
