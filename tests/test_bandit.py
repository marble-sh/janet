from __future__ import annotations

import json
from pathlib import Path

import pytest

from janet.harness.bandit import (
    OPERATOR_INSTRUCTIONS,
    OPERATORS,
    REWARD_VERSION,
    ArmStats,
    BanditError,
    BanditState,
    format_bandit,
    reward_for,
)


def test_fresh_state_has_every_operator() -> None:
    state = BanditState.fresh()
    assert tuple(state.arms) == OPERATORS
    assert all(stats.pulls == 0 and stats.mean == 0.0 for stats in state.arms.values())
    assert set(OPERATOR_INSTRUCTIONS) == set(OPERATORS)


def test_choose_explores_untried_arms_in_order() -> None:
    state = BanditState.fresh()
    for expected in OPERATORS:
        assert state.choose() == expected
        state.record(expected, 0.0)
    assert state.choose() in OPERATORS


def test_choose_exploits_the_best_mean_without_exploration() -> None:
    state = BanditState.fresh()
    for name in OPERATORS:
        state.record(name, 0.0)
    state.record("skill", 1.0)
    assert state.choose(0.0) == "skill"


def test_exploration_bonus_favours_a_less_pulled_arm() -> None:
    state = BanditState.fresh()
    state.arms["prompt"] = ArmStats(pulls=10, total_reward=5.0)
    state.arms["skill"] = ArmStats(pulls=10, total_reward=5.0)
    state.arms["budget"] = ArmStats(pulls=10, total_reward=5.0)
    state.arms["prompt+skill"] = ArmStats(pulls=1, total_reward=0.0)
    assert state.choose(2.0) == "prompt+skill"


def test_equal_scores_tie_break_in_operator_order() -> None:
    state = BanditState.fresh()
    for name in OPERATORS:
        state.arms[name] = ArmStats(pulls=1, total_reward=0.5)
    assert state.choose(0.0) == "prompt"


def test_negative_exploration_and_unknown_arm_are_rejected() -> None:
    state = BanditState.fresh()
    with pytest.raises(BanditError, match="exploration"):
        state.choose(-1.0)
    with pytest.raises(BanditError, match="unknown"):
        state.record("bogus", 1.0)


def test_record_updates_the_policy() -> None:
    state = BanditState.fresh()
    state.record("prompt", 1.0)
    state.record("prompt", -0.5)
    stats = state.arms["prompt"]
    assert stats.pulls == 2
    assert stats.mean == pytest.approx(0.25)
    assert state.updated_at


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "bandit.json"
    state = BanditState.fresh()
    state.record("skill", 0.5)
    state.save(path)
    loaded = BanditState.load(path)
    assert loaded.arms["skill"].pulls == 1
    assert loaded.arms["skill"].mean == pytest.approx(0.5)
    assert loaded.updated_at == state.updated_at
    assert loaded.to_dict()["total_pulls"] == 1
    assert loaded.to_dict()["reward_version"] == REWARD_VERSION


def test_load_accepts_a_list_of_arms(tmp_path: Path) -> None:
    path = tmp_path / "bandit.json"
    path.write_text(
        json.dumps(
            {
                "reward_version": REWARD_VERSION,
                "arms": [
                    {"name": "prompt", "pulls": 2, "total_reward": 1.5},
                    {"name": "bogus", "pulls": 9, "total_reward": 9.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    state = BanditState.load(path)
    assert state.arms["prompt"].pulls == 2
    assert state.arms["prompt"].total_reward == pytest.approx(1.5)
    assert "bogus" not in state.arms


def test_load_accepts_a_mapping_of_arms(tmp_path: Path) -> None:
    path = tmp_path / "bandit.json"
    path.write_text(
        json.dumps(
            {
                "reward_version": REWARD_VERSION,
                "arms": {"budget": {"pulls": 3, "total_reward": -0.5}},
            }
        ),
        encoding="utf-8",
    )
    state = BanditState.load(path)
    assert state.arms["budget"].pulls == 3
    assert state.arms["budget"].total_reward == pytest.approx(-0.5)


def test_load_is_robust_to_bad_input(tmp_path: Path) -> None:
    missing = BanditState.load(tmp_path / "missing.json")
    assert missing.to_dict()["total_pulls"] == 0
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert BanditState.load(path).to_dict()["total_pulls"] == 0
    path.write_text("[]", encoding="utf-8")
    assert BanditState.load(path).to_dict()["total_pulls"] == 0
    path.write_text(
        json.dumps(
            {
                "reward_version": REWARD_VERSION,
                "arms": [{"name": "prompt", "pulls": "x", "total_reward": True}],
            }
        ),
        encoding="utf-8",
    )
    state = BanditState.load(path)
    assert state.arms["prompt"].pulls == 0
    assert state.arms["prompt"].total_reward == 0.0


def test_load_discards_a_state_from_another_reward_version(tmp_path: Path) -> None:
    path = tmp_path / "bandit.json"
    path.write_text(
        json.dumps(
            {
                "reward_version": REWARD_VERSION - 1,
                "arms": [{"name": "prompt", "pulls": 5, "total_reward": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    assert BanditState.load(path).to_dict()["total_pulls"] == 0
    path.write_text(
        json.dumps(
            {
                "reward_version": REWARD_VERSION,
                "arms": [{"name": "prompt", "pulls": 5, "total_reward": 5.0}],
            }
        ),
        encoding="utf-8",
    )
    assert BanditState.load(path).arms["prompt"].pulls == 5


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "improved": True,
                "error": None,
                "anchor_score": 0.8,
                "parent_anchor": 0.7,
                "heldout_score": 0.6,
                "parent_heldout": 0.5,
            },
            1.0,
        ),
        (
            {
                "improved": True,
                "error": None,
                "anchor_score": 0.795,
                "parent_anchor": 0.785,
                "heldout_score": 0.7444,
                "parent_heldout": 0.7444,
            },
            0.25,
        ),
        (
            {
                "improved": True,
                "error": None,
                "anchor_score": 0.8,
                "parent_anchor": 0.7,
                "heldout_score": 0.4,
                "parent_heldout": 0.5,
            },
            -0.25,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.8,
                "parent_anchor": 0.7,
                "heldout_score": 0.6,
                "parent_heldout": 0.5,
            },
            0.5,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.8,
                "parent_anchor": 0.7,
                "heldout_score": 0.5,
                "parent_heldout": 0.5,
            },
            0.0,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.8,
                "parent_anchor": 0.7,
                "heldout_score": 0.4,
                "parent_heldout": 0.5,
            },
            -0.5,
        ),
        (
            {
                "improved": False,
                "error": "bad",
                "anchor_score": None,
                "parent_anchor": None,
                "heldout_score": None,
                "parent_heldout": None,
            },
            -0.25,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": None,
                "parent_anchor": None,
                "heldout_score": None,
                "parent_heldout": None,
            },
            -0.05,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.6,
                "parent_anchor": 0.5,
                "heldout_score": None,
                "parent_heldout": None,
            },
            0.5,
        ),
        (
            {
                "improved": True,
                "error": None,
                "anchor_score": 0.6,
                "parent_anchor": 0.5,
                "heldout_score": None,
                "parent_heldout": None,
            },
            1.0,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.5,
                "parent_anchor": 0.5,
                "heldout_score": None,
                "parent_heldout": None,
            },
            0.0,
        ),
        (
            {
                "improved": False,
                "error": None,
                "anchor_score": 0.4,
                "parent_anchor": 0.5,
                "heldout_score": None,
                "parent_heldout": None,
            },
            -0.5,
        ),
    ],
)
def test_reward_shaping(kwargs: dict[str, object], expected: float) -> None:
    assert reward_for(**kwargs) == expected  # type: ignore[arg-type]


def test_to_dict_and_format_are_inspectable() -> None:
    state = BanditState.fresh()
    state.record("budget", -0.5)
    payload = state.to_dict()
    assert payload["reward_version"] == REWARD_VERSION
    assert payload["total_pulls"] == 1
    arms = payload["arms"]
    assert isinstance(arms, list)
    assert arms[0]["name"] == "prompt"
    assert "budget" in format_bandit(state)
