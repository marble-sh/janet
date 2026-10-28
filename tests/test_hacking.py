from __future__ import annotations

from janet.eval.hacking import (
    audit_trajectories,
    duplicate_solutions,
    hardcoding_flags,
    solution_fingerprint,
)


def test_fingerprint_ignores_whitespace_and_comments() -> None:
    left = "def f(x):\n    # comment\n    return x + 1\n"
    right = "def f( x ):\n    return x+1\n"
    assert solution_fingerprint(left) == solution_fingerprint(right)
    assert solution_fingerprint("def f():\n    return 1\n") != solution_fingerprint(
        "def f():\n    return 2\n"
    )


def test_fingerprint_ignores_a_renamed_entrypoint() -> None:
    left = "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n"
    right = "def merge_intervals(x, lo, hi):\n    return min(max(x, lo), hi)\n"
    assert solution_fingerprint(left) == solution_fingerprint(right)


def test_duplicate_solutions_across_tasks() -> None:
    same = "def solve(x):\n    return x\n"
    samples = (("a", same), ("b", same), ("c", "def solve(x):\n    return x + 1\n"))
    groups = duplicate_solutions(samples)
    assert groups == (("a", ("a", "b")),)
    assert duplicate_solutions((("a", same),)) == ()
    assert duplicate_solutions((("a", same), ("b", same), ("c", same))) == (("a", ("a", "b", "c")),)


def test_hardcoding_flags_use_whole_tokens() -> None:
    assert hardcoding_flags("def f():\n    return flatten_nested\n", ("flatten_nested",)) == (
        "flatten_nested",
    )
    assert hardcoding_flags("def flatten_nested():\n    return 1\n", ("flatten_nested",)) == (
        "flatten_nested",
    )
    assert hardcoding_flags("flatten_nested_extra = 1\n", ("flatten_nested",)) == ()
    assert hardcoding_flags("x = 1\n", ("flatten_nested",)) == ()


def test_hardcoding_flags_ignore_comments_and_docstrings() -> None:
    source = (
        '"""flatten the list and merge_intervals."""\n'
        "def solve(value):\n"
        "    # flatten_nested is another task\n"
        '    """docstring mentioning clamp_value"""\n'
        "    return value\n"
    )
    assert hardcoding_flags(source, ("flatten", "flatten_nested", "clamp_value")) == ()


def test_hardcoding_flags_see_string_literals() -> None:
    assert hardcoding_flags('x = "clamp_value"\n', ("clamp_value",)) == ("clamp_value",)


def test_audit_trajectories_reports_both_modes() -> None:
    same = "def solve(x):\n    return 1\n"
    report = audit_trajectories(
        (("a", same), ("b", same), ("c", "clamp_value(3)")),
        forbidden=("clamp_value",),
    )
    payload = report.to_dict()
    assert report.flagged == 2
    assert payload["duplicate_groups"] == [{"first_task": "a", "tasks": ["a", "b"]}]
    assert payload["hardcoded"] == [{"task": "c", "terms": ["clamp_value"]}]


def test_audit_trajectories_can_scope_forbidden_terms_per_task() -> None:
    report = audit_trajectories(
        (("a", "def a():\n    return 1\n"), ("b", "a = 1\n")),
        forbidden_by_task={"a": (), "b": ("a",)},
    )
    assert report.hardcoded == (("b", ("a",)),)


def test_audit_trajectories_skips_empty_sources() -> None:
    report = audit_trajectories((("a", ""), ("b", "  ")))
    assert report.samples == 0
    assert report.duplicate_groups == ()
    assert report.hardcoded == ()
