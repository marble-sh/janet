from __future__ import annotations

from janet.eval.mutants import generate_mutants

_SOURCE = """
def classify(n):
    if n < 0:
        return -1
    if n == 0:
        return 0
    total = 0
    for i in range(n):
        total += i
    return total + 1
"""


def test_mutants_are_distinct_and_changed() -> None:
    mutants = generate_mutants(_SOURCE, limit=100, seed="classify")
    assert mutants
    assert len({mutant.source for mutant in mutants}) == len(mutants)
    assert all(mutant.source != _SOURCE for mutant in mutants)
    assert all(mutant.description for mutant in mutants)
    assert [mutant.index for mutant in mutants] == list(range(len(mutants)))


def test_mutants_are_deterministic_and_sample_is_stable() -> None:
    first = generate_mutants(_SOURCE, limit=4, seed="fixed")
    second = generate_mutants(_SOURCE, limit=4, seed="fixed")
    assert first == second
    assert len(first) == 4


def test_mutants_cover_operators() -> None:
    descriptions = " ".join(
        mutant.description for mutant in generate_mutants(_SOURCE, limit=100, seed="x")
    )
    assert "compare" in descriptions
    assert "constant" in descriptions
    assert "augassign" in descriptions


def test_source_without_sites_has_no_mutants() -> None:
    assert generate_mutants("def f():\n    return 1.5\n", limit=10) == ()


def test_boolean_and_unary_sites() -> None:
    source = "def f(a, b):\n    return not a and b\n"
    mutants = generate_mutants(source, limit=10)
    descriptions = {mutant.description for mutant in mutants}
    assert any(description.startswith("boolop") for description in descriptions)
    assert any(description.startswith("drop not") for description in descriptions)
