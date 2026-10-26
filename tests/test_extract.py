from __future__ import annotations

from janet.eval.extract import extract_python, looks_like_python


def test_extract_fenced_block() -> None:
    text = "Here is the fix:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_python(text) == "def f():\n    return 1"


def test_extract_takes_the_last_block() -> None:
    text = "```python\nOLD = 1\n```\n```python\nNEW = 2\n```"
    assert extract_python(text) == "NEW = 2"


def test_extract_accepts_py_and_case_insensitive() -> None:
    assert extract_python("```PY\nx = 1\n```") == "x = 1"
    assert extract_python("```Python\nx = 1\n```") == "x = 1"


def test_extract_unfenced_text_is_returned_stripped() -> None:
    assert extract_python("  x = 1  ") == "x = 1"


def test_looks_like_python() -> None:
    assert looks_like_python("def f():\n    return 1")
    assert looks_like_python("class C:\n    pass")
    assert looks_like_python("import math")
    assert looks_like_python("x = 1")
    assert not looks_like_python("")
    assert not looks_like_python("This is prose, not code.")
