"""Extract a Python solution from a model response."""

from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    """Return the last fenced Python block, or the stripped text if unfenced.

    Last-block wins because models often show an intermediate version first and
    the corrected version last. An unfenced response is returned as-is: the
    verifier will decide whether it is valid Python.
    """

    matches = _FENCE.findall(text)
    if matches:
        return str(matches[-1]).strip()
    return text.strip()


def looks_like_python(text: str) -> bool:
    """A weak, cheap check used to reject prose-only responses early."""

    stripped = text.strip()
    if not stripped:
        return False
    if "def " in stripped or "class " in stripped or "import " in stripped:
        return True
    try:
        compile(stripped, "<candidate>", "exec")
    except SyntaxError:
        return False
    return True
