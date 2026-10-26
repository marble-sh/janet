"""Black-box worker executed inside the sandbox.

Usage: ``python worker.py SOLUTION INPUTS ENTRYPOINT``. ``INPUTS`` is a JSON
list of argument lists. The worker prints a JSON object; it never sees expected
outputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_janet_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("usage: worker.py SOLUTION INPUTS ENTRYPOINT", file=sys.stderr)
        return 2
    solution_path = Path(args[0])
    inputs_path = Path(args[1])
    entrypoint = args[2]
    try:
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": f"inputs: {exc}"}))
        return 0
    if not isinstance(inputs, list):
        print(json.dumps({"error": "inputs must be a list"}))
        return 0
    try:
        module = _load(solution_path)
        function = getattr(module, entrypoint)
    except Exception as exc:
        print(json.dumps({"error": f"import: {type(exc).__name__}: {exc}"}))
        return 0
    if not callable(function):
        print(json.dumps({"error": f"entrypoint {entrypoint!r} is not callable"}))
        return 0

    results: list[dict[str, Any]] = []
    for case_args in inputs:
        if not isinstance(case_args, list):
            results.append({"ok": False, "error": "case args must be a list"})
            continue
        try:
            value = function(*case_args)
        except Exception as exc:
            results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"ok": True, "value": value})
    print(json.dumps({"results": results}, default=repr))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
