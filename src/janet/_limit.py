"""Tiny exec wrapper that applies rlimits and CPU affinity, then execs.

Run as ``python -m janet._limit ... -- command``. This exists so the governor
can enforce limits without ``preexec_fn`` (which is unsafe in a threaded
process).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import resource
import sys
from collections.abc import Sequence

_DEFAULT_FSIZE = 64 * 1024 * 1024
_DEFAULT_NOFILE = 256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="janet._limit", add_help=False)
    parser.add_argument("--mem-bytes", type=int, default=0)
    parser.add_argument("--cpu-seconds", type=int, default=0)
    parser.add_argument("--cpus", type=str, default="")
    parser.add_argument("--fsize-bytes", type=int, default=_DEFAULT_FSIZE)
    parser.add_argument("--nofile", type=int, default=_DEFAULT_NOFILE)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _apply_limits(args: argparse.Namespace) -> None:
    if args.mem_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (args.mem_bytes, args.mem_bytes))
    if args.cpu_seconds > 0:
        soft = args.cpu_seconds
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))
    if args.fsize_bytes > 0:
        resource.setrlimit(resource.RLIMIT_FSIZE, (args.fsize_bytes, args.fsize_bytes))
    if args.nofile > 0:
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.nofile, args.nofile))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.cpus:
        cpus = {int(part) for part in args.cpus.split(",") if part.strip()}
        if cpus:
            os.sched_setaffinity(0, cpus)
    if args.nice:
        with contextlib.suppress(OSError):
            os.nice(args.nice)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, remaining = parser.parse_known_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    command = command + remaining
    if not command:
        parser.print_usage(sys.stderr)
        return 2
    try:
        _apply_limits(args)
    except (OSError, ValueError) as exc:
        print(f"janet._limit: cannot apply limits: {exc}", file=sys.stderr)
        return 125
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        print(f"janet._limit: cannot exec {command[0]!r}: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
