"""W1 shape-hard tasks: stateful semantics, not just edge cases.

These are dev-eligible difficulty-5 tasks whose failure modes are conceptual
(coalescing, dependency graphs, jumps, pattern classes, scheduling) rather than
missing validation of an otherwise simple function.
"""

from __future__ import annotations

from janet.bench.seeds import CaseSeed, TaskSeed


def _stub(name: str, params: str) -> str:
    return f"def {name}({params}):\n    raise NotImplementedError\n"


def _c(*args: object, err: str | None = None) -> CaseSeed:
    return CaseSeed(tuple(args), error=err)


def _task(
    id: str,
    title: str,
    prompt: str,
    entry: str,
    buggy: str,
    reference: str,
    visible: tuple[CaseSeed, ...],
    anchor: tuple[CaseSeed, ...],
    saturated: bool = False,
) -> TaskSeed:
    return TaskSeed(
        id=id,
        family="shape",
        title=title,
        prompt=prompt,
        entrypoint=entry,
        buggy=buggy,
        reference=reference,
        visible=visible,
        anchor=anchor,
        difficulty=5,
        saturated=saturated,
        timeout_s=20.0,
    )


_ALL: tuple[TaskSeed, ...] = (
    _task(
        "allocate_blocks",
        "First-fit allocator with coalescing",
        "Implement allocate_blocks(capacity, requests). Memory starts as one free "
        "block [0, capacity). Each request is ['alloc', size] or ['free', handle]. "
        "An alloc uses the lowest-addressed free block that fits, splits it, and "
        "returns a new handle (1, 2, ...); if nothing fits it returns None. A free "
        "releases a live handle, coalesces with adjacent free blocks, and returns "
        "True; freeing an unknown or already-freed handle returns False. capacity "
        "and sizes must be positive integers; malformed requests, sizes, handles, "
        "or unknown operations raise ValueError. Return the list of results.",
        "allocate_blocks",
        _stub("allocate_blocks", "capacity, requests"),
        r"""def allocate_blocks(capacity, requests):
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    blocks = [[0, capacity, None]]
    next_handle = 1
    results = []
    for request in requests:
        if not isinstance(request, (list, tuple)) or not request:
            raise ValueError("invalid request")
        op = request[0]
        if op == "alloc":
            if (
                len(request) != 2
                or not isinstance(request[1], int)
                or isinstance(request[1], bool)
                or request[1] <= 0
            ):
                raise ValueError("invalid allocation size")
            size = request[1]
            chosen = None
            for index, block in enumerate(blocks):
                if block[2] is None and block[1] >= size:
                    chosen = index
                    break
            if chosen is None:
                results.append(None)
                continue
            start, length, _ = blocks[chosen]
            handle = next_handle
            next_handle += 1
            blocks[chosen] = [start, size, handle]
            if length > size:
                blocks.insert(chosen + 1, [start + size, length - size, None])
            results.append(handle)
        elif op == "free":
            if (
                len(request) != 2
                or not isinstance(request[1], int)
                or isinstance(request[1], bool)
            ):
                raise ValueError("invalid handle")
            handle = request[1]
            index = next(
                (i for i, block in enumerate(blocks) if block[2] == handle), None
            )
            if index is None:
                results.append(False)
                continue
            blocks[index][2] = None
            if index + 1 < len(blocks) and blocks[index + 1][2] is None:
                blocks[index][1] += blocks[index + 1][1]
                del blocks[index + 1]
            if index > 0 and blocks[index - 1][2] is None:
                blocks[index - 1][1] += blocks[index][1]
                del blocks[index]
            results.append(True)
        else:
            raise ValueError("unknown operation")
    return results
""",
        (
            _c(10, [["alloc", 3], ["alloc", 4], ["free", 1], ["alloc", 5]]),
            _c(5, [["alloc", 6]]),
            _c(4, [["alloc", 2], ["alloc", 2]]),
        ),
        (
            _c(10, [["alloc", 3], ["alloc", 3], ["free", 1], ["free", 2], ["alloc", 6]]),
            _c(6, [["alloc", 2], ["free", 9], ["free", 1], ["free", 1]]),
            _c(5, [["alloc", 5], ["free", 1], ["alloc", 5]]),
            _c(10, [["alloc", 3], ["alloc", 3], ["alloc", 4], ["free", 2], ["alloc", 4]]),
            _c(5, [["alloc", 0]], err="ValueError"),
            _c(5, [["alloc", True]], err="ValueError"),
            _c(5, [["free", "1"]], err="ValueError"),
            _c(5, [["bogus", 1]], err="ValueError"),
            _c(0, [], err="ValueError"),
        ),
    ),
    _task(
        "schedule_tasks",
        "Minimum-late single-machine schedule",
        "Implement schedule_tasks(tasks). Each task is [name, duration, deadline] "
        "with a unique name, a positive integer duration, and a non-negative "
        "integer deadline. Schedule every task on one machine starting at time 0, "
        "in the order you return. A task is late when its completion time exceeds "
        "its deadline. Return [order, late_count] where order is the list of names "
        "that minimizes late_count; among minimizers choose the lexicographically "
        "smallest name sequence. An empty list returns [[], 0]. Malformed tasks or "
        "duplicate names raise ValueError.",
        "schedule_tasks",
        _stub("schedule_tasks", "tasks"),
        r"""from itertools import permutations


def schedule_tasks(tasks):
    parsed = []
    for task in tasks:
        if not isinstance(task, (list, tuple)) or len(task) != 3:
            raise ValueError("each task must be [name, duration, deadline]")
        name, duration, deadline = task
        if not isinstance(name, str) or not name:
            raise ValueError("invalid name")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
        ):
            raise ValueError("invalid duration")
        if (
            not isinstance(deadline, int)
            or isinstance(deadline, bool)
            or deadline < 0
        ):
            raise ValueError("invalid deadline")
        parsed.append((name, duration, deadline))
    if len({name for name, _, _ in parsed}) != len(parsed):
        raise ValueError("duplicate task name")
    if not parsed:
        return [[], 0]
    best = None
    for order in permutations(range(len(parsed))):
        clock = 0
        late = 0
        names = []
        for index in order:
            name, duration, deadline = parsed[index]
            clock += duration
            if clock > deadline:
                late += 1
            names.append(name)
        key = (late, names)
        if best is None or key < best[0]:
            best = (key, names, late)
    return [best[1], best[2]]
""",
        (
            _c([["a", 2, 2], ["b", 1, 3]]),
            _c([["a", 3, 2], ["b", 1, 1]]),
            _c([]),
        ),
        (
            _c([["a", 2, 5], ["b", 2, 4], ["c", 1, 3]]),
            _c([["a", 3, 3], ["b", 3, 6], ["c", 1, 7], ["d", 2, 9]]),
            _c([["a", 1, 0], ["b", 1, 0], ["c", 1, 0]]),
            _c([["a", 2, 10], ["b", 1, 10], ["c", 1, 10], ["d", 1, 10]]),
            _c([["a", 4, 4], ["b", 2, 2], ["c", 2, 8], ["d", 1, 9], ["e", 1, 10]]),
            _c([["a", 0, 1]], err="ValueError"),
            _c([["a", 1, 1], ["a", 1, 1]], err="ValueError"),
        ),
        saturated=True,
    ),
    _task(
        "evaluate_cells",
        "Spreadsheet formulas with cycle detection",
        "Implement evaluate_cells(cells). cells maps uppercase cell names to a "
        "number or a formula string beginning with '='. Formulas use numbers, cell "
        "references (uppercase letters then digits), + - * /, parentheses, and "
        "unary minus. Return {name: value} for every cell, sorted by name, with "
        "float values. Raise ValueError for an unknown reference, a dependency "
        "cycle, division by zero, a malformed formula, or a cell value that is "
        "neither a number nor a formula string.",
        "evaluate_cells",
        _stub("evaluate_cells", "cells"),
        r"""def evaluate_cells(cells):
    if not isinstance(cells, dict):
        raise ValueError("cells must be an object")
    cache = {}
    visiting = set()

    def cell_value(name):
        if name in cache:
            return cache[name]
        if name in visiting:
            raise ValueError("cycle detected")
        if name not in cells:
            raise ValueError("unknown cell")
        visiting.add(name)
        result = eval_value(cells[name])
        visiting.remove(name)
        cache[name] = result
        return result

    def eval_value(raw):
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        if not isinstance(raw, str) or not raw.startswith("="):
            raise ValueError("invalid cell")
        tokens = tokenize(raw[1:])
        result, position = parse_expr(tokens, 0)
        if position != len(tokens):
            raise ValueError("malformed formula")
        return result

    def tokenize(text):
        tokens = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue
            if char.isdigit() or (
                char == "." and index + 1 < len(text) and text[index + 1].isdigit()
            ):
                end = index
                while end < len(text) and (text[end].isdigit() or text[end] == "."):
                    end += 1
                literal = text[index:end]
                if literal.count(".") > 1:
                    raise ValueError("malformed number")
                tokens.append(("num", float(literal)))
                index = end
                continue
            if char.isalpha():
                end = index
                while end < len(text) and text[end].isalpha():
                    end += 1
                digits = end
                while digits < len(text) and text[digits].isdigit():
                    digits += 1
                if digits == end:
                    raise ValueError("malformed reference")
                tokens.append(("ref", text[index:digits]))
                index = digits
                continue
            if char in "+-*/()":
                tokens.append((char, char))
                index += 1
                continue
            raise ValueError("invalid character")
        return tokens

    def parse_expr(tokens, position):
        value, position = parse_term(tokens, position)
        while position < len(tokens) and tokens[position][0] in ("+", "-"):
            operator = tokens[position][0]
            right, position = parse_term(tokens, position + 1)
            value = value + right if operator == "+" else value - right
        return value, position

    def parse_term(tokens, position):
        value, position = parse_factor(tokens, position)
        while position < len(tokens) and tokens[position][0] in ("*", "/"):
            operator = tokens[position][0]
            right, position = parse_factor(tokens, position + 1)
            if operator == "*":
                value *= right
            else:
                if right == 0:
                    raise ValueError("division by zero")
                value /= right
        return value, position

    def parse_factor(tokens, position):
        if position >= len(tokens):
            raise ValueError("malformed formula")
        kind, token = tokens[position]
        if kind == "num":
            return float(token), position + 1
        if kind == "ref":
            if not isinstance(token, str):
                raise ValueError("malformed reference")
            return cell_value(token), position + 1
        if kind == "-":
            value, position = parse_factor(tokens, position + 1)
            return -value, position
        if kind == "(":
            value, position = parse_expr(tokens, position + 1)
            if position >= len(tokens) or tokens[position][0] != ")":
                raise ValueError("unbalanced parentheses")
            return value, position + 1
        raise ValueError("malformed formula")

    return {name: cell_value(name) for name in sorted(cells)}
""",
        (
            _c({"A1": 2, "A2": "=A1+3"}),
            _c({"A1": "=2*3"}),
            _c({"A1": 1, "A2": "=A1*2", "A3": "=A2+A1"}),
        ),
        (
            _c({"A1": "=(1+2)*3"}),
            _c({"A1": "=A2", "A2": "=A1"}, err="ValueError"),
            _c({"A1": "=B9"}, err="ValueError"),
            _c({"A1": "=1/0"}, err="ValueError"),
            _c({"A1": "=A1+"}, err="ValueError"),
            _c({"A1": True}, err="ValueError"),
            _c(
                {"A1": 1, "B1": "=A1+1", "C1": "=B1*2"},
            ),
            _c({"A1": "=1/2"}),
            _c({"A1": "=-2+3"}),
        ),
    ),
    _task(
        "run_vm",
        "Stack machine with jumps and variables",
        "Implement run_vm(program). Instructions are lists: ['push', n], ['pop'], "
        "['add'], ['sub'], ['mul'], ['dup'], ['store', name], ['load', name], "
        "['emit'], ['jmp', target], ['jz', target], ['halt']. pc starts at 0 and "
        "advances by one unless a jump is taken; jz pops a value and jumps when it "
        "is zero. emit pops and appends to the output. Return the output list. "
        "Raise ValueError for malformed instructions, stack underflow, an unknown "
        "variable, an out-of-range jump, or more than 1000 executed instructions. "
        "Running past the end stops normally.",
        "run_vm",
        _stub("run_vm", "program"),
        r"""def run_vm(program):
    stack = []
    variables = {}
    output = []
    pc = 0
    steps = 0

    def pop():
        if not stack:
            raise ValueError("stack underflow")
        return stack.pop()

    while 0 <= pc < len(program):
        steps += 1
        if steps > 1000:
            raise ValueError("step limit exceeded")
        instruction = program[pc]
        if not isinstance(instruction, (list, tuple)) or not instruction:
            raise ValueError("invalid instruction")
        op = instruction[0]
        if op == "push":
            if (
                len(instruction) != 2
                or not isinstance(instruction[1], int)
                or isinstance(instruction[1], bool)
            ):
                raise ValueError("invalid push")
            stack.append(instruction[1])
        elif op == "pop":
            if len(instruction) != 1:
                raise ValueError("invalid pop")
            pop()
        elif op in ("add", "sub", "mul"):
            if len(instruction) != 1:
                raise ValueError("invalid arithmetic")
            right = pop()
            left = pop()
            if op == "add":
                stack.append(left + right)
            elif op == "sub":
                stack.append(left - right)
            else:
                stack.append(left * right)
        elif op == "dup":
            if len(instruction) != 1:
                raise ValueError("invalid dup")
            if not stack:
                raise ValueError("stack underflow")
            stack.append(stack[-1])
        elif op == "store":
            if len(instruction) != 2 or not isinstance(instruction[1], str):
                raise ValueError("invalid store")
            variables[instruction[1]] = pop()
        elif op == "load":
            if len(instruction) != 2 or not isinstance(instruction[1], str):
                raise ValueError("invalid load")
            if instruction[1] not in variables:
                raise ValueError("unknown variable")
            stack.append(variables[instruction[1]])
        elif op == "emit":
            if len(instruction) != 1:
                raise ValueError("invalid emit")
            output.append(pop())
        elif op in ("jmp", "jz"):
            if (
                len(instruction) != 2
                or not isinstance(instruction[1], int)
                or isinstance(instruction[1], bool)
            ):
                raise ValueError("invalid jump")
            if not 0 <= instruction[1] < len(program):
                raise ValueError("invalid jump target")
            if op == "jmp":
                pc = instruction[1]
                continue
            if pop() == 0:
                pc = instruction[1]
                continue
        elif op == "halt":
            break
        else:
            raise ValueError("unknown op")
        pc += 1
    return output
""",
        (
            _c([["push", 2], ["push", 3], ["add"], ["emit"], ["halt"]]),
            _c([["push", 1], ["store", "x"], ["load", "x"], ["emit"], ["halt"]]),
            _c([]),
        ),
        (
            _c([["push", 0], ["jz", 4], ["push", 9], ["emit"], ["halt"]]),
            _c([["push", 1], ["jz", 4], ["push", 9], ["emit"], ["halt"]]),
            _c([["push", 7], ["dup"], ["add"], ["emit"], ["halt"]]),
            _c([["push", 5], ["push", 3], ["sub"], ["emit"], ["halt"]]),
            _c([["pop"]], err="ValueError"),
            _c([["push", 1], ["bogus"]], err="ValueError"),
            _c([["load", "missing"], ["emit"]], err="ValueError"),
            _c([["jmp", 9]], err="ValueError"),
            _c([["jmp", 0]], err="ValueError"),
            _c([["push"]], err="ValueError"),
        ),
        saturated=True,
    ),
    _task(
        "glob_match",
        "Glob pattern matching with classes",
        "Implement glob_match(pattern, text). Support '*' for any sequence, '?' "
        "for one character, and character classes: [abc], [a-z], and a leading "
        "! or ^ for negation. Everything else matches literally. An unterminated "
        "class, an empty class, or a reversed range raises ValueError. The whole "
        "text must match. Return True or False.",
        "glob_match",
        _stub("glob_match", "pattern, text"),
        r"""def glob_match(pattern, text):
    def parse_class(index):
        position = index + 1
        negate = False
        if position < len(pattern) and pattern[position] in "!^":
            negate = True
            position += 1
        chars = set()
        ranges = []
        if position < len(pattern) and pattern[position] == "]":
            raise ValueError("empty character class")
        while position < len(pattern) and pattern[position] != "]":
            if (
                position + 2 < len(pattern)
                and pattern[position + 1] == "-"
                and pattern[position + 2] != "]"
            ):
                if pattern[position] > pattern[position + 2]:
                    raise ValueError("reversed range")
                ranges.append((pattern[position], pattern[position + 2]))
                position += 3
            else:
                chars.add(pattern[position])
                position += 1
        if position >= len(pattern):
            raise ValueError("unterminated character class")

        def member(char):
            inside = char in chars or any(low <= char <= high for low, high in ranges)
            return not inside if negate else inside

        return member, position + 1

    memo = {}

    def match(pattern_index, text_index):
        key = (pattern_index, text_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern):
            result = text_index == len(text)
        else:
            char = pattern[pattern_index]
            if char == "*":
                result = match(pattern_index + 1, text_index) or (
                    text_index < len(text) and match(pattern_index, text_index + 1)
                )
            elif char == "?":
                result = text_index < len(text) and match(
                    pattern_index + 1, text_index + 1
                )
            elif char == "[":
                member, next_index = parse_class(pattern_index)
                result = (
                    text_index < len(text)
                    and member(text[text_index])
                    and match(next_index, text_index + 1)
                )
            else:
                result = (
                    text_index < len(text)
                    and text[text_index] == char
                    and match(pattern_index + 1, text_index + 1)
                )
        memo[key] = result
        return result

    return match(0, 0)
""",
        (
            _c("a*b", "acccb"),
            _c("a?c", "abc"),
            _c("abc", "abc"),
        ),
        (
            _c("[a-c]x", "bx"),
            _c("[!a-c]x", "dx"),
            _c("[a-c]x", "dx"),
            _c("*", ""),
            _c("a*b*c", "aXbYc"),
            _c("a[", "a[", err="ValueError"),
            _c("[]", "x", err="ValueError"),
            _c("[z-a]", "x", err="ValueError"),
            _c("a?c", "ac"),
            _c("[^ab]", "c"),
        ),
    ),
)


def shape_seeds() -> tuple[TaskSeed, ...]:
    return _ALL


__all__ = ["shape_seeds"]
