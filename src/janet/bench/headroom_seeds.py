"""W1 headroom family: harder verified tasks derived from trusted references.

These are solution-centric in shape: each task starts from a reference
implementation and a set of executable cases, with a buggy stub the policy must
repair. They are split across dev/anchor/fresh so the loop has room above the
saturated easy suite.
"""

from __future__ import annotations

from janet.bench.seeds import CaseSeed, TaskSeed


def _stub(name: str, params: str) -> str:
    return f"def {name}({params}):\n    raise NotImplementedError\n"


def _c(*args: object, err: str | None = None) -> CaseSeed:
    return CaseSeed(tuple(args), error=err)


_ALL: tuple[TaskSeed, ...] = (
    TaskSeed(
        id="weighted_interval_schedule",
        family="headroom",
        title="Weighted interval scheduling with a deterministic tie-break",
        prompt=(
            "Implement weighted_interval_schedule(intervals). Each interval is "
            "[start, end, weight]. Return [best_weight, chosen] where chosen is the "
            "chronologically ordered list of selected [start, end, weight] entries. "
            "Intervals with end <= start or negative weight are invalid and must "
            "raise ValueError. Touching intervals (end == next start) are compatible. "
            "On a weight tie choose the lexicographically smaller chosen list. An "
            "empty input returns [0, []]."
        ),
        entrypoint="weighted_interval_schedule",
        buggy=_stub("weighted_interval_schedule", "intervals"),
        reference=r"""def weighted_interval_schedule(intervals):
    items = []
    for item in intervals:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("each interval must have three values")
        start, end, weight = item
        if end <= start or weight < 0:
            raise ValueError("invalid interval")
        items.append((start, end, weight))
    items.sort(key=lambda value: (value[1], value[0]))
    n = len(items)
    best = [(0, ())] * (n + 1)
    for i in range(1, n + 1):
        start, end, weight = items[i - 1]
        j = i - 1
        while j > 0 and items[j - 1][1] > start:
            j -= 1
        included = (best[j][0] + weight, best[j][1] + ((start, end, weight),))
        excluded = best[i - 1]
        if included[0] > excluded[0] or (
            included[0] == excluded[0] and included[1] < excluded[1]
        ):
            best[i] = included
        else:
            best[i] = excluded
    total, chosen = best[n]
    return [total, [list(item) for item in chosen]]
""",
        visible=(
            _c([[1, 3, 5], [2, 4, 6], [3, 5, 4]]),
            _c([]),
        ),
        anchor=(
            _c([[1, 4, 5], [2, 3, 10], [3, 5, 4], [5, 6, 2]]),
            _c([[1, 1, 5]], err="ValueError"),
            _c([[1, 2, -1]], err="ValueError"),
            _c([[1, 2, 1], [2, 3, 1], [1, 3, 2]]),
            _c([[1, 2, 0]]),
            _c([[1, 2, 0], [2, 3, 1]]),
            _c([[1, 2, 1], [1, 2, 1]]),
            _c([[1, 2, 1], [2, 3, 1], [3, 4, 1]]),
            _c([[1, 2, 2], [2, 3, 1], [1, 3, 3]]),
        ),
        difficulty=4,
        saturated=True,
        timeout_s=20.0,
    ),
    TaskSeed(
        id="token_bucket_allow",
        family="headroom",
        title="Token bucket with continuous refill",
        prompt=(
            "Implement token_bucket_allow(capacity, refill_per_second, requests). "
            "The bucket starts full at t=0 and refills continuously up to capacity. "
            "Each request is [timestamp_seconds, tokens]. Process requests in order "
            "and return a list of booleans: True when the request is allowed and the "
            "tokens are consumed, False otherwise. Timestamps must be non-decreasing; "
            "negative capacity, refill rate, or requested tokens are invalid and must "
            "raise ValueError. Use a small tolerance for floating-point refill."
        ),
        entrypoint="token_bucket_allow",
        buggy=_stub("token_bucket_allow", "capacity, refill_per_second, requests"),
        reference=r"""def token_bucket_allow(capacity, refill_per_second, requests):
    if capacity < 0 or refill_per_second < 0:
        raise ValueError("capacity and refill rate must be non-negative")
    tokens = float(capacity)
    last = 0.0
    allowed = []
    for timestamp, wanted in requests:
        if timestamp < last or wanted < 0:
            raise ValueError("invalid request stream")
        tokens = min(float(capacity), tokens + (timestamp - last) * refill_per_second)
        last = float(timestamp)
        if wanted <= tokens + 1e-9:
            tokens -= wanted
            allowed.append(True)
        else:
            allowed.append(False)
    return allowed
""",
        visible=(
            _c(2, 1.0, [[0, 1], [0, 1], [0, 1], [1, 1]]),
            _c(1, 0.0, [[0, 1], [0, 1]]),
        ),
        anchor=(
            _c(3, 1.0, [[0, 5]]),
            _c(2, 0.5, [[0, 2], [2, 1], [2, 1]]),
            _c(0, 10.0, [[0, 0], [1, 1]]),
            _c(1, 1.0, [[2, 1], [1, 1]], err="ValueError"),
            _c(1, 1.0, [[0, -1]], err="ValueError"),
            _c(2, 1.0, [[0, 0], [0, 0], [0, 2]]),
            _c(2, 1.0, [[0, 2], [0.5, 1], [1.0, 1]]),
            _c(1, 1.0, [[0, 1], [1, 1]]),
            _c(0, 0.0, [[0, 0], [0, 1]]),
            _c(1, 1.0, [[-1, 0]], err="ValueError"),
        ),
        difficulty=4,
        saturated=True,
        timeout_s=20.0,
    ),
    TaskSeed(
        id="topological_order",
        family="headroom",
        title="Deterministic topological order with cycle detection",
        prompt=(
            "Implement topological_order(n, edges). Nodes are 0..n-1 and each edge "
            "[before, after] means before must appear earlier. Return a topological "
            "order that always picks the smallest available node. If a cycle exists, "
            "or an edge is out of range, self-referential, or duplicated, raise "
            "ValueError. n must be non-negative."
        ),
        entrypoint="topological_order",
        buggy=_stub("topological_order", "n, edges"),
        reference=r"""import heapq


def topological_order(n, edges):
    if n < 0:
        raise ValueError("n must be non-negative")
    adjacency = [set() for _ in range(n)]
    indegree = [0] * n
    for before, after in edges:
        if not (0 <= before < n and 0 <= after < n) or before == after:
            raise ValueError("invalid edge")
        if after in adjacency[before]:
            raise ValueError("duplicate edge")
        adjacency[before].add(after)
        indegree[after] += 1
    ready = [node for node in range(n) if indegree[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for successor in sorted(adjacency[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(order) != n:
        raise ValueError("cycle detected")
    return order
""",
        visible=(),
        anchor=(
            _c(4, [[0, 1], [0, 2], [1, 3], [2, 3]]),
            _c(3, []),
            _c(0, []),
            _c(2, [[0, 1], [1, 0]], err="ValueError"),
            _c(2, [[0, 1], [0, 1]], err="ValueError"),
            _c(3, [[0, 3]], err="ValueError"),
            _c(-1, [], err="ValueError"),
        ),
        difficulty=5,
        timeout_s=20.0,
        anchor_only=True,
    ),
    TaskSeed(
        id="min_cost_grid",
        family="headroom",
        title="Grid shortest path with obstacles",
        prompt=(
            "Implement min_cost_grid(grid). grid is a rectangular list of rows of "
            "non-negative integers; 0 is an obstacle. Start at the top-left and move "
            "right or down to the bottom-right. Return the minimum sum including the "
            "start and end cells, or -1 if unreachable. An empty or ragged grid must "
            "raise ValueError. The start or end may itself be an obstacle."
        ),
        entrypoint="min_cost_grid",
        buggy=_stub("min_cost_grid", "grid"),
        reference=r"""def min_cost_grid(grid):
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    rows = len(grid)
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("grid must be rectangular")
    infinity = float("inf")
    cost = [[infinity] * cols for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 0:
                continue
            if row == 0 and col == 0:
                cost[row][col] = grid[row][col]
                continue
            best = infinity
            if row > 0:
                best = min(best, cost[row - 1][col])
            if col > 0:
                best = min(best, cost[row][col - 1])
            if best < infinity:
                cost[row][col] = best + grid[row][col]
    return -1 if cost[-1][-1] == infinity else cost[-1][-1]
""",
        visible=(),
        anchor=(
            _c([[1, 3, 1], [1, 5, 1], [4, 2, 1]]),
            _c([[1, 0], [0, 1]]),
            _c([[5]]),
            _c([[1, 2, 3], [4, 5, 6]]),
            _c([[1, 1], [1]], err="ValueError"),
            _c([], err="ValueError"),
            _c([[0]]),
        ),
        difficulty=5,
        timeout_s=20.0,
        anchor_only=True,
    ),
    TaskSeed(
        id="word_ladder_length",
        family="headroom",
        title="Shortest word-ladder transformation",
        prompt=(
            "Implement word_ladder_length(begin, end, words). Return the number of "
            "words in the shortest transformation sequence from begin to end, changing "
            "one letter at a time, where every intermediate word is in words. Return "
            "0 when no sequence exists. begin and end need not be in words. Words must "
            "be non-empty, lowercase, and the same length as begin; otherwise raise "
            "ValueError. If begin == end, return 1."
        ),
        entrypoint="word_ladder_length",
        buggy=_stub("word_ladder_length", "begin, end, words"),
        reference=r"""from collections import deque


def word_ladder_length(begin, end, words):
    if not begin or not end or len(begin) != len(end):
        raise ValueError("begin and end must be non-empty and the same length")
    bank = set(words)
    if any(not word or len(word) != len(begin) for word in bank):
        raise ValueError("all words must match the begin length")
    if begin == end:
        return 1
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    queue = deque([(begin, 1)])
    seen = {begin}
    while queue:
        word, distance = queue.popleft()
        for index in range(len(word)):
            for letter in alphabet:
                candidate = word[:index] + letter + word[index + 1 :]
                if candidate == end:
                    return distance + 1
                if candidate in bank and candidate not in seen:
                    seen.add(candidate)
                    queue.append((candidate, distance + 1))
    return 0
""",
        visible=(),
        anchor=(
            _c("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]),
            _c("hit", "cog", ["hot", "dot", "dog", "lot", "log"]),
            _c("a", "c", ["a", "b", "c"]),
            _c("a", "a", []),
            _c("a", "bb", ["a"], err="ValueError"),
            _c("", "a", ["a"], err="ValueError"),
        ),
        difficulty=5,
        timeout_s=20.0,
        fresh_only=True,
    ),
    TaskSeed(
        id="json_pointer_get",
        family="headroom",
        title="RFC 6901 JSON Pointer lookup",
        prompt=(
            "Implement json_pointer_get(document, pointer) for RFC 6901 JSON Pointer. "
            "The empty pointer returns the document. Split tokens on '/', decode ~1 as "
            "'/' and ~0 as '~', traverse dict keys or list indices, and return the "
            "value. Raise KeyError for a missing key, IndexError for an out-of-range "
            "list index, ValueError for a malformed pointer or non-integer list token, "
            "and TypeError when traversing a scalar."
        ),
        entrypoint="json_pointer_get",
        buggy=_stub("json_pointer_get", "document, pointer"),
        reference=r"""def json_pointer_get(document, pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("pointer must be empty or start with /")
    if pointer == "":
        return document
    current = document
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if "~" in token:
            raise ValueError("invalid pointer escape")
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise ValueError("list tokens must be non-negative integers")
            index = int(token)
            if index >= len(current):
                raise IndexError(index)
            current = current[index]
        else:
            raise TypeError("cannot traverse a scalar")
    return current
""",
        visible=(),
        anchor=(
            _c({"a": {"b": [10, 20]}}, "/a/b/1"),
            _c({"a/b": 1}, "/a~1b"),
            _c({"": 2}, "/"),
            _c({"a": 1}, ""),
            _c([], "/0", err="IndexError"),
            _c({"a": 1}, "a", err="ValueError"),
            _c({"a": 1}, "/b", err="KeyError"),
            _c({"a": 1}, "/a~2", err="ValueError"),
        ),
        difficulty=4,
        timeout_s=20.0,
        fresh_only=True,
    ),
)


def headroom_seeds() -> tuple[TaskSeed, ...]:
    return _ALL


def all_headroom_seeds() -> tuple[TaskSeed, ...]:
    return _ALL
