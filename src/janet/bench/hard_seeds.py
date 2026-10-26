"""P3.3 headroom family: tasks a 7B model fails often.

Three shapes: spec-to-code from a stub, multi-rule logic, and a bug (or several)
inside a larger file. Splits are assigned by *stratified difficulty* from a
measured one-shot baseline (see ``_SPLIT``), never by whether a harness helps.
"""

from __future__ import annotations

from dataclasses import replace

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
) -> TaskSeed:
    return TaskSeed(
        id=id,
        family="hard_repair",
        title=title,
        prompt=prompt,
        entrypoint=entry,
        buggy=buggy,
        reference=reference,
        visible=visible,
        anchor=anchor,
        difficulty=3,
        timeout_s=20.0,
    )


_ALL: tuple[TaskSeed, ...] = (
    _task(
        "lru_hits",
        "LRU cache hit counter",
        "Implement lru_hits(capacity, accesses): simulate an LRU cache of the given "
        "capacity over the list of keys and return how many accesses were hits. A hit "
        "refreshes the key's recency; a miss inserts it, evicting the least recently "
        "used key when full. A capacity of 0 or less never hits.",
        "lru_hits",
        _stub("lru_hits", "capacity, accesses"),
        r"""def lru_hits(capacity, accesses):
    if capacity <= 0:
        return 0
    cache = []
    hits = 0
    for key in accesses:
        if key in cache:
            hits += 1
            cache.remove(key)
        elif len(cache) == capacity:
            cache.pop(0)
        cache.append(key)
    return hits
""",
        (_c(2, [1, 2, 1, 3, 2]), _c(1, [1, 1, 1]), _c(3, [])),
        (
            _c(2, [1, 2, 3, 1, 2, 3]),
            _c(0, [1, 1]),
            _c(3, [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]),
            _c(2, [1, 2, 1, 3, 1, 2]),
            _c(1, ["a", "b", "a", "a"]),
        ),
    ),
    _task(
        "calc_expr",
        "Integer calculator with precedence and unary minus",
        "Implement calc(expr): evaluate an integer arithmetic expression containing "
        "non-negative integer literals, + - * /, parentheses, unary minus, and spaces. "
        "* and / bind tighter than + and -, and operators of equal precedence associate "
        "left. Division truncates toward zero (7/-2 is -3). Raise ValueError for any "
        "invalid character, malformed expression, empty expression, unary plus, or "
        "division by zero.",
        "calc",
        _stub("calc", "expr"),
        r"""def calc(expr):
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(int(expr[i:j]))
            i = j
        elif ch in "+-*/()":
            tokens.append(ch)
            i += 1
        else:
            raise ValueError("bad character")
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        tok = peek()
        pos += 1
        return tok

    def parse_expr():
        value = parse_term()
        while peek() in ("+", "-"):
            op = take()
            rhs = parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term():
        value = parse_factor()
        while peek() in ("*", "/"):
            op = take()
            rhs = parse_factor()
            if op == "*":
                value *= rhs
            else:
                if rhs == 0:
                    raise ValueError("division by zero")
                q = abs(value) // abs(rhs)
                value = q if (value < 0) == (rhs < 0) else -q
        return value

    def parse_factor():
        tok = take()
        if tok == "-":
            return -parse_factor()
        if tok == "(":
            value = parse_expr()
            if take() != ")":
                raise ValueError("missing )")
            return value
        if isinstance(tok, int):
            return tok
        raise ValueError("unexpected token")

    result = parse_expr()
    if pos != len(tokens):
        raise ValueError("trailing tokens")
    return result
""",
        (_c("1 + 2 * 3"), _c("(1 + 2) * 3"), _c("-4 / 3")),
        (
            _c("7 / -2"),
            _c("2 * (3 + 4) - 5"),
            _c("--3"),
            _c("10 - 4 - 3"),
            _c("100 / 7 / 2"),
            _c("2 +", err="ValueError"),
            _c("1 / (2 - 2)", err="ValueError"),
            _c("+3", err="ValueError"),
            _c("(1 + 2", err="ValueError"),
        ),
    ),
    _task(
        "topo_sort_min",
        "Lexicographically smallest topological order",
        "Implement topo_sort(n, edges): nodes are 0..n-1 and each edge (a, b) means a "
        "must come before b. Return the lexicographically smallest topological order "
        "as a list (always pick the smallest available node). Duplicate edges are "
        "allowed. Raise ValueError if the graph has a cycle.",
        "topo_sort",
        _stub("topo_sort", "n, edges"),
        r"""import heapq


def topo_sort(n, edges):
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, nxt)
    if len(order) != n:
        raise ValueError("cycle")
    return order
""",
        (_c(3, [[0, 1], [1, 2]]), _c(3, []), _c(4, [[3, 0], [2, 1]])),
        (
            _c(5, [[4, 0], [4, 1], [3, 1], [0, 2]]),
            _c(4, [[0, 1], [0, 1], [1, 2]]),
            _c(1, []),
            _c(3, [[0, 1], [1, 2], [2, 0]], err="ValueError"),
            _c(6, [[5, 2], [5, 0], [4, 0], [4, 1], [2, 3], [3, 1]]),
        ),
    ),
    _task(
        "wrap_text",
        "Greedy word wrap",
        "Implement wrap(text, width): split text on whitespace into words and greedily "
        "pack them into lines of at most width characters, joining words with a single "
        "space. A word longer than width goes on its own line unbroken. Return the list "
        "of lines; text with no words returns []. width is at least 1.",
        "wrap",
        _stub("wrap", "text, width"),
        r"""def wrap(text, width):
    lines = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
""",
        (_c("the quick brown fox", 10), _c("", 5), _c("a b c", 3)),
        (
            _c("supercalifragilistic is long", 8),
            _c("  spaced   out   words  ", 7),
            _c("one two three four five six", 9),
            _c("aaa bbb ccc", 7),
            _c("x", 1),
        ),
    ),
    _task(
        "compare_versions",
        "Compare dotted versions with pre-release suffixes",
        "Implement compare_versions(a, b) returning -1, 0 or 1. A version is dotted "
        "non-negative integers optionally followed by '-' and a suffix (e.g. '1.2.0-rc'). "
        "Compare the numeric parts numerically, treating missing parts as 0 ('1.0' equals "
        "'1.0.0'). If the numeric parts are equal, a version with a suffix is lower than "
        "one without; two different suffixes compare as strings.",
        "compare_versions",
        _stub("compare_versions", "a, b"),
        r"""def compare_versions(a, b):
    def parse(version):
        main, _, suffix = version.partition("-")
        return [int(part) for part in main.split(".")], suffix

    na, sa = parse(a)
    nb, sb = parse(b)
    size = max(len(na), len(nb))
    na += [0] * (size - len(na))
    nb += [0] * (size - len(nb))
    if na != nb:
        return -1 if na < nb else 1
    if sa == sb:
        return 0
    if not sa:
        return 1
    if not sb:
        return -1
    return -1 if sa < sb else 1
""",
        (_c("1.2", "1.10"), _c("1.0", "1.0.0"), _c("2.0.0-rc", "2.0.0")),
        (
            _c("1.0.0-alpha", "1.0.0-beta"),
            _c("1.0.1", "1.0.0-rc"),
            _c("10.0", "9.99.99"),
            _c("1.0.0-b", "1.0.0-a"),
            _c("3", "3.0.0.0"),
            _c("0.9", "0.10"),
        ),
    ),
    _task(
        "decode_nested",
        "Decode nested repeat strings",
        "Implement decode_string(s) for the encoding k[text], meaning text repeated k "
        "times, where k is one or more digits and encodings can nest ('3[a2[c]]' is "
        "'accaccacc'). Characters outside brackets are copied as is. The input is always "
        "well formed.",
        "decode_string",
        _stub("decode_string", "s"),
        r"""def decode_string(s):
    stack = []
    cur = ""
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append((cur, num))
            cur = ""
            num = 0
        elif ch == "]":
            prev, k = stack.pop()
            cur = prev + cur * k
        else:
            cur += ch
    return cur
""",
        (_c("3[a]"), _c("2[ab]c"), _c("3[a2[c]]")),
        (
            _c("10[x]"),
            _c("2[a2[b3[c]]]"),
            _c("ab2[c]d"),
            _c(""),
            _c("1[2[3[z]]]"),
        ),
    ),
    _task(
        "parse_csv_row",
        "Parse one CSV line with quoted fields",
        "Implement parse_csv_line(line) returning the list of field strings. Fields are "
        "separated by commas. A field may be wrapped in double quotes, in which case "
        "commas inside are literal and two consecutive double quotes inside stand for one "
        "double quote. An empty line is one empty field. Quotes are never part of the "
        "returned value unless doubled.",
        "parse_csv_line",
        _stub("parse_csv_line", "line"),
        r"""def parse_csv_line(line):
    fields = []
    cur = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_quotes = False
            else:
                cur.append(ch)
        elif ch == '"':
            in_quotes = True
        elif ch == ",":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    fields.append("".join(cur))
    return fields
""",
        (_c("a,b,c"), _c('a,"b,c",d'), _c("")),
        (
            _c('"say ""hi""",x'),
            _c(",,"),
            _c('"a,b","c,d"'),
            _c('plain,"",end'),
            _c('"""",1'),
        ),
    ),
    _task(
        "min_meeting_rooms",
        "Minimum rooms for half-open meetings",
        "Implement min_rooms(meetings): meetings is a list of (start, end) pairs, each "
        "occupying the half-open interval [start, end). Return the minimum number of "
        "rooms so no two overlapping meetings share a room; a meeting may start exactly "
        "when another ends in the same room. Meetings with start >= end are ignored.",
        "min_rooms",
        _stub("min_rooms", "meetings"),
        r"""def min_rooms(meetings):
    events = []
    for start, end in meetings:
        if start < end:
            events.append((start, 1))
            events.append((end, -1))
    events.sort()
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best
""",
        (_c([[0, 30], [5, 10], [15, 20]]), _c([[1, 2], [2, 3]]), _c([])),
        (
            _c([[1, 5], [2, 6], [3, 7], [8, 9]]),
            _c([[4, 4], [1, 2]]),
            _c([[0, 10], [10, 20], [5, 15]]),
            _c([[1, 10], [2, 3], [3, 4], [4, 5]]),
            _c([[9, 3], [1, 2]]),
        ),
    ),
    _task(
        "next_permutation",
        "Next lexicographic permutation",
        "Implement next_permutation(items): return a new list holding the next "
        "lexicographically greater permutation of items. If items is already the greatest "
        "permutation, return the items sorted ascending. Items may repeat; do not modify "
        "the input.",
        "next_permutation",
        _stub("next_permutation", "items"),
        r"""def next_permutation(items):
    a = list(items)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1 :] = reversed(a[i + 1 :])
    return a
""",
        (_c([1, 2, 3]), _c([3, 2, 1]), _c([1, 1, 5])),
        (
            _c([1, 3, 2]),
            _c([2, 3, 1]),
            _c([1, 5, 1]),
            _c([]),
            _c([4]),
            _c([1, 2, 3, 6, 5, 4]),
        ),
    ),
    _task(
        "date_add_days",
        "Add days to a Gregorian date",
        "Implement date_add_days(y, m, d, n): return the tuple (year, month, day) that is "
        "n days after the valid Gregorian date y-m-d; n may be negative or large. Do not "
        "use the datetime or calendar modules. Leap years: divisible by 4, except "
        "centuries unless divisible by 400.",
        "date_add_days",
        _stub("date_add_days", "y, m, d, n"),
        r"""def date_add_days(y, m, d, n):
    def leap(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def dim(year, month):
        return [31, 29 if leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]

    while n > 0:
        left = dim(y, m) - d
        if n <= left:
            d += n
            n = 0
        else:
            n -= left + 1
            d = 1
            m += 1
            if m > 12:
                m = 1
                y += 1
    while n < 0:
        if -n < d:
            d += n
            n = 0
        else:
            n += d
            m -= 1
            if m < 1:
                m = 12
                y -= 1
            d = dim(y, m)
    return (y, m, d)
""",
        (_c(2024, 2, 28, 1), _c(2023, 12, 31, 1), _c(2024, 3, 1, -1)),
        (
            _c(2000, 1, 1, 366),
            _c(1900, 2, 28, 1),
            _c(2024, 1, 31, 30),
            _c(2023, 3, 1, -365),
            _c(2100, 2, 28, 1),
            _c(2024, 5, 15, 0),
        ),
    ),
    _task(
        "subtract_intervals",
        "Subtract interval lists",
        "Implement subtract_intervals(a, b): a and b are lists of [start, end) intervals, "
        "each list sorted and non-overlapping on its own. Return the sorted list of "
        "[start, end] intervals covering the points in a that are not in b. Empty "
        "intervals are not returned.",
        "subtract_intervals",
        _stub("subtract_intervals", "a, b"),
        r"""def subtract_intervals(a, b):
    result = []
    for start, end in a:
        cur = start
        for bs, be in b:
            if be <= cur or bs >= end:
                continue
            if bs > cur:
                result.append([cur, bs])
            cur = max(cur, be)
            if cur >= end:
                break
        if cur < end:
            result.append([cur, end])
    return result
""",
        (_c([[0, 10]], [[2, 4]]), _c([[0, 5]], []), _c([], [[1, 2]])),
        (
            _c([[0, 10], [12, 20]], [[3, 5], [8, 14]]),
            _c([[0, 10]], [[0, 10]]),
            _c([[1, 3], [5, 9]], [[0, 2], [6, 7]]),
            _c([[0, 100]], [[10, 20], [30, 40], [50, 60]]),
            _c([[5, 8]], [[0, 5], [8, 9]]),
        ),
    ),
    _task(
        "json_path_get",
        "Safe nested lookup by path string",
        "Implement json_path_get(obj, path): look up a value inside nested dicts and "
        "lists by a path such as 'a.b[2].c' or '[0].name'. Dot-separated names index "
        "dicts; [n] indexes lists. If any step is missing, out of range, or applied to the "
        "wrong type, return None. An empty path returns obj itself.",
        "json_path_get",
        _stub("json_path_get", "obj, path"),
        r"""import re


def json_path_get(obj, path):
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = obj
    for tok in tokens:
        if tok.startswith("["):
            idx = int(tok[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return None
            cur = cur[tok]
    return cur
""",
        (
            _c({"a": {"b": [1, 2, 3]}}, "a.b[1]"),
            _c([{"n": 5}], "[0].n"),
            _c({"a": 1}, ""),
        ),
        (
            _c({"a": {"b": [{"c": 7}]}}, "a.b[0].c"),
            _c({"a": [1]}, "a[3]"),
            _c({"a": 1}, "a.b"),
            _c({"x": [[1, 2], [3, 4]]}, "x[1][0]"),
            _c({"0": "zero"}, "0"),
            _c([1, 2], "x"),
        ),
    ),
    _task(
        "knapsack_pick",
        "Knapsack with lexicographically smallest optimal items",
        "Implement knapsack_pick(weights, values, capacity): weights and values are "
        "positive integers for items 0..n-1 and each item may be used at most once. Among "
        "all item sets with the maximum total value that fit within capacity, return the "
        "one whose sorted index list is lexicographically smallest, as a list of indices. "
        "Return [] if nothing fits.",
        "knapsack_pick",
        _stub("knapsack_pick", "weights, values, capacity"),
        r"""def knapsack_pick(weights, values, capacity):
    n = len(weights)
    best = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for w in range(capacity + 1):
            best[i][w] = best[i + 1][w]
            if weights[i] <= w:
                best[i][w] = max(best[i][w], values[i] + best[i + 1][w - weights[i]])
    picked = []
    w = capacity
    for i in range(n):
        if weights[i] <= w and values[i] + best[i + 1][w - weights[i]] == best[i][w]:
            picked.append(i)
            w -= weights[i]
    return picked
""",
        (_c([1, 2, 3], [6, 10, 12], 5), _c([5], [1], 4), _c([2, 2], [3, 3], 2)),
        (
            _c([2, 3, 4, 5], [3, 4, 5, 6], 5),
            _c([1, 1, 1], [1, 1, 1], 2),
            _c([4, 3, 2], [5, 4, 3], 5),
            _c([3, 4, 5], [4, 5, 6], 0),
            _c([1, 3, 4, 5], [1, 4, 5, 7], 7),
        ),
    ),
    _task(
        "order_totals",
        "Fix the order pricing module",
        "totals_by_customer(orders) returns {customer: total_cents}. Each order has "
        "'customer', 'items' (dicts with integer 'price' in cents and 'qty') and an "
        "optional 'coupon'. Rules: an order with no item cost has total 0. Coupon "
        "'SAVE10' takes sub//10 off when the subtotal is at least 10000. Shipping is 500 "
        "unless the discounted subtotal is at least 5000, then 0. Tax is 8% of the "
        "discounted subtotal only (not shipping), rounded half up as (x*8+50)//100. The "
        "order total is discounted + shipping + tax. A customer's value is the sum over "
        "all of their orders. The module below breaks several of these rules. Fix it.",
        "totals_by_customer",
        r"""def line_total(item):
    return item["price"] * item["qty"]


def subtotal(order):
    return sum(line_total(i) for i in order["items"])


def discount(order, sub):
    if order.get("coupon") == "SAVE10" and sub > 10000:
        return sub // 10
    return 0


def shipping(order, discounted):
    if discounted < 5000:
        return 500
    return 0


def tax(amount):
    return amount * 8 // 100


def order_total(order):
    sub = subtotal(order)
    disc = discount(order, sub)
    discounted = sub - disc
    ship = shipping(order, discounted)
    return discounted + ship + tax(discounted + ship)


def totals_by_customer(orders):
    totals = {}
    for order in orders:
        totals[order["customer"]] = order_total(order)
    return totals
""",
        r"""def line_total(item):
    return item["price"] * item["qty"]


def subtotal(order):
    return sum(line_total(i) for i in order["items"])


def discount(order, sub):
    if order.get("coupon") == "SAVE10" and sub >= 10000:
        return sub // 10
    return 0


def shipping(discounted):
    return 0 if discounted >= 5000 else 500


def tax(amount):
    return (amount * 8 + 50) // 100


def order_total(order):
    sub = subtotal(order)
    if sub == 0:
        return 0
    discounted = sub - discount(order, sub)
    return discounted + shipping(discounted) + tax(discounted)


def totals_by_customer(orders):
    totals = {}
    for order in orders:
        name = order["customer"]
        totals[name] = totals.get(name, 0) + order_total(order)
    return totals
""",
        (
            _c([{"customer": "a", "items": [{"price": 1000, "qty": 2}]}]),
            _c([{"customer": "a", "items": []}]),
            _c(
                [
                    {"customer": "a", "items": [{"price": 6000, "qty": 1}]},
                    {"customer": "a", "items": [{"price": 6000, "qty": 1}]},
                ]
            ),
        ),
        (
            _c([{"customer": "b", "items": [{"price": 5000, "qty": 2}], "coupon": "SAVE10"}]),
            _c([{"customer": "b", "items": [{"price": 9999, "qty": 1}], "coupon": "SAVE10"}]),
            _c([{"customer": "c", "items": [{"price": 4999, "qty": 1}]}]),
            _c([{"customer": "c", "items": [{"price": 5000, "qty": 1}]}]),
            _c(
                [
                    {"customer": "d", "items": [{"price": 333, "qty": 3}]},
                    {"customer": "e", "items": []},
                    {"customer": "d", "items": [{"price": 1, "qty": 1}], "coupon": "SAVE10"},
                ]
            ),
        ),
    ),
    _task(
        "stock_report",
        "Fix the stock ledger",
        "stock_report(events) processes events (op, sku, qty) with non-negative integer "
        "qty and returns {'stock': {...}, 'shortfall': {...}}. 'in' adds qty to the sku. "
        "'out' removes up to the available stock; anything it could not remove is added to "
        "that sku's shortfall, and stock never goes negative. 'adjust' sets the on-hand "
        "count to exactly qty. Any other op raises ValueError. 'stock' omits skus whose "
        "count is 0; 'shortfall' omits skus with none. The code below breaks several of "
        "these rules. Fix it.",
        "stock_report",
        r"""def apply_event(stock, event):
    op, sku, qty = event
    if op == "in":
        stock[sku] = stock.get(sku, 0) + qty
    elif op == "out":
        stock[sku] = stock.get(sku, 0) - qty
    elif op == "adjust":
        stock[sku] = stock.get(sku, 0) + qty
    return stock


def stock_report(events):
    stock = {}
    for event in events:
        apply_event(stock, event)
    return {"stock": stock, "shortfall": {}}
""",
        r"""def apply_event(stock, shortfall, event):
    op, sku, qty = event
    if op == "in":
        stock[sku] = stock.get(sku, 0) + qty
    elif op == "out":
        have = stock.get(sku, 0)
        taken = min(have, qty)
        stock[sku] = have - taken
        if qty > taken:
            shortfall[sku] = shortfall.get(sku, 0) + qty - taken
    elif op == "adjust":
        stock[sku] = qty
    else:
        raise ValueError("unknown op")


def stock_report(events):
    stock = {}
    shortfall = {}
    for event in events:
        apply_event(stock, shortfall, event)
    return {
        "stock": {k: v for k, v in stock.items() if v > 0},
        "shortfall": shortfall,
    }
""",
        (
            _c([["in", "a", 5], ["out", "a", 2]]),
            _c([["in", "a", 1], ["out", "a", 3]]),
            _c([["adjust", "z", 4]]),
        ),
        (
            _c([["in", "a", 5], ["adjust", "a", 2], ["out", "a", 2]]),
            _c([["out", "q", 4], ["out", "q", 1]]),
            _c([["in", "a", 3], ["in", "b", 2], ["out", "a", 3], ["out", "b", 5]]),
            _c([["in", "a", 3], ["bogus", "a", 1]], err="ValueError"),
            _c([]),
        ),
    ),
    _task(
        "max_circular_sum",
        "Maximum circular subarray sum",
        "Implement max_circular_sum(nums): nums is a non-empty list of integers treated "
        "as circular (the end wraps to the start). Return the largest sum of a non-empty "
        "contiguous (possibly wrapping) subarray that uses each element at most once.",
        "max_circular_sum",
        _stub("max_circular_sum", "nums"),
        r"""def max_circular_sum(nums):
    total = sum(nums)
    best_max = cur_max = nums[0]
    best_min = cur_min = nums[0]
    for x in nums[1:]:
        cur_max = max(x, cur_max + x)
        best_max = max(best_max, cur_max)
        cur_min = min(x, cur_min + x)
        best_min = min(best_min, cur_min)
    if best_max < 0:
        return best_max
    return max(best_max, total - best_min)
""",
        (_c([1, -2, 3, -2]), _c([5, -3, 5]), _c([-3, -2, -3])),
        (
            _c([3, -1, 2, -1]),
            _c([3, -2, 2, -3]),
            _c([-2, 4, -5, 4, -5, 9, 4]),
            _c([7]),
            _c([0, 0]),
            _c([8, -8, 9, -9, 10, -11, 12]),
        ),
    ),
    _task(
        "cron_field_match",
        "Match a cron field expression",
        "Implement cron_matches(field, value) returning True if the integer value "
        "matches the cron field. A field is comma-separated parts; the value matches if "
        "it matches any part. Parts: '*' (any value), '*/n' (value divisible by n), 'a' "
        "(exactly a), 'a-b' (a <= value <= b inclusive), 'a-b/n' (in the range and "
        "(value - a) divisible by n).",
        "cron_matches",
        _stub("cron_matches", "field, value"),
        r"""def cron_matches(field, value):
    for part in field.split(","):
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if base == "*":
            if value % step == 0:
                return True
        elif "-" in base:
            lo, hi = (int(x) for x in base.split("-"))
            if lo <= value <= hi and (value - lo) % step == 0:
                return True
        elif int(base) == value:
            return True
    return False
""",
        (_c("*", 7), _c("*/15", 30), _c("1-5", 6)),
        (
            _c("1-10/3", 7),
            _c("1-10/3", 8),
            _c("0,15,30,45", 45),
            _c("5,10-12", 11),
            _c("*/7", 0),
            _c("2-4/2", 4),
            _c("2-4/2", 3),
        ),
    ),
    _task(
        "min_window_substr",
        "Smallest window containing all characters",
        "Implement min_window(s, t): return the shortest substring of s that contains "
        "every character of t, counting multiplicity (if t has two 'a', the window needs "
        "at least two 'a'). If several windows tie, return the leftmost. Return '' if no "
        "window exists or t is empty.",
        "min_window",
        _stub("min_window", "s, t"),
        r"""from collections import Counter


def min_window(s, t):
    if not t:
        return ""
    need = Counter(t)
    missing = len(t)
    best = (None, None)
    left = 0
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if best[0] is None or right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            missing += 1
            left += 1
    if best[0] is None:
        return ""
    return s[best[0] : best[1] + 1]
""",
        (_c("ADOBECODEBANC", "ABC"), _c("a", "aa"), _c("ab", "")),
        (
            _c("aa", "aa"),
            _c("bbaa", "aba"),
            _c("xyzzy", "zy"),
            _c("abcabc", "cba"),
            _c("hello world", "lo w"),
        ),
    ),
)

# Stratified assignment from the measured one-shot baseline in
# docs/state/p33-headroom-2026-10-24.md: tasks are ranked by baseline score and
# dealt round-robin so every split spans the same difficulty range.
_SPLIT: dict[str, str] = {
    # snake-dealt by ascending baseline score: dev, anchor, fresh, fresh, anchor, dev, ...
    "date_add_days": "dev",
    "min_meeting_rooms": "anchor",
    "calc_expr": "fresh",
    "parse_csv_row": "fresh",
    "topo_sort_min": "anchor",
    "order_totals": "dev",
    "knapsack_pick": "dev",
    "json_path_get": "anchor",
    "subtract_intervals": "fresh",
    "wrap_text": "fresh",
    "compare_versions": "anchor",
    "next_permutation": "dev",
    "cron_field_match": "dev",
    "decode_nested": "anchor",
    "lru_hits": "fresh",
    "max_circular_sum": "fresh",
    "min_window_substr": "anchor",
    "stock_report": "dev",
}


def hard_seeds() -> tuple[TaskSeed, ...]:
    """The headroom tasks with their split flags applied."""

    out: list[TaskSeed] = []
    for seed in _ALL:
        split = _SPLIT.get(seed.id, "dev")
        if split == "anchor":
            out.append(replace(seed, anchor_only=True))
        elif split == "fresh":
            out.append(replace(seed, fresh_only=True))
        else:
            out.append(seed)
    return tuple(out)


def all_hard_seeds() -> tuple[TaskSeed, ...]:
    """Unassigned tasks (used for difficulty probing)."""

    return _ALL
