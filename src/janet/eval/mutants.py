"""AST mutants of a reference implementation.

A test input is only worth adding to the evaluator if it separates the reference
from a plausible wrong program. These mutants are the plausible wrong programs:
single-site edits (flipped comparison, off-by-one constant, swapped operator).
"""

from __future__ import annotations

import ast
import copy
import random
from dataclasses import dataclass

_COMPARE_SWAP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
_BINOP_SWAP: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Add,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
}


@dataclass(frozen=True, slots=True)
class Mutant:
    index: int
    description: str
    source: str


class _Collector(ast.NodeVisitor):
    """Enumerates mutation sites in a stable order."""

    def __init__(self) -> None:
        self.sites: list[ast.AST] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(type(op) in _COMPARE_SWAP for op in node.ops):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) in _BINOP_SWAP:
            self.sites.append(node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.sites.append(node)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.sites.append(node)


class _Mutator(ast.NodeTransformer):
    def __init__(self, target: int) -> None:
        self.target = target
        self.count = -1
        self.description = ""

    def _hit(self) -> bool:
        self.count += 1
        return self.count == self.target

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if any(type(op) in _COMPARE_SWAP for op in node.ops) and self._hit():
            ops: list[ast.cmpop] = []
            swapped = False
            for op in node.ops:
                if not swapped and type(op) in _COMPARE_SWAP:
                    ops.append(_COMPARE_SWAP[type(op)]())
                    swapped = True
                else:
                    ops.append(op)
            node.ops = ops
            self.description = f"compare {ast.unparse(node)}"
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in _BINOP_SWAP and self._hit():
            node.op = _BINOP_SWAP[type(node.op)]()
            self.description = f"binop {ast.unparse(node)}"
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub)) and self._hit():
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            self.description = f"augassign {ast.unparse(node)}"
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if self._hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.description = f"boolop {ast.unparse(node)}"
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit():
            self.description = f"drop not {ast.unparse(node)}"
            return node.operand
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit():
            self.description = f"constant {node.value} -> {node.value + 1}"
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node


def generate_mutants(source: str, *, limit: int = 30, seed: str = "") -> tuple[Mutant, ...]:
    """Distinct single-site mutants of ``source`` (deterministic for a given seed)."""

    tree = ast.parse(source)
    collector = _Collector()
    collector.visit(tree)
    original = ast.unparse(tree)
    mutants: list[Mutant] = []
    seen = {original}
    for site in range(len(collector.sites)):
        mutator = _Mutator(site)
        mutated = mutator.visit(copy.deepcopy(tree))
        ast.fix_missing_locations(mutated)
        if mutator.count < site:
            continue
        text = ast.unparse(mutated)
        if text in seen:
            continue
        seen.add(text)
        mutants.append(Mutant(index=len(mutants), description=mutator.description, source=text))
    if len(mutants) > limit:
        rng = random.Random(seed)
        mutants = sorted(rng.sample(mutants, limit), key=lambda m: m.index)
        mutants = [Mutant(i, m.description, m.source) for i, m in enumerate(mutants)]
    return tuple(mutants)
