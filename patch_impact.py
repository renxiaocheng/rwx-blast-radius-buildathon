"""Find old trace lines for edits restricted to an existing local expression."""

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import symtable
from typing import Protocol


class PatchEdit(Protocol):
    start: int
    removed: list[str]
    added: list[str]


@dataclass
class Changes:
    lines: set[int] = field(default_factory=set)
    gaps: set[int] = field(default_factory=set)

    def add(self, start: int, count: int) -> None:
        if count:
            self.lines.update(range(start, start + count))
        else:
            self.gaps.add(start)


def apply_edits(lines: Sequence[str], edits: Iterable[PatchEdit]) -> tuple[list[str], Changes, Changes] | None:
    patched = []
    previous_end = 0
    old_changes, new_changes = Changes(), Changes()
    for edit in edits:
        start = edit.start - 1
        end = start + len(edit.removed)
        if start < previous_end or end > len(lines) or list(lines[start:end]) != edit.removed:
            return None
        patched.extend(lines[previous_end:start])
        new_start = len(patched) + 1
        patched.extend(edit.added)
        old_changes.add(edit.start, len(edit.removed))
        new_changes.add(new_start, len(edit.added))
        previous_end = end
    patched.extend(lines[previous_end:])
    return patched, old_changes, new_changes


UNSAFE_EXPRESSION_NODES = (
    ast.NamedExpr, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await, ast.GeneratorExp,
    ast.ListComp, ast.SetComp, ast.DictComp,
)


def ordinary_expression(expression: ast.expr) -> bool:
    for node in ast.walk(expression):
        if isinstance(node, UNSAFE_EXPRESSION_NODES):
            return False
        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile", "globals", "locals"}:
            return False
    return True


class MaskedExpression(ast.expr):
    """A comparison marker that cannot appear in parsed Python code."""


class LocalExpressions(ast.NodeVisitor):
    """Erase changed expressions while preserving all other syntax for comparison."""

    VALUE_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr)

    def __init__(self, changes: Changes):
        self.changes = changes
        self.in_function = False
        self.lines = set()
        self.gaps = set()
        self.spans = []
        self.values = []

    @property
    def complete(self) -> bool:
        return self.lines == self.changes.lines and self.gaps == self.changes.gaps

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators, defaults, and annotations can execute at definition time.
        self.visit_body(node.body, in_function=True)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.visit_body(node.body, in_function=False)

    def visit_body(self, body: list[ast.stmt], *, in_function: bool) -> None:
        previous = self.in_function
        self.in_function = in_function
        for node in body:
            self.visit(node)
        self.in_function = previous

    def generic_visit(self, node: ast.AST) -> None:
        if self.in_function and isinstance(node, self.VALUE_STATEMENTS):
            # A function's leading string is visible as its docstring without
            # calling it. Standalone strings offer no useful trace anchor.
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                self.mask(node, "value", node.end_lineno)
        elif self.in_function and isinstance(node, (ast.If, ast.While)):
            self.mask(node, "test", node.test.end_lineno)
        elif self.in_function and isinstance(node, (ast.For, ast.AsyncFor)):
            self.mask(node, "iter", node.iter.end_lineno)
        elif self.in_function and isinstance(node, ast.Match):
            self.mask(node, "subject", node.subject.end_lineno)
        elif self.in_function and isinstance(node, ast.Raise):
            self.mask(node, "exc", node.end_lineno)
            self.mask(node, "cause", node.end_lineno)
        elif self.in_function and isinstance(node, ast.Assert):
            self.mask(node, "test", node.end_lineno)
            self.mask(node, "msg", node.end_lineno)
        super().generic_visit(node)

    def mask(self, node: ast.stmt, attribute: str, end: int) -> None:
        first = node.lineno
        affected_lines = {line for line in self.changes.lines if first <= line <= end}
        # A pure insertion has no old line. Both surrounding lines must still
        # be inside this same logical operation to anchor it to old coverage.
        affected_gaps = {gap for gap in self.changes.gaps if first < gap <= end}
        if not affected_lines and not affected_gaps:
            return

        expression = getattr(node, attribute)
        if expression is None or not ordinary_expression(expression):
            return
        self.lines.update(affected_lines)
        self.gaps.update(affected_gaps)
        self.spans.append(range(first, end + 1))
        self.values.append(ast.dump(expression))
        setattr(node, attribute, MaskedExpression())


def closure_signature(table: symtable.SymbolTable) -> tuple:
    # Changing a cell or free variable can affect an earlier operation in the
    # same scope, even on executions that never reach the changed expression.
    free = tuple(sorted(symbol.get_name() for symbol in table.get_symbols() if symbol.is_free()))
    return table.get_type(), table.get_name(), free, tuple(map(closure_signature, table.get_children()))


def statement_impact(
    name: str, lines: Sequence[str], edits: Iterable[PatchEdit]
) -> tuple[range, ...] | None:
    """Return executable old line ranges or defer to the whole function.

    Replacing a value or condition inside an existing statement can only
    affect a test once execution reaches that operation. Confirm the surrounding
    syntax is identical and use the complete old operation, including physical
    lines that can receive coverage for a joined or multiline expression.
    """
    result = apply_edits(lines, edits)
    if result is None:
        return None
    patched, old_changes, new_changes = result
    old_source = "\n".join(lines) + "\n"
    new_source = "\n".join(patched) + "\n"
    try:
        old_tree = ast.parse(old_source, name, type_comments=True)
        new_tree = ast.parse(new_source, name, type_comments=True)
    except (SyntaxError, ValueError, RecursionError):
        return None

    old_scope, new_scope = LocalExpressions(old_changes), LocalExpressions(new_changes)
    old_scope.visit(old_tree)
    new_scope.visit(new_tree)
    if not old_scope.complete or not new_scope.complete or old_scope.values == new_scope.values:
        return None
    if ast.dump(old_tree) != ast.dump(new_tree):
        return None
    try:
        old_closures = closure_signature(symtable.symtable(old_source, name, "exec"))
        new_closures = closure_signature(symtable.symtable(new_source, name, "exec"))
    except (SyntaxError, ValueError, RecursionError):
        return None
    if old_closures != new_closures:
        return None
    return tuple(dict.fromkeys(old_scope.spans)) or None
