"""Bound exact integer guard inputs without capping the number of visits."""

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import operator
from pathlib import Path
import symtable
import tokenize
from io import StringIO

from patch_impact import PatchEdit, apply_edits, closure_signature


PROBE = "__rwx_branch_input__"
VERSION = 1
MAX_INTEGER_BITS = 80
OPERATORS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq, "ne": operator.ne}
AST_OPERATORS = {
    ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne",
}
DEFAULT_PATHS = frozenset({"sqlglot/optimizer/normalize.py"})
UNSEEN = object()


def int_literal(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value.bit_length() <= MAX_INTEGER_BITS:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = node.operand
        if isinstance(inner, ast.Constant) and type(inner.value) is int and inner.value.bit_length() <= MAX_INTEGER_BITS:
            return -inner.value if isinstance(node.op, ast.USub) else inner.value
    return None


def simple_operand(node: ast.expr) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) or int_literal(node) is not None


def operation(node: ast.Compare) -> str | None:
    if len(node.ops) != 1 or not simple_operand(node.left) or not simple_operand(node.comparators[0]):
        return None
    return AST_OPERATORS.get(type(node.ops[0]))


def default_include(path: str, comparison: ast.Compare) -> bool:
    """Prioritize ordered comparisons and syntactically numeric equalities."""
    op = operation(comparison)
    return op is not None and (
        op not in {"eq", "ne"}
        or int_literal(comparison.left) is not None
        or int_literal(comparison.comparators[0]) is not None
    )


def site_name(path: str, node: ast.Compare, op: str) -> str:
    return f"{path}:{node.lineno}:{node.col_offset}:{op}"


def guard_comparisons(expression: ast.expr):
    if isinstance(expression, ast.Compare):
        yield expression
    elif isinstance(expression, ast.BoolOp):
        for part in expression.values:
            yield from guard_comparisons(part)
    elif isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        yield from guard_comparisons(expression.operand)


class GuardSites(ast.NodeVisitor):
    def __init__(self, tree: ast.Module):
        self.in_function = False
        self.sites = []
        self.visit(tree)

    def visit_body(self, statements: list[ast.stmt], in_function: bool):
        previous = self.in_function
        self.in_function = in_function
        try:
            for statement in statements:
                self.visit(statement)
        finally:
            self.in_function = previous

    def visit_FunctionDef(self, node):
        self.visit_body(node.body, True)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.visit_body(node.body, False)

    def visit_If(self, node):
        if self.in_function:
            self.sites.extend(compare for compare in guard_comparisons(node.test) if operation(compare) is not None)
        self.generic_visit(node)

    visit_While = visit_If
    visit_IfExp = visit_If
    visit_Assert = visit_If


@dataclass
class BranchEvidence:
    """Aggregate original operand ranges in three difference-sign partitions."""

    sites: dict[str, dict[str, list[list[int] | None] | None]] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)
    unsupported: bool = False

    def register(self, site: str) -> None:
        self.sites.setdefault(site, {})

    def observe(self, site: str, node: str | None, left, right) -> None:
        if node is None:
            self.shared.add(site)
            return
        if site in self.shared:
            return
        tests = self.sites[site]
        bands = tests.get(node, UNSEEN)
        if bands is None:
            return
        if type(left) is not int or type(right) is not int or left.bit_length() > MAX_INTEGER_BITS or right.bit_length() > MAX_INTEGER_BITS:
            tests[node] = None
            return
        if bands is UNSEEN:
            bands = tests[node] = [None, None, None]
        delta = left - right
        index = 0 if delta < 0 else 2 if delta > 0 else 1
        old = bands[index]
        if old is None:
            bands[index] = [delta, delta, left, left, right, right]
        else:
            self.expand(old, delta, left, right)

    @staticmethod
    def expand(bounds: list[int], delta: int, left: int, right: int):
        if delta < bounds[0]: bounds[0] = delta
        if delta > bounds[1]: bounds[1] = delta
        if left < bounds[2]: bounds[2] = left
        if left > bounds[3]: bounds[3] = left
        if right < bounds[4]: bounds[4] = right
        if right > bounds[5]: bounds[5] = right

    def export(self) -> dict:
        return {
            "version": VERSION,
            "sites": {site: rows if site not in self.shared else {} for site, rows in sorted(self.sites.items())},
            "shared": sorted(self.shared),
            "unsupported": self.unsupported,
        }

    def merge(self, data: Mapping | None) -> None:
        if not data or data.get("version") != VERSION:
            self.unsupported = True
            return
        self.unsupported |= data.get("unsupported", False)
        self.shared.update(data.get("shared", ()))
        for site, additions in data.get("sites", {}).items():
            tests = self.sites.setdefault(site, {})
            if site in self.shared:
                tests.clear()
                continue
            for node, bands in additions.items():
                old = tests.get(node, UNSEEN)
                if bands is None or old is None:
                    tests[node] = None
                elif old is UNSEEN:
                    tests[node] = [None if bounds is None else list(bounds) for bounds in bands]
                else:
                    for index, bounds in enumerate(bands):
                        if bounds is None:
                            continue
                        if old[index] is None:
                            old[index] = list(bounds)
                        else:
                            self.expand(old[index], bounds[0], bounds[2], bounds[4])
                            self.expand(old[index], bounds[1], bounds[3], bounds[5])


def branch_instrumentation(
    source: Path,
    evidence: BranchEvidence,
    current_node: Callable[[], str | None],
    *,
    paths: Iterable[str] = DEFAULT_PATHS,
    include: Callable[[str, ast.Compare], bool] = default_include,
    enabled: Callable[[], bool] | None = None,
):
    """Install probes in selected source paths through the existing AST loader."""
    paths = frozenset(paths)
    source = Path(source)

    def record(site, op, left, right):
        if enabled is None or enabled():
            evidence.observe(site, current_node(), left, right)
        # Forward the old operation exactly once, including user-defined
        # dispatch. Unsupported operands never prove a test equivalent.
        return OPERATORS[op](left, right)

    def transform(path: str, tree: ast.Module) -> ast.Module:
        if path not in paths or not (source / path).is_file():
            return tree
        replacements = {}
        for node in GuardSites(tree).sites:
            if not include(path, node):
                continue
            op = operation(node)
            site = site_name(path, node, op)
            evidence.register(site)
            replacements[node] = site, op

        class Instrument(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                chosen = replacements.get(node)
                if chosen is None:
                    return node
                site, op = chosen
                return ast.copy_location(ast.Call(
                    func=ast.Name(id=PROBE, ctx=ast.Load()),
                    args=[ast.Constant(site), ast.Constant(op), node.left, node.comparators[0]], keywords=[],
                ), node)

        return Instrument().visit(tree) if replacements else tree

    return transform, {PROBE: record}


@dataclass(frozen=True)
class GuardEdit:
    site: str
    previous: str
    current: str
    left_constant: int | None = None
    right_constant: int | None = None


def comments(text: str) -> list[tuple[int, str]]:
    return [(token.start[0], token.string) for token in tokenize.generate_tokens(StringIO(text).readline) if token.type == tokenize.COMMENT]


def bound_argument(node: ast.Name, parents: Mapping[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
        if isinstance(current, (ast.Lambda, ast.ClassDef)):
            return False
    else:
        return False
    if not any(isinstance(arg, ast.arg) and arg.arg == node.id for arg in ast.iter_child_nodes(current.args)):
        return False
    return not any(
        (isinstance(part, ast.Name) and part.id == node.id and isinstance(part.ctx, ast.Del))
        or (isinstance(part, ast.ExceptHandler) and part.name == node.id)
        or (isinstance(part, ast.Name) and part.id in {"exec", "eval", "locals", "_getframe", "getframeinfo"})
        for part in ast.walk(current)
    )


def replacement_operand(old: ast.expr, new: ast.expr, parents: Mapping[ast.AST, ast.AST]) -> tuple[bool, int | None]:
    if ast.dump(old) == ast.dump(new):
        return True, None
    constant = int_literal(new)
    # A later visit can raise before the probe is called. Require removed
    # lookups to be literals or parameters that cannot be unbound in scope.
    pure = int_literal(old) is not None or isinstance(old, ast.Name) and bound_argument(old, parents)
    return constant is not None and pure, constant


def branch_changes(path: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> tuple[GuardEdit, ...]:
    result = apply_edits(lines, edits)
    if result is None:
        return ()
    patched, old_changes, new_changes = result
    if not old_changes.lines or not new_changes.lines or old_changes.gaps or new_changes.gaps:
        return ()
    old_text, new_text = ("\n".join(parts) + "\n" for parts in (lines, patched))
    try:
        if comments(old_text) != comments(new_text):
            return ()
        trees = [ast.parse(text, path, type_comments=True) for text in (old_text, new_text)]
        matched = []
        for tree, change in zip(trees, (old_changes, new_changes)):
            comparisons = [
                node for node in GuardSites(tree).sites
                if any(node.lineno <= line <= node.end_lineno for line in change.lines)
            ]
            if not comparisons or not all(
                any(node.lineno <= line <= node.end_lineno for node in comparisons) for line in change.lines
            ):
                return ()
            matched.append(comparisons)
        if len(matched[0]) != len(matched[1]):
            return ()

        changes = []
        parents = {child: node for node in ast.walk(trees[0]) for child in ast.iter_child_nodes(node)}
        # Only the selected comparison nodes may differ. Match old/new sites
        # in AST traversal order and confirm the unedited structure below.
        for before, after in zip(*matched):
            previous, current = operation(before), operation(after)
            valid_left, left = replacement_operand(before.left, after.left, parents)
            valid_right, right = replacement_operand(before.comparators[0], after.comparators[0], parents)
            if not valid_left or not valid_right:
                return ()
            if previous != current or left is not None or right is not None:
                changes.append(GuardEdit(site_name(path, before, previous), previous, current, left, right))
            after.left, after.ops, after.comparators = before.left, before.ops, before.comparators
        if not changes or ast.dump(trees[0]) != ast.dump(trees[1]):
            return ()
        if closure_signature(symtable.symtable(old_text, path, "exec")) != closure_signature(symtable.symtable(new_text, path, "exec")):
            return ()
        return tuple(changes)
    except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
        return ()


def interval_truth(op: str, low: int, high: int) -> bool | None:
    if op in {"eq", "ne"}:
        if low == high == 0:
            return op == "eq"
        if low > 0 or high < 0:
            return op == "ne"
        return None
    comparator = OPERATORS[op]
    first, last = comparator(low, 0), comparator(high, 0)
    return first if first is last else None


def equivalent(change: GuardEdit, bands: Sequence | None) -> bool:
    if not bands or not any(band is not None for band in bands):
        return False
    for bounds in bands:
        if bounds is None:
            continue
        d_min, d_max, l_min, l_max, r_min, r_max = bounds
        before = interval_truth(change.previous, d_min, d_max)
        left, right = change.left_constant, change.right_constant
        if left is None and right is None:
            lower, upper = d_min, d_max
        elif right is None:
            lower, upper = left - r_max, left - r_min
        elif left is None:
            lower, upper = l_min - right, l_max - right
        else:
            lower = upper = left - right
        if before is None or interval_truth(change.current, lower, upper) is not before:
            return False
    return True


def branch_tests(
    changes: Iterable[GuardEdit], candidates: set[str], evidence: BranchEvidence, *, complete: frozenset[str] = frozenset(),
) -> set[str] | None:
    """Keep affected and unknown tests; callers retain source/reflection readers."""
    changes = tuple(changes)
    if not changes or evidence.unsupported or any(edit.site not in evidence.sites or edit.site in evidence.shared for edit in changes):
        return None
    proven = set(candidates)
    for edit in changes:
        rows = evidence.sites[edit.site]
        proven = {
            node for node in proven
            if (node in complete and node not in rows) or equivalent(edit, rows.get(node))
        }
    return candidates - proven
