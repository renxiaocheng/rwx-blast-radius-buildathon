"""Classify source-only edits and literal SQLGlot Expression metadata edits."""

import ast
from collections.abc import Iterable, Sequence
from importlib.util import resolve_name
from io import StringIO
import marshal
from pathlib import PurePosixPath
import re
import tokenize
from types import CodeType

from patch_impact import Changes, MaskedExpression, PatchEdit, apply_edits, statement_impact


ENCODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
EXPRESSION_BASE = "sqlglot.expressions.core.Expression"
EXPRESSION_EXPORTS = {
    EXPRESSION_BASE,
    "sqlglot.expressions.Expression",
    "sqlglot.exp.Expression",
}


def source_directive(line: str) -> bool:
    return line.startswith(("#!", "\ufeff")) or ENCODING.match(line) is not None


def code_signature(code: CodeType) -> tuple:
    # Format 2 keeps binary numbers and omits reference tags that depend on the
    # runtime's string interning and current reference counts.
    constants = tuple(
        code_signature(value) if isinstance(value, CodeType) else (type(value), marshal.dumps(value, 2))
        for value in code.co_consts
    )
    return (
        code.co_name, code.co_qualname, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags,
        code.co_code, code.co_exceptiontable, code.co_names, code.co_varnames,
        code.co_cellvars, code.co_freevars, constants,
    )


def comment_tokens(source: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (token.type, token.string)
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER}
    )


def compiled_equivalent(
    lines: Sequence[str], patched: Sequence[str], edits: Sequence[PatchEdit], old: CodeType, new: CodeType,
) -> bool:
    """Compare executable code and constants, leaving source locations to observers."""
    if any(source_directive(line) for edit in edits for line in (*edit.removed, *edit.added)):
        return False
    old_header = [(i, line) for i, line in enumerate(lines[:2]) if source_directive(line)]
    new_header = [(i, line) for i, line in enumerate(patched[:2]) if source_directive(line)]
    if old_header != new_header:
        return False
    try:
        if code_signature(old) == code_signature(new):
            return True
        # Python 3.13 stores a class's __firstlineno__ in its constants.
        # A comment can shift this metadata and is handled by source-reader checks.
        if not any(not line.strip() or "#" in line for edit in edits for line in (*edit.removed, *edit.added)):
            return False
        before = "\n".join(lines) + "\n"
        after = "\n".join(patched) + "\n"
        return comment_tokens(before) == comment_tokens(after) and ast.dump(
            ast.parse(before, type_comments=True),
        ) == ast.dump(ast.parse(after, type_comments=True))
    except (SyntaxError, TypeError, ValueError, RecursionError, OverflowError, tokenize.TokenError):
        return False


def changed_test_methods(name: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> set[str] | None:
    """Recognize local statement edits inside existing pytest test functions."""
    path = PurePosixPath(name)
    if path.suffix != ".py" or not path.stem.startswith("test_"):
        return None
    spans = statement_impact(name, lines, edits)
    if not spans:
        return None
    try:
        tree = ast.parse("\n".join(lines) + "\n", name)
    except (SyntaxError, ValueError, RecursionError):
        return None
    tests = []
    for node in tree.body:
        members = node.body if isinstance(node, ast.ClassDef) else (node,)
        tests.extend(
            item for item in members
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_")
        )
    methods = set()
    for span in spans:
        current = {
            test.name for test in tests
            if test.body[0].lineno <= span.start and span.stop - 1 <= test.end_lineno
        }
        if not current:
            return None
        methods.update(current)
    return methods


def qualified_reference(node: ast.expr, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        prefix = qualified_reference(node.value, bindings)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def import_bindings(node: ast.Import | ast.ImportFrom, package: str, bindings: dict[str, str]) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname:
                bindings[alias.asname] = alias.name
            else:
                root = alias.name.partition(".")[0]
                bindings[root] = root
        return

    module = node.module or ""
    if node.level:
        try:
            module = resolve_name("." * node.level + module, package)
        except ImportError:
            bindings.clear()
            return
    for alias in node.names:
        if alias.name == "*":
            bindings.clear()
        else:
            bindings[alias.asname or alias.name] = f"{module}.{alias.name}"


def invalidate_bindings(node: ast.stmt, bindings: dict[str, str]) -> None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        bindings.pop(node.name, None)
        return
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bindings.pop(child.id, None)
        elif isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings.pop(child.name, None)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                if alias.name == "*":
                    bindings.clear()
                else:
                    bindings.pop(alias.asname or alias.name.partition(".")[0], None)


def arg_types_assignment(node: ast.ClassDef) -> ast.Assign | ast.AnnAssign | None:
    candidates = []
    for child in node.body:
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
        elif isinstance(child, ast.AnnAssign) and child.simple:
            target = child.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "arg_types":
            candidates.append(child)
    return candidates[0] if len(candidates) == 1 else None


def literal_arg_types(value: ast.expr | None) -> bool:
    # Expr.__init_subclass__ reads this dictionary to derive required_args.
    # Keep the schema expected by that method and exclude executed expressions.
    if not isinstance(value, ast.Dict):
        return False
    keys = {
        key.value for key in value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    return len(keys) == len(value.keys) and all(
        isinstance(item, ast.Constant) and type(item.value) is bool for item in value.values
    )


def mask_arg_types(tree: ast.Module, module: str, changes: Changes) -> dict[str, str] | None:
    bindings = {}
    values = {}
    covered_lines, covered_gaps = set(), set()
    package = module.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_bindings(node, package, bindings)
            continue
        if (
            isinstance(node, ast.ClassDef)
            and len(node.bases) == 1
            and not node.decorator_list
            and not node.keywords
            and qualified_reference(node.bases[0], bindings) in EXPRESSION_EXPORTS
        ):
            assignment = arg_types_assignment(node)
            if assignment is not None and literal_arg_types(assignment.value):
                first, last = assignment.lineno, assignment.end_lineno
                within = {line for line in changes.lines if first <= line <= last}
                gaps = {gap for gap in changes.gaps if first < gap <= last}
                if within or gaps:
                    name = f"{module}.{node.name}"
                    if name in values:
                        return None
                    values[name] = ast.dump(assignment.value)
                    assignment.value = MaskedExpression()
                    covered_lines.update(within)
                    covered_gaps.update(gaps)
        invalidate_bindings(node, bindings)
        if isinstance(node, ast.ClassDef) and f"{module}.{node.name}" == EXPRESSION_BASE:
            bindings[node.name] = EXPRESSION_BASE

    if covered_lines != changes.lines or covered_gaps != changes.gaps:
        return None
    return values


def expression_arg_types_impact(
    name: str, lines: Sequence[str], edits: Iterable[PatchEdit]
) -> set[str] | None:
    """Identify Expression classes whose existing arg_types literals changed."""
    path = PurePosixPath(name)
    if path.suffix != ".py" or path.parts[:2] != ("sqlglot", "expressions") or path.stem == "__init__":
        return None
    module = ".".join(path.with_suffix("").parts)
    result = apply_edits(lines, edits)
    if result is None:
        return None
    patched, original_changes, modified_changes = result
    try:
        original = ast.parse("\n".join(lines) + "\n", name, type_comments=True)
        modified = ast.parse("\n".join(patched) + "\n", name, type_comments=True)
        original_values = mask_arg_types(original, module, original_changes)
        modified_values = mask_arg_types(modified, module, modified_changes)
        if not original_values or not modified_values or original_values.keys() != modified_values.keys():
            return None
        if ast.dump(original) != ast.dump(modified):
            return None
        return {name for name, value in original_values.items() if value != modified_values[name]} or None
    except (SyntaxError, ValueError, RecursionError):
        return None
