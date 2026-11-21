"""Classify source-independent edits to a single pinned mapping entry."""

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath
import tokenize

from dialect_impact import (
    _comments,
    _constant_changes,
    _factory_arguments,
    _factory_binding,
    _literal_shape,
    _module_name,
    _returns_deferred_function,
)
from patch_impact import PatchEdit, apply_edits


@dataclass(frozen=True)
class MappingEdit:
    origin: str
    field: str
    kind: str
    key: str


def _name(assignment: ast.stmt) -> str | None:
    if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1:
        target = assignment.targets[0]
    elif isinstance(assignment, ast.AnnAssign) and assignment.value is not None:
        target = assignment.target
    else:
        return None
    if isinstance(target, ast.Name) and target.id.isupper():
        return target.id
    return None


def declarations(tree: ast.Module, module: str) -> dict[str, ast.Assign | ast.AnnAssign | None]:
    found = {}

    def visit(body, scope):
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                if not statement.decorator_list and not statement.keywords:
                    visit(statement.body, f"{scope}.{statement.name}")
                continue
            field = _name(statement)
            if field is not None:
                key = f"{scope}.{field}"
                found[key] = statement if key not in found else None

    visit(tree.body, module)
    return found


def key_label(key: ast.expr) -> str | None:
    if isinstance(key, ast.Constant) and type(key.value) is str:
        return ast.dump(key)
    if isinstance(key, ast.Attribute):
        base = key.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name):
            return ast.dump(key)
    return None


def _plain_value(old: ast.expr, new: ast.expr) -> bool:
    try:
        return _literal_shape(ast.literal_eval(old)) == _literal_shape(ast.literal_eval(new))
    except (SyntaxError, TypeError, ValueError):
        return False


def _lambda_factory(factory: ast.FunctionDef) -> bool:
    body = factory.body[1:] if ast.get_docstring(factory) is not None else factory.body
    if (
        factory.decorator_list or getattr(factory, "type_params", ()) or len(body) != 1
        or not isinstance(body[0], ast.Return) or not isinstance(body[0].value, ast.Lambda)
    ):
        return False
    args = body[0].value.args
    return not args.defaults and all(default is None for default in args.kw_defaults)


class MappingImpact:
    def __init__(self, source: Path):
        self.source = source
        self.trees: dict[str, ast.Module | None] = {}

    def _tree(self, path: str) -> ast.Module | None:
        if path not in self.trees:
            try:
                self.trees[path] = ast.parse((self.source / path).read_bytes(), filename=path, type_comments=True)
            except (OSError, SyntaxError, ValueError):
                self.trees[path] = None
        return self.trees[path]

    def _factory(self, path: str, name: str, before: int) -> bool:
        module = self._tree(path)
        if module is None:
            return False
        binding = _factory_binding(module, name)
        if binding is None or binding.lineno >= before:
            return False
        if isinstance(binding, ast.ImportFrom):
            imports = [alias for alias in binding.names if (alias.asname or alias.name) == name]
            if len(imports) != 1:
                return False
            imported = binding.module or ""
            if binding.level:
                try:
                    package = _module_name(path).rpartition(".")[0]
                    imported = resolve_name("." * binding.level + imported, package)
                except ImportError:
                    return False
            if imported != "sqlglot" and not imported.startswith("sqlglot."):
                return False
            relative = imported.replace(".", "/")
            module = self._tree(f"{relative}/__init__.py") or self._tree(f"{relative}.py")
            if module is None:
                return False
            binding = _factory_binding(module, imports[0].name)
        return isinstance(binding, ast.FunctionDef) and (
            _lambda_factory(binding) or _returns_deferred_function(binding, module)
        )

    def _dict_change(self, path: str, old: ast.Dict, new: ast.Dict, line: int) -> str | None:
        if len(old.keys) != len(new.keys) or any(
            ast.dump(before) != ast.dump(after) if before is not None and after is not None
            else before is not after
            for before, after in zip(old.keys, new.keys)
        ):
            return None
        changed = []
        for key, before, after in zip(old.keys, old.values, new.values):
            if ast.dump(before) == ast.dump(after):
                continue
            label = key_label(key)
            if label is None:
                return None
            if _plain_value(before, after):
                changed.append(label)
            elif isinstance(before, ast.Lambda) and isinstance(after, ast.Lambda) and _constant_changes(before, after):
                changed.append(label)
            elif (
                isinstance(before, ast.Call) and isinstance(after, ast.Call)
                and isinstance(before.func, ast.Name) and self._factory(path, before.func.id, line)
                and _factory_arguments(before, after)
            ):
                changed.append(label)
            else:
                return None
        return changed[0] if len(changed) == 1 else None

    def changes(self, path: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> set[MappingEdit]:
        filename = PurePosixPath(path)
        if filename.parts[:1] != ("sqlglot",) or filename.suffix != ".py" or ".." in filename.parts:
            return set()
        result = apply_edits(lines, edits)
        if result is None:
            return set()
        patched, before, after = result
        old_source, new_source = ("\n".join(source) + "\n" for source in (lines, patched))
        try:
            if [text for _, text in _comments(old_source)] != [text for _, text in _comments(new_source)]:
                return set()
            old_tree = ast.parse(old_source, path, type_comments=True)
            new_tree = ast.parse(new_source, path, type_comments=True)
        except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
            return set()
        module = _module_name(path)
        old_assignments, new_assignments = declarations(old_tree, module), declarations(new_tree, module)
        changes = set()
        affected_ranges = [[], []]
        for origin, old in old_assignments.items():
            new = new_assignments.get(origin)
            if old is None or new is None or ast.dump(old.value) == ast.dump(new.value):
                continue
            if isinstance(old.value, ast.Dict) and isinstance(new.value, ast.Dict):
                kind, key = "dict", self._dict_change(path, old.value, new.value, old.lineno)
            elif isinstance(old.value, ast.Set) and isinstance(new.value, ast.Set):
                original = {key_label(item) for item in old.value.elts if isinstance(item, ast.Constant)}
                modified = {key_label(item) for item in new.value.elts if isinstance(item, ast.Constant)}
                delta = original ^ modified
                if None in original | modified or len(original) != len(old.value.elts) or len(modified) != len(new.value.elts) or len(delta) != 1:
                    return set()
                kind, key = "set", next(iter(delta))
            else:
                return set()
            if key is None:
                return set()
            changes.add(MappingEdit(origin, origin.rsplit(".", 1)[-1], kind, key))
            for ranges, assignment in zip(affected_ranges, (old, new)):
                ranges.append((assignment.lineno, assignment.end_lineno))
            new.value = old.value
        if ast.dump(old_tree) != ast.dump(new_tree):
            return set()
        for change, ranges in zip((before, after), affected_ranges):
            if any(not any(start <= line <= stop for start, stop in ranges) for line in change.lines):
                return set()
            if any(not any(start < gap <= stop for start, stop in ranges) for gap in change.gaps):
                return set()
        return changes
