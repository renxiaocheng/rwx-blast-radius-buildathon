"""Conservative test selection for pinned SQLGlot dialect data assignments."""

import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
from importlib.util import resolve_name
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import tokenize
from typing import Callable, Iterable, Protocol


_FAMILY_PACKAGES = {"dialects", "parsers", "generators", "tokenizers", "typing"}
_INDEX_VERSION = 1
_REGISTRY_NAMES = {
    "Dialects",
    "DIALECT_MODULE_NAMES",
    "MODULE_BY_DIALECT",
    "MODULE_BY_ATTRIBUTE",
    "_classes",
    "classes",
    "iter_modules",
    "walk_packages",
    "import_module",
}


class Edit(Protocol):
    start: int
    removed: list[str]
    added: list[str]


def _module_name(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _family(path: str, dialects: set[str]) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) == 3 and parts[0] == "sqlglot" and parts[1] in _FAMILY_PACKAGES:
        name = parts[2].removesuffix(".py")
        if name in dialects:
            return name
    return None


def _imports(path: str, nodes: Iterable[ast.AST]) -> set[str]:
    module = _module_name(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    imports = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parent = node.module or ""
            if node.level:
                try:
                    parent = resolve_name("." * node.level + parent, package)
                except ImportError:
                    continue
            imports.add(parent)
            imports.update(f"{parent}.{alias.name}" for alias in node.names)
            if parent == "sqlglot.dialects":
                imports.update(f"{parent}.{alias.name.casefold()}" for alias in node.names)
    return imports


def _read_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _references(path: str, tree: ast.AST) -> set[str]:
    nodes = tuple(ast.walk(tree))
    names = {name for node in nodes if (name := _read_name(node)) is not None}
    return names | _imports(path, nodes)


def _test_surfaces(path: str, tree: ast.Module) -> tuple[set[str], dict[str, set[str]]]:
    shared = set()
    methods = {}

    def visit(body: list[ast.stmt], scope: tuple[str, ...]):
        for statement in body:
            if isinstance(statement, ast.ClassDef):
                for part in (*statement.bases, *statement.keywords, *statement.decorator_list):
                    shared.update(_references(path, part))
                visit(statement.body, (*scope, statement.name))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test_"):
                qualified = "::".join((*scope, statement.name))
                methods[qualified] = _references(path, statement)
            else:
                # A helper, fixture, or module-level import can affect tests
                # in this file that do not refer to it directly.
                shared.update(_references(path, statement))

    visit(tree.body, ())
    return shared, methods


def _assignments(body, scope=()):
    assignments = {}
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            assignments.update(_assignments(statement.body, (*scope, statement.name)))
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target = statement.target
        else:
            continue
        if isinstance(target, ast.Name):
            name = (*scope, target.id)
            assignments[name] = None if name in assignments else statement
    return assignments


def _literal_shape(value):
    if isinstance(value, dict):
        return dict, [(type(key), key, _literal_shape(item)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return type(value), [_literal_shape(item) for item in value]
    if isinstance(value, set):
        # Changing set membership can affect import-time derivations.
        return set, value
    return type(value)


def _constant_changes(old, new) -> set[int] | None:
    """Return old lines with replaced constants, or reject any structural change."""
    if type(old) is not type(new):
        return None
    if isinstance(old, ast.Constant):
        if type(old.value) is not type(new.value) or old.kind != new.kind:
            return None
        return {old.lineno} if ast.dump(old) != ast.dump(new) else set()
    if isinstance(old, ast.Lambda) and ast.dump(old.args) != ast.dump(new.args):
        return None
    if isinstance(old, ast.AST):
        pairs = ((value, getattr(new, key)) for key, value in ast.iter_fields(old))
    elif isinstance(old, list):
        if len(old) != len(new):
            return None
        pairs = zip(old, new)
    else:
        return set() if old == new else None

    lines = set()
    for before, after in pairs:
        changes = _constant_changes(before, after)
        if changes is None:
            return None
        lines.update(changes)
    return lines


def _factory_arguments(old: ast.Call, new: ast.Call) -> set[int] | None:
    if ast.dump(old.func) != ast.dump(new.func):
        return None
    if [keyword.arg for keyword in old.keywords] != [keyword.arg for keyword in new.keywords]:
        return None
    if any(keyword.arg is None for keyword in old.keywords):
        return None
    if len(old.args) != len(new.args):
        return None

    old_values = [*old.args, *(keyword.value for keyword in old.keywords)]
    new_values = [*new.args, *(keyword.value for keyword in new.keywords)]
    if not all(isinstance(value, ast.Constant) for value in (*old_values, *new_values)):
        return None
    return _constant_changes(old_values, new_values)


def _deferred_changes(
    old: ast.expr, new: ast.expr, is_factory: Callable[[str], bool]
) -> set[int] | None:
    if not isinstance(old, ast.Dict) or not isinstance(new, ast.Dict):
        return None
    if len(old.keys) != len(new.keys) or any(
        (before is None) != (after is None)
        or (before is not None and ast.dump(before) != ast.dump(after))
        for before, after in zip(old.keys, new.keys)
    ):
        return None

    lines = set()
    for key, before, after in zip(old.keys, old.values, new.values):
        if ast.dump(before) == ast.dump(after):
            continue
        if key is None:
            return None
        if isinstance(before, ast.Lambda) and isinstance(after, ast.Lambda):
            changes = _constant_changes(before, after)
        elif (
            isinstance(before, ast.Call)
            and isinstance(after, ast.Call)
            and isinstance(before.func, ast.Name)
            and is_factory(before.func.id)
        ):
            changes = _factory_arguments(before, after)
        else:
            return None
        if not changes:
            return None
        lines.update(changes)
    return lines


def _factory_binding(tree: ast.Module, name: str) -> ast.FunctionDef | ast.ImportFrom | None:
    bindings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            bindings.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "*" for alias in node.names):
                return None
            if any((alias.asname or alias.name.partition(".")[0]) == name for alias in node.names):
                bindings.append(node)
        elif isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            return None
        elif isinstance(node, ast.arg) and node.arg == name:
            return None
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return None
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name:
            return None
        elif isinstance(node, ast.MatchMapping) and node.rest == name:
            return None
    if len(bindings) != 1 or bindings[0] not in tree.body:
        return None
    binding = bindings[0]
    return binding if isinstance(binding, (ast.FunctionDef, ast.ImportFrom)) else None


def _returns_deferred_function(factory: ast.FunctionDef, module: ast.Module) -> bool:
    body = factory.body
    if ast.get_docstring(factory) is not None:
        body = body[1:]
    if factory.decorator_list or getattr(factory, "type_params", ()) or len(body) != 2:
        return False

    inner, returned = body
    if not isinstance(inner, ast.FunctionDef) or inner.decorator_list or getattr(inner, "type_params", ()):
        return False
    if inner.args.defaults or any(default is not None for default in inner.args.kw_defaults):
        return False
    deferred_annotations = any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in module.body
    )
    if not deferred_annotations and (
        inner.returns is not None
        or any(isinstance(node, ast.arg) and node.annotation is not None for node in ast.walk(inner.args))
    ):
        return False
    return (
        isinstance(returned, ast.Return)
        and isinstance(returned.value, ast.Name)
        and returned.value.id == inner.name
    )


def _comments(source: str) -> list[tuple[int, str]]:
    return [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def _data_changes(
    tree: ast.Module,
    original_lines: list[str],
    edits: list[Edit],
    is_factory: Callable[[str, int], bool],
) -> set[tuple[str, ...]]:
    patched = original_lines.copy()
    for edit in reversed(edits):
        start = edit.start - 1
        end = start + len(edit.removed)
        if not (0 <= start < end <= len(original_lines)) or len(edit.removed) != len(edit.added):
            return set()
        if original_lines[start:end] != edit.removed:
            return set()
        patched[start:end] = edit.added

    old_assignments = _assignments(tree.body)
    assignment_ranges = [(node.lineno, node.end_lineno) for node in old_assignments.values() if node is not None]
    if not all(
        any(first <= line <= last for first, last in assignment_ranges)
        for edit in edits for line in range(edit.start, edit.start + len(edit.removed))
    ):
        return set()

    amended_source = "\n".join(patched) + "\n"
    try:
        amended = ast.parse(amended_source)
    except (SyntaxError, ValueError):
        return set()
    new_assignments = _assignments(amended.body)
    fields = set()
    modified_ranges = []
    rewrites_deferred_value = False
    for key, old in old_assignments.items():
        new = new_assignments.get(key)
        if old is None or new is None or ast.dump(old.value) == ast.dump(new.value):
            continue
        try:
            same_shape = _literal_shape(ast.literal_eval(old.value)) == _literal_shape(ast.literal_eval(new.value))
        except (ValueError, TypeError, SyntaxError):
            same_shape = False
        if same_shape:
            modified_ranges.append((old.lineno, old.end_lineno))
        elif len(key) > 1:
            deferred_lines = _deferred_changes(old.value, new.value, lambda name: is_factory(name, old.lineno))
            if not deferred_lines:
                return set()
            modified_ranges.extend((line, line) for line in deferred_lines)
            rewrites_deferred_value = True
        else:
            return set()
        fields.add(key)
        new.value = old.value

    # Names, decorators, imports, and control flow must all be exactly as they
    # were. Keep edits outside the changed assignments on the broad path.
    if ast.dump(tree) != ast.dump(amended):
        return set()
    if rewrites_deferred_value:
        try:
            if _comments("\n".join(original_lines) + "\n") != _comments(amended_source):
                return set()
        except tokenize.TokenError:
            return set()
    if not all(
        any(first <= line <= last for first, last in modified_ranges)
        for edit in edits for line in range(edit.start, edit.start + len(edit.removed))
    ):
        return set()
    return fields


class DialectImpact:
    def __init__(
        self,
        source: Path,
        coverage_lines: dict[str, dict[int, set[int]]],
        catalog: Iterable[dict],
        *,
        index_path: Path | None = None,
    ):
        self.source = source
        self.coverage_lines = coverage_lines
        self.catalog = tuple(catalog)
        self.index_path = index_path
        self.dialects = {
            path.stem for path in (source / "sqlglot" / "dialects").glob("*.py")
            if path.stem not in {"__init__", "dialect"}
        }
        self._trees: dict[str, ast.Module | None] = {}
        self._indexed = False
        self.family_tests = {}
        self.field_tests = {}

    def _paths(self) -> set[str]:
        paths = {path.relative_to(self.source).as_posix() for path in (self.source / "sqlglot").rglob("*.py")}
        paths.update(test["file"] for test in self.catalog)
        return paths

    def _tree(self, path: str) -> ast.Module | None:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse((self.source / path).read_bytes(), filename=path)
            except (OSError, SyntaxError, ValueError):
                self._trees[path] = None
        return self._trees[path]

    def _source_digests(self, paths: set[str]) -> dict[str, str]:
        return {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest() for name in sorted(paths)}

    def build_index(self) -> dict:
        """Collect static facts from the pinned source and catalog test files."""
        paths = self._paths()
        trees = {name: self._tree(name) for name in sorted(paths)}
        if not all(tree is not None for tree in trees.values()):
            raise ValueError("Could not parse every pinned module and catalog test file")

        fields = {
            assignment[-1]
            for path, tree in trees.items()
            if _family(path, self.dialects) is not None
            for assignment in _assignments(tree.body)
        }

        def relevant(references: set[str]) -> list[str]:
            return sorted(
                name for name in references
                if name in fields or name in _REGISTRY_NAMES or name.startswith("sqlglot.")
                or name.casefold().partition(",")[0].strip() in self.dialects
            )

        import_points = {}
        surfaces = {}
        field_lines = defaultdict(lambda: defaultdict(set))
        for path, tree in trees.items():
            import_points[path] = [
                [node.lineno, sorted(_imports(path, (node,)))]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            if path.startswith("tests/"):
                shared, methods = _test_surfaces(path, tree)
                surfaces[path] = [relevant(shared), {name: relevant(refs) for name, refs in methods.items()}]
                continue
            for node in ast.walk(tree):
                name = _read_name(node)
                if name in fields:
                    field_lines[name][path].update(range(node.lineno, node.end_lineno + 1))

        return {
            "version": _INDEX_VERSION,
            "source": self._source_digests(paths),
            "imports": import_points,
            "test_surfaces": surfaces,
            "field_lines": {
                field: {path: sorted(lines) for path, lines in by_file.items()}
                for field, by_file in field_lines.items()
            },
        }

    def _index_data(self) -> dict:
        if self.index_path is not None:
            opener = gzip.open if self.index_path.suffix == ".gz" else open
            try:
                with opener(self.index_path, "rt", encoding="utf-8") as stream:
                    index = json.load(stream)
                if (
                    index.get("version") == _INDEX_VERSION
                    and all(isinstance(index.get(key), dict) for key in ("imports", "test_surfaces", "field_lines"))
                    and index.get("source") == self._source_digests(self._paths())
                ):
                    return index
            except (OSError, ValueError, AttributeError):
                pass
        try:
            return self.build_index()
        except (OSError, ValueError):
            return {"imports": {}, "test_surfaces": {}, "field_lines": {}}

    def _load(self) -> None:
        if self._indexed:
            return
        self._indexed = True
        index = self._index_data()
        self.import_points = index["imports"]
        self.field_lines = index["field_lines"]
        self.imports = {
            path: set().union(*(imports for _, imports in points))
            for path, points in self.import_points.items()
        }
        surfaces = index["test_surfaces"]
        self.test_references = {}
        for test in self.catalog:
            if test["file"] not in surfaces:
                continue
            shared, methods = surfaces[test["file"]]
            name = test["id"].partition("::")[2].partition("[")[0]
            references = methods.get(name)
            if references is None:
                references = set().union(*methods.values())
            self.test_references[test["number"]] = set(shared).union(references)

    def _tests_for_family(self, family: str) -> set[int]:
        modules_by_family = defaultdict(set)
        for path in self.import_points:
            current = _family(path, self.dialects)
            if current is not None:
                modules_by_family[current].add(path)

        families_by_import = {
            name: current
            for current, paths in modules_by_family.items()
            for path in paths
            for name in (_module_name(path), f"sqlglot.dialects.{current}")
        }
        affected = {family}
        while True:
            dependents = {
                current
                for current, paths in modules_by_family.items()
                if any(
                    families_by_import.get(name) in affected
                    for path in paths for name in self.imports[path]
                )
            }
            if dependents <= affected:
                break
            affected.update(dependents)

        paths = set().union(*(modules_by_family[current] for current in affected))
        users = set().union(*(
            numbers for path in paths for numbers in self.coverage_lines.get(path, {}).values()
        ))
        if not users:
            return set()

        modules = {_module_name(path) for path in paths}
        for path, points in self.import_points.items():
            lines = self.coverage_lines.get(path, {})
            for line, imports in points:
                if modules.intersection(imports):
                    users.update(lines.get(line, ()))
        for test in self.catalog:
            path = test["file"]
            references = self.test_references.get(test["number"], set())
            if (
                modules.intersection(references)
                or Path(path).stem.removeprefix("test_") in affected
                or _REGISTRY_NAMES.intersection(references)
                or any(name.casefold().partition(",")[0].strip() in affected for name in references)
            ):
                users.add(test["number"])
        return users

    def _static_readers(self, field: str) -> set[int]:
        return {
            test["number"]
            for test in self.catalog
            if field in self.test_references.get(test["number"], ())
        }

    def _tests_for_field(self, field: str) -> set[int]:
        if field not in self.field_tests:
            users = self._static_readers(field)
            for path, positions in self.field_lines.get(field, {}).items():
                lines = self.coverage_lines.get(path, {})
                for line in positions:
                    users.update(lines.get(line, ()))
            self.field_tests[field] = users
        return self.field_tests[field]

    def _factory_is_deferred(self, path: str, name: str, before: int) -> bool:
        module = self._tree(path)
        if module is None:
            return False
        binding = _factory_binding(module, name)
        if binding is None or binding.lineno >= before:
            return False
        if isinstance(binding, ast.ImportFrom):
            aliases = [alias for alias in binding.names if (alias.asname or alias.name) == name]
            if len(aliases) != 1:
                return False
            imported = binding.module or ""
            if binding.level:
                package = _module_name(path).rpartition(".")[0]
                try:
                    imported = resolve_name("." * binding.level + imported, package)
                except ImportError:
                    return False
            if imported != "sqlglot" and not imported.startswith("sqlglot."):
                return False
            relative = imported.replace(".", "/")
            module = self._tree(f"{relative}/__init__.py") or self._tree(f"{relative}.py")
            if module is None:
                return False
            binding = _factory_binding(module, aliases[0].name)
        return isinstance(binding, ast.FunctionDef) and _returns_deferred_function(binding, module)

    def targets(
        self, path: str, original_lines: list[str], edits: list[Edit]
    ) -> set[tuple[str | None, str]]:
        if _family(path, self.dialects) is None:
            return set()
        tree = self._tree(path)
        if tree is None:
            return set()
        return {
            (".".join((_module_name(path), *scope)), field) if scope else (None, field)
            for *scope, field in _data_changes(
                tree, original_lines, edits, lambda name, before: self._factory_is_deferred(path, name, before)
            )
        }

    def select(
        self,
        path: str,
        original_lines: list[str],
        edits: list[Edit],
        *,
        class_uses: dict[str, set[int]] | None = None,
        shared_class_uses: set[str] | frozenset[str] = frozenset(),
    ) -> set[int] | None:
        targets = self.targets(path, original_lines, edits)
        if not targets:
            return None
        self._load()
        family = _family(path, self.dialects)
        if family not in self.family_tests:
            self.family_tests[family] = self._tests_for_family(family)
        users = self.family_tests[family].copy()
        if not users:
            return None
        for target, field in targets:
            observed = class_uses.get(target) if class_uses and target else None
            if observed and target not in shared_class_uses:
                users.update(observed, self._static_readers(field))
            else:
                users.update(self._tests_for_field(field))
        return users


def main() -> None:
    parser = argparse.ArgumentParser(description="Index pinned SQLGlot dialect source and catalog test references")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.test_catalog.read_text(encoding="utf-8"))["tests"]
    index = DialectImpact(args.source, {}, catalog).build_index()
    serialized = json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if args.output.suffix == ".gz":
        serialized = gzip.compress(serialized, mtime=0)
    args.output.write_bytes(serialized)


if __name__ == "__main__":
    main()
