"""Baseline witnesses for reads of SQLGlot dialect and expression fields."""

import ast
import builtins
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import sys
import types

from source_impact import expression_arg_types_impact


_METADATA = frozenset({"arg_types", "required_args"})
_FAMILY_BASES = frozenset({
    "sqlglot.dialects.dialect.Dialect",
    "sqlglot.expressions.core.Expr",
    "sqlglot.generator.Generator",
    "sqlglot.parser.Parser",
    "sqlglot.tokens.Tokenizer",
})
_BUILTIN_ACCESS = {
    name: getattr(builtins, name) for name in ("getattr", "hasattr", "setattr", "delattr", "vars")
}
_BUILTIN_ACCESS["getattr_static"] = inspect.getattr_static
_METHOD_ACCESS = {"__getattribute__", "__getattr__", "__setattr__", "__delattr__"}
_REFLECTIVE_READS = set(_BUILTIN_ACCESS) | _METHOD_ACCESS | {"attrgetter"}
_DYNAMIC_HELPER = "__rwx_attribute_dynamic_read__"
_FAMILY_PACKAGES = {"dialects", "generators", "parsers", "tokenizers", "typing"}
_VERSION = 1
_CENSUS_VERSION = 1

Site = tuple[str, int, str]
ClassField = tuple[str | None, str]


def _field(name: str) -> bool:
    return name in _METADATA or (name.isascii() and name.isupper() and name.isidentifier())


def _source_files(source: Path) -> Iterable[Path]:
    return (source / "sqlglot").rglob("*.py")


def candidate_attribute_names(source: Path) -> frozenset[str]:
    """Profile explicit reads of baseline fields that can be patch targets."""
    fields = set(_METADATA)
    used = set()
    dialects = {path.stem for path in (source / "sqlglot" / "dialects").glob("*.py")}
    dialects.difference_update(("__init__", "dialect"))
    for path in _source_files(source):
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                used.add(node.attr)
            elif isinstance(node, ast.ClassDef) and path.parent.name in _FAMILY_PACKAGES and path.stem in dialects:
                for member in node.body:
                    if isinstance(member, ast.AnnAssign):
                        targets = [member.target]
                    elif isinstance(member, ast.Assign):
                        targets = member.targets
                    else:
                        continue
                    fields.update(target.id for target in targets if isinstance(target, ast.Name) and _field(target.id))
    return frozenset(fields & used | {"__dict__"})


def _receiver_names(cls: type) -> tuple[str, ...]:
    names = []
    for base in type.__getattribute__(cls, "__mro__"):
        module = type.__getattribute__(base, "__module__")
        if type(module) is str and module.startswith("sqlglot."):
            qualified = type.__getattribute__(base, "__qualname__")
            names.append(f"{module}.{qualified}")
    return tuple(names) if _FAMILY_BASES.intersection(names) else ()


class AttributeRecorder:
    """Accumulate and merge per-test observations from the parent and workers."""

    def __init__(self):
        self.sites: set[Site] = set()
        self.dynamic_sites: set[Site] = set()
        self.unresolved: set[Site] = set()
        self.reads = defaultdict(set)
        self.dynamic_reads = defaultdict(set)
        self._classes = {}

    def register_site(self, path: str, line: int, field: str) -> None:
        self.sites.add((path, line, field))

    def register_dynamic_site(self, path: str, line: int, field: str | None = None) -> None:
        self.dynamic_sites.add((path, line, field or "*"))

    def mark_unresolved(self, path: str, line: int = 0, field: str | None = None) -> None:
        self.unresolved.add((path, line, field or "*"))

    def _record(self, receiver, site: Site, node_id: str | None, reads) -> None:
        cls = type(receiver)
        if issubclass(cls, type):
            cls = receiver
        # A SQLGlot metaclass can override equality. Cache by object identity
        # and retain the object to prevent reusing an ID after it is collected.
        identity = id(cls)
        entry = self._classes.get(identity)
        if entry is None:
            entry = self._classes[identity] = (cls, _receiver_names(cls))
        names = entry[1]
        if names:
            reads[site, names].add(node_id)

    def record(self, receiver, field: str, path: str, line: int, node_id: str | None) -> None:
        self._record(receiver, (path, line, field), node_id, self.reads)

    def record_dynamic(self, receiver, field: str | None, path: str, line: int, node_id: str | None) -> None:
        self._record(receiver, (path, line, field or "*"), node_id, self.dynamic_reads)

    def export(self) -> dict:
        def export_rows(reads):
            return [
                [*site, names, sorted(node for node in nodes if node is not None), None in nodes]
                for (site, names), nodes in sorted(reads.items())
            ]

        return {
            "version": _VERSION,
            "sites": sorted(self.sites),
            "dynamic_sites": sorted(self.dynamic_sites),
            "unresolved": sorted(self.unresolved),
            "reads": export_rows(self.reads),
            "dynamic_reads": export_rows(self.dynamic_reads),
        }

    def merge(self, data: Mapping) -> None:
        if data.get("version") != _VERSION:
            self.mark_unresolved("*")
            return
        self.sites.update(tuple(site) for site in data["sites"])
        self.dynamic_sites.update(tuple(site) for site in data["dynamic_sites"])
        self.unresolved.update(tuple(site) for site in data["unresolved"])
        for key, reads in (("reads", self.reads), ("dynamic_reads", self.dynamic_reads)):
            for path, line, field, names, tests, shared in data[key]:
                observed = reads[(path, line, field), tuple(names)]
                observed.update(tests)
                if shared:
                    observed.add(None)


def mark_preloaded_modules(source: Path, recorder: AttributeRecorder) -> None:
    """Flag package code imported before the witness loader could instrument it."""
    root = source.resolve()
    for name, module in tuple(sys.modules.items()):
        if module is None or (name != "sqlglot" and not name.startswith("sqlglot.")):
            continue
        filename = vars(module).get("__file__")
        try:
            relative = Path(filename).resolve().relative_to(root).as_posix()
        except (TypeError, ValueError, OSError):
            relative = "*"
        recorder.mark_unresolved(relative)


@dataclass(frozen=True)
class Access:
    site: Site
    field_pattern: str | None = None


def _literal_name(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _reflection_aliases(tree: ast.Module) -> dict[str, str]:
    return {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in {"builtins", "inspect", "operator"}
        for alias in node.names
        if alias.asname and alias.name in _REFLECTIVE_READS
    }


def _call_access(path: str, node: ast.Call, aliases: Mapping[str, str]) -> tuple[str, Access] | None:
    symbol = _name(node.func)
    symbol = aliases.get(symbol, symbol)
    if symbol not in _REFLECTIVE_READS:
        return None
    if symbol in _BUILTIN_ACCESS and symbol != "vars" and len(node.args) > 1:
        attribute = _literal_name(node.args[1])
    elif (symbol in _METHOD_ACCESS or symbol == "attrgetter") and len(node.args) == 1:
        attribute = _literal_name(node.args[0])
    else:
        attribute = None
    return symbol, Access((path, node.lineno, attribute or "*"), attribute)


def dynamic_instrumentation(recorder: AttributeRecorder, current_node: Callable[[], str | None]):
    """Build hooks for comparison_witnesses.trace_witnesses's extra transformer."""
    def invoke(operation, path, line, symbol, marker, /, *args, **kwargs):
        if symbol in _BUILTIN_ACCESS and operation is _BUILTIN_ACCESS[symbol] and args:
            receiver = args[0]
            attribute = args[1] if len(args) > 1 and type(args[1]) is str else None
        elif symbol in _METHOD_ACCESS and type(operation) in {
            types.MethodType, types.BuiltinMethodType, types.MethodWrapperType,
        }:
            receiver = operation.__self__
            attribute = args[0] if args and type(args[0]) is str else None
        elif symbol in _METHOD_ACCESS and isinstance(operation, types.WrapperDescriptorType) and args:
            receiver = args[0]
            attribute = args[1] if len(args) > 1 and type(args[1]) is str else None
        else:
            recorder.mark_unresolved(path, line, marker)
            return operation(*args, **kwargs)
        recorder.record_dynamic(receiver, attribute, path, line, current_node())
        return operation(*args, **kwargs)

    class DynamicAccess(ast.NodeTransformer):
        def __init__(self, path: str, aliases: Mapping[str, str]):
            self.path = path
            self.aliases = aliases

        def visit_Call(self, node: ast.Call):
            witness = _call_access(self.path, node, self.aliases)
            self.generic_visit(node)
            if witness is None:
                return node
            symbol, access = witness
            path, line, marker = access.site
            # The zero-argument vars form observes its caller's frame.
            # attrgetter produces a callable whose identity must be preserved.
            unsafe = symbol == "attrgetter" or (
                symbol == "vars"
                and (len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Starred))
            )
            if unsafe:
                recorder.mark_unresolved(path, line, marker)
                return node
            recorder.register_dynamic_site(path, line, marker)
            return ast.copy_location(ast.Call(
                func=ast.Name(id=_DYNAMIC_HELPER, ctx=ast.Load()),
                args=[node.func, *(ast.Constant(value=x) for x in (path, line, symbol, marker)), *node.args],
                keywords=node.keywords,
            ), node)

    def transform(path: str, tree: ast.Module) -> ast.Module:
        return DynamicAccess(path, _reflection_aliases(tree)).visit(tree)

    return transform, {_DYNAMIC_HELPER: invoke}


def _dictionary_field(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> str | None:
    parent = parents.get(node)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return _literal_name(parent.slice)
    if isinstance(parent, ast.Compare) and len(parent.comparators) == 1 and parent.comparators[0] is node:
        if isinstance(parent.ops[0], (ast.In, ast.NotIn)):
            return _literal_name(parent.left)
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "get":
        caller = parents.get(parent)
        if isinstance(caller, ast.Call) and caller.func is parent and caller.args:
            return _literal_name(caller.args[0])
    return None


def _function_scope(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> ast.AST | None:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return node
        if isinstance(node, (ast.ClassDef, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return None
    return None


def _local_binding(node: ast.Name, parents: Mapping[ast.AST, ast.AST]) -> bool:
    scope = _function_scope(node, parents)
    if scope is None:
        return False
    same_scope = [item for item in ast.walk(scope) if _function_scope(item, parents) is scope]
    if any(isinstance(item, (ast.Global, ast.Nonlocal)) and node.id in item.names for item in same_scope):
        return False
    return any(
        isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == node.id
        for item in same_scope
    )


def _body_read(node: ast.Name, scope: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents and parents[current] is not scope:
        current = parents[current]
    body = scope.body if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else ()
    return current in body


def _observed_value(path: str, node: ast.AST, field: str, sites: set[Site]) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == field and (path, node.lineno, field) in sites
    if isinstance(node, ast.IfExp):
        return _observed_value(path, node.body, field, sites) and _observed_value(path, node.orelse, field, sites)
    if isinstance(node, ast.Subscript):
        return _observed_value(path, node.value, field, sites)
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "set", "tuple", "frozenset"}
        and len(node.args) == 1 and not node.keywords
    ):
        return _observed_value(path, node.args[0], field, sites)
    return False


def _observed_local_alias(path: str, node: ast.Name, parents: dict[ast.AST, ast.AST], sites: set[Site]) -> bool:
    scope = _function_scope(node, parents)
    if scope is None or not _body_read(node, scope, parents) or not _local_binding(node, parents):
        return False
    stores = []
    for child in ast.walk(scope):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id == node.id:
            stores.append(child)
        elif isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if child.name == node.id:
                return False
        elif isinstance(child, ast.MatchMapping) and child.rest == node.id:
            return False
        elif isinstance(child, ast.alias) and (child.asname or child.name.partition(".")[0]) == node.id:
            return False
    if len(stores) != 1 or stores[0].lineno >= node.lineno or isinstance(stores[0].ctx, ast.Del):
        return False
    assignment = parents[stores[0]]
    if assignment not in scope.body or any(
        isinstance(child, ast.arg) and child.arg == node.id for child in ast.walk(scope.args)
    ):
        return False
    if isinstance(assignment, ast.Assign) and assignment.targets == stores:
        rhs = assignment.value
    elif isinstance(assignment, ast.AnnAssign) and assignment.target is stores[0] and assignment.value is not None:
        rhs = assignment.value
    else:
        return False
    # Computing the alias is already attributed to the receiver whose field
    # was read. Never exempt an eager name from an unrelated lexical scope.
    return _observed_value(path, rhs, node.id, sites)


def _loop_attribute_names(argument: ast.AST, parents) -> set[str]:
    if not isinstance(argument, ast.Name):
        return set()
    scope = _function_scope(argument, parents)
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    inside = argument
    while (parent := parents.get(inside)) is not None and parent is not scope:
        if isinstance(parent, ast.For) and inside in parent.body and isinstance(parent.target, ast.Tuple):
            target = next((item for item in parent.target.elts if isinstance(item, ast.Name) and item.id == argument.id), None)
            if target is not None:
                break
        inside = parent
    else:
        return set()
    if any(
        (isinstance(node, ast.Name) and node.id == argument.id and node is not target and not isinstance(node.ctx, ast.Load))
        or (isinstance(node, (ast.Global, ast.Nonlocal)) and argument.id in node.names)
        or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)) and node.name == argument.id)
        or (isinstance(node, ast.alias) and (node.asname or node.name.partition(".")[0]) == argument.id)
        or (isinstance(node, ast.arg) and node.arg == argument.id)
        or (isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == argument.id)
        or (isinstance(node, ast.MatchMapping) and node.rest == argument.id)
        for node in ast.walk(scope)
    ):
        return set()
    if not isinstance(parent.iter, (ast.Tuple, ast.List)) or any(isinstance(node, ast.Starred) for node in parent.target.elts):
        return set()
    offset = parent.target.elts.index(target)
    names = set()
    for row in parent.iter.elts:
        if not isinstance(row, (ast.Tuple, ast.List)) or len(row.elts) != len(parent.target.elts):
            return set()
        if any(isinstance(item, ast.Starred) for item in row.elts):
            return set()
        attribute = _literal_name(row.elts[offset])
        if attribute is None:
            return set()
        names.add(attribute)
    return names


def _reflective_accesses(path: str, nodes, parents, aliases, *, test_sources: bool = False) -> Iterable[Access]:
    for node in nodes:
        if isinstance(node, ast.Attribute) and node.attr == "__dict__" and isinstance(node.ctx, ast.Load):
            yield Access((path, node.lineno, "__dict__"), _dictionary_field(node, parents))
        elif isinstance(node, ast.Call):
            witness = _call_access(path, node, aliases)
            if witness is not None:
                symbol, access = witness
                if (
                    test_sources and access.field_pattern is None and isinstance(node.func, ast.Name)
                    and node.func.id == symbol == "getattr" and len(node.args) >= 2
                    and not any(
                        isinstance(item, ast.Name) and item.id == symbol and not isinstance(item.ctx, ast.Load)
                        or isinstance(item, ast.arg) and item.arg == symbol
                        or isinstance(item, ast.alias) and (item.asname or item.name.partition(".")[0]) == symbol
                        or isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == symbol
                        for item in nodes
                    )
                ):
                    names = _loop_attribute_names(node.args[1], parents)
                    if names:
                        yield from (Access((path, node.lineno, name), name) for name in sorted(names))
                        continue
                yield access
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            name = _name(node)
            if aliases.get(name, name) in _REFLECTIVE_READS:
                if isinstance(node, ast.Name) and _local_binding(node, parents):
                    continue
                caller = parents.get(node)
                if not isinstance(caller, ast.Call) or caller.func is not node:
                    yield Access((path, node.lineno, "?escape"))


def _derived_required_args(access, parents) -> bool:
    """Recognize the eager, pure derivation already guarded by source_impact."""
    reference = ast.parse("cls.required_args = {k for k, v in cls.arg_types.items() if v}").body[0]
    current = access
    while current in parents and not isinstance(current, ast.stmt):
        current = parents[current]
    return isinstance(current, ast.Assign) and ast.dump(current) == ast.dump(reference)


class AttributeWitnesses:
    """Select tests only when a safe edit has complete, unshared read evidence."""

    def __init__(self, source: Path, payload: Mapping, numbers_by_id: Mapping[str, int]):
        self.source = source
        self.numbers_by_id = numbers_by_id
        self.files_by_number = {number: node.partition("::")[0] for node, number in numbers_by_id.items()}
        self.recorder = AttributeRecorder()
        self.recorder.merge(payload)
        self._census = payload.get("census")
        self._loaded = False
        self._source_complete = True
        self.explicit = defaultdict(set)
        self.bare_fields: set[str] = set()
        self.derived: set[Site] = set()
        self.reflective: list[Access] = []
        self.test_accesses: dict[str, tuple[set[str], tuple[Access, ...]] | None] = {}
        self.by_field = defaultdict(list)
        self.namespace_receivers = defaultdict(set)
        self.dynamic_receivers = defaultdict(set)
        for (site, names), tests in self.recorder.reads.items():
            self.by_field[site[2]].append((site, names, tests))
            if site[2] == "__dict__":
                self.namespace_receivers[site[:2]].update(names)
        for site, names in self.recorder.dynamic_reads:
            self.dynamic_receivers[site].update(names)

    def _source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(_source_files(self.source)):
            digest.update(path.relative_to(self.source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def _sites_fingerprint(self) -> str:
        encoded = json.dumps(sorted(self.recorder.sites), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _census_fingerprint(census: dict) -> str:
        body = {key: value for key, value in census.items() if key != "checksum"}
        encoded = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _restore_census(self) -> bool:
        census = self._census
        if not isinstance(census, dict) or census.get("version") != _CENSUS_VERSION:
            return False
        try:
            if (
                census["checksum"] != self._census_fingerprint(census)
                or census["source"] != self._source_fingerprint()
                or census["sites"] != self._sites_fingerprint()
            ):
                return False
            explicit = {field: {tuple(site) for site in sites} for field, sites in census["explicit"].items()}
            bare_fields = set(census["bare_fields"])
            derived = {tuple(site) for site in census["derived"]}
            reflective = [Access(tuple(site), pattern) for site, pattern in census["reflective"]]
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return False
        self.explicit.update(explicit)
        self.bare_fields = bare_fields
        self.derived = derived
        self.reflective = reflective
        return True

    def build_census(self) -> dict:
        """Package pinned static facts without including any test observations."""
        self._load()
        if not self._source_complete:
            raise ValueError("Could not parse every pinned SQLGlot source file")
        census = {
            "version": _CENSUS_VERSION,
            "source": self._source_fingerprint(),
            "sites": self._sites_fingerprint(),
            "explicit": {field: sorted(sites) for field, sites in self.explicit.items()},
            "bare_fields": sorted(self.bare_fields),
            "derived": sorted(self.derived),
            "reflective": [[access.site, access.field_pattern] for access in self.reflective],
        }
        census["checksum"] = self._census_fingerprint(census)
        return census

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self._restore_census():
            return
        for path in _source_files(self.source):
            relative = path.relative_to(self.source).as_posix()
            try:
                tree = ast.parse(path.read_bytes(), filename=relative)
            except (OSError, SyntaxError, ValueError):
                self.recorder.mark_unresolved(relative)
                self._source_complete = False
                continue
            nodes = tuple(ast.walk(tree))
            parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
            aliases = _reflection_aliases(tree)
            self.reflective.extend(_reflective_accesses(relative, nodes, parents, aliases))
            ambiguous_sites = set()
            for node in nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and _field(node.id):
                    if not _observed_local_alias(relative, node, parents, self.recorder.sites):
                        self.bare_fields.add(node.id)
                if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load) or not _field(node.attr):
                    continue
                site = (relative, node.lineno, node.attr)
                if site in self.explicit[node.attr]:
                    ambiguous_sites.add(site)
                self.explicit[node.attr].add(site)
                if relative == "sqlglot/expressions/core.py" and node.attr == "arg_types" and _derived_required_args(node, parents):
                    self.derived.add(site)
            # A physical line can contain several reads with different uses.
            self.derived.difference_update(ambiguous_sites)

    def _has_dynamic_read(self, receiver: str, field: str) -> bool:
        for access in self.reflective:
            if access.field_pattern not in {None, field}:
                continue
            path, line, marker = access.site
            if marker == "__dict__":
                if access.site not in self.recorder.sites or receiver in self.namespace_receivers[path, line]:
                    return True
            else:
                if marker == "?escape":
                    return True
                if access.site not in self.recorder.dynamic_sites and (path, line, "*") not in self.recorder.dynamic_sites:
                    return True
                if receiver in self.dynamic_receivers[path, line, field] or receiver in self.dynamic_receivers[path, line, "*"]:
                    return True
        return False

    def _test_access(self, filename: str) -> tuple[set[str], tuple[Access, ...]] | None:
        if filename in self.test_accesses:
            return self.test_accesses[filename]
        self.test_accesses[filename] = None
        if not filename.startswith("tests/") or ".." in PurePosixPath(filename).parts:
            return None
        try:
            tree = ast.parse((self.source / filename).read_bytes(), filename=filename)
        except (OSError, SyntaxError, ValueError):
            return None
        nodes = tuple(ast.walk(tree))
        parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        names = set()
        for node in nodes:
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.update(re.findall(r"\b[A-Za-z_]\w*\b", node.value))
            elif isinstance(node, ast.alias):
                names.update(node.name.split("."))
        reflection = tuple(_reflective_accesses(filename, nodes, parents, _reflection_aliases(tree), test_sources=True))
        result = (names, reflection)
        self.test_accesses[filename] = result
        return result

    def _extra_tests(self, fields: set[str], mentions: Callable[[set[str]], set[int]]) -> set[int]:
        protected = set()
        possible = mentions(fields | _REFLECTIVE_READS | {"__dict__"})
        for number in possible:
            access = self._test_access(self.files_by_number.get(number, ""))
            if access is None:
                protected.add(number)
                continue
            names, reflection = access
            if names & fields or any(site.field_pattern is None or site.field_pattern in fields for site in reflection):
                protected.add(number)
        return protected

    def _tests(self, targets: set[ClassField], mentions: Callable[[set[str]], set[int]]) -> set[int] | None:
        self._load()
        selected = set()
        for receiver, field in targets:
            if receiver is None or field in self.bare_fields or any(
                marker in {field, "*"} for _, _, marker in self.recorder.unresolved
            ):
                return None
            sites = self.explicit.get(field)
            if not sites or not sites <= self.recorder.sites or self._has_dynamic_read(receiver, field):
                return None
            witnessed = set()
            for site, names, tests in self.by_field[field]:
                if site not in sites or receiver not in names:
                    continue
                if None in tests and site not in self.derived:
                    return None
                witnessed.update(self.numbers_by_id[node] for node in tests if node in self.numbers_by_id)
            if not witnessed:
                return None
            selected.update(witnessed)
        # Pytest rewrites its own modules without our probes. Keep every test
        # in files that read a field or contain relevant or unknown reflection.
        selected.update(self._extra_tests({field for _, field in targets}, mentions))
        return selected

    def select_patch(
        self,
        path: str,
        lines: Sequence[str],
        edits: Iterable,
        *,
        dialect_targets: Callable[[str, list[str], list], set[ClassField]],
        mentions: Callable[[set[str]], set[int]],
    ) -> set[int] | None:
        edits = list(edits)
        expression_classes = expression_arg_types_impact(path, lines, edits)
        if expression_classes:
            targets = {(receiver, field) for receiver in expression_classes for field in _METADATA}
        else:
            targets = dialect_targets(path, list(lines), edits)
        return self._tests(targets, mentions) if targets else None
