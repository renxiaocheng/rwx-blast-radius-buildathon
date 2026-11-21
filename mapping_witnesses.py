"""Exact mapping identity and key observations on the unmodified checkout."""

import ast
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import FunctionType, MethodType

from attribute_witnesses import _REFLECTIVE_READS, _reflection_aliases, _reflective_accesses
from dialect_impact import _module_name
from mapping_impact import MappingImpact, declarations, key_label


_REG = "__rwx_mapping_register__"
_KEY = "__rwx_mapping_key__"
_USE = "__rwx_mapping_use__"
_BOUND = "__rwx_mapping_bound__"
_SUB = "__rwx_mapping_subscript__"
_MEMBER = "__rwx_mapping_member__"
_FORWARD = "__rwx_mapping_forward__"
_VERSION = 1
_CENSUS_VERSION = 1
_METHODS = {"get", "keys", "pop", "__getitem__", "__contains__"}
Site = tuple[str, int, int, str]


def mapping_source_files(source: Path) -> Iterable[Path]:
    return (source / "sqlglot").rglob("*.py")


def mapping_fields(source: Path) -> frozenset[str]:
    fields = set()
    for file in mapping_source_files(source):
        relative = file.relative_to(source).as_posix()
        try:
            tree = ast.parse(file.read_bytes(), filename=relative)
        except (OSError, SyntaxError, ValueError):
            continue
        fields.update(
            origin.rsplit(".", 1)[-1]
            for origin, statement in declarations(tree, _module_name(relative)).items()
            if statement is not None and isinstance(statement.value, (ast.Dict, ast.Set))
        )
    return frozenset(fields)


def _token(key) -> str | None:
    if type(key) is str:
        return f"str:{key}"
    if type(key) is type:
        module = type.__getattribute__(key, "__module__")
        if type(module) is str and module.startswith("sqlglot."):
            return f"class:{module}.{type.__getattribute__(key, '__qualname__')}"
    return None


def _site(path: str, node: ast.Attribute | ast.Name, field: str) -> Site:
    return path, node.lineno, node.col_offset, field


def field_load(node: ast.AST, fields: frozenset[str]) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in fields:
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in fields:
        return node.attr
    return None


def _direct(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> str:
    parent = parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in _METHODS:
        call = parents.get(parent)
        if isinstance(call, ast.Call) and call.func is parent:
            return "operation"
    if (
        isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load)
        and not any(isinstance(child, ast.Slice) for child in ast.walk(parent.slice))
    ):
        return "operation"
    if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
        return "keys"
    if isinstance(parent, ast.Compare) and len(parent.ops) == 1:
        op = parent.ops[0]
        if isinstance(op, (ast.In, ast.NotIn)) and parent.comparators[0] is node:
            return "operation"
        if isinstance(op, (ast.Is, ast.IsNot)):
            other = parent.left if parent.comparators[0] is node else parent.comparators[0]
            if isinstance(other, ast.Constant) and other.value is None:
                return "identity"
    return "opaque"


def _function(node: ast.AST, parents: Mapping[ast.AST, ast.AST]):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
        if isinstance(node, (ast.ClassDef, ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            return None
    return None


def _uses(function: ast.AST, name: str) -> tuple[ast.Name, ...]:
    return tuple(
        node for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
    )


def _body_local(node: ast.AST, function: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> bool:
    first = node
    while first in parents and parents[first] is not function:
        first = parents[first]
    return first in function.body and _function(node, parents) is function


def _safe_uses(function, name: str, parents: Mapping[ast.AST, ast.AST]) -> bool:
    uses = _uses(function, name)
    return bool(uses) and all(_body_local(node, function, parents) and _direct(node, parents) == "operation" for node in uses)


def _safe_alias(assignment: ast.Assign, parents: Mapping[ast.AST, ast.AST]) -> bool:
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return False
    scope = _function(assignment, parents)
    if scope is None or not _body_local(assignment, scope, parents):
        return False
    name = assignment.targets[0].id
    uses = _uses(scope, name)
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names for node in ast.walk(scope)):
        return False
    # An existing argument may be supplied by a caller. Once assigned, the
    # exact mapping is still used only at instrumented operations or is None.
    return bool(uses) and all(
        _body_local(node, scope, parents) and _direct(node, parents) in {"operation", "identity"} for node in uses
    )


class MappingRecorder:
    def __init__(self):
        self.origins: dict[int, tuple[object, set[str]]] = {}
        self.declared: set[str] = set()
        self.labels = defaultdict(set)
        self.sites: set[Site] = set()
        self.keys_only: set[str] = set()
        self.opaque: set[str] = set()
        self.reads = defaultdict(set)
        self.unresolved: bool = False

    def name_for(self, value) -> set[str] | None:
        entry = self.origins.get(id(value))
        return entry[1] if entry is not None and entry[0] is value else None

    def export(self) -> dict:
        return {
            "version": _VERSION,
            "declared": sorted(self.declared),
            "labels": [[origin, label, sorted(tokens)] for (origin, label), tokens in sorted(self.labels.items())],
            "sites": sorted(self.sites),
            "keys_only": sorted(self.keys_only),
            "opaque": sorted(self.opaque),
            "reads": [
                [origin, token, sorted(node for node in nodes if node is not None), None in nodes]
                for (origin, token), nodes in sorted(self.reads.items())
            ],
            "unresolved": self.unresolved,
        }

    def merge(self, data: Mapping) -> None:
        if data.get("version") != _VERSION:
            self.unresolved = True
            return
        self.declared.update(data["declared"])
        self.sites.update(tuple(site) for site in data["sites"])
        self.keys_only.update(data["keys_only"])
        self.opaque.update(data["opaque"])
        self.unresolved |= data["unresolved"]
        for origin, label, tokens in data["labels"]:
            self.labels[origin, label].update(tokens)
        for origin, token, children, shared in data["reads"]:
            self.reads[origin, token].update(children)
            if shared:
                self.reads[origin, token].add(None)


def mapping_instrumentation(source: Path, recorder: MappingRecorder, current_node: Callable[[], str | None]):
    fields = mapping_fields(source)
    root = str(source.resolve()) + "/"
    trusted: dict[tuple[str, int], tuple[tuple[str, ...], set[str]]] = {}

    def register(origin, value):
        if type(value) not in {dict, set}:
            recorder.opaque.add(origin)
            return value
        identity = id(value)
        if identity not in recorder.origins:
            recorder.origins[identity] = value, set()
        recorder.origins[identity][1].add(origin)
        recorder.declared.add(origin)
        return value

    def reg_key(origin, label, value):
        token = _token(value)
        if token is not None:
            recorder.labels[origin, label].add(token)
        return value

    def use(value, mode):
        owners = recorder.name_for(value)
        if owners:
            if mode == "keys":
                recorder.keys_only.update(owners)
            elif mode == "opaque":
                recorder.opaque.update(owners)
        return value

    def read(value, key, membership=False):
        owners = recorder.name_for(value)
        if not owners:
            return
        if membership and type(value) is dict:
            # Dict keys are unchanged by the supported dict edit.
            return
        token = _token(key)
        if token is None:
            recorder.opaque.update(owners)
        else:
            node = current_node()
            for origin in owners:
                recorder.reads[origin, token].add(node)

    def bound(value, method):
        callback = getattr(value, method)
        if not recorder.name_for(value):
            return callback

        def call(*args, **kwargs):
            if method == "keys":
                use(value, "keys")
            elif args and not kwargs:
                read(value, args[0], membership=method == "__contains__")
            else:
                use(value, "opaque")
            return callback(*args, **kwargs)

        return call

    def subscript(value, key):
        read(value, key)
        return value[key]

    def member(key, value, negated):
        read(value, key, membership=True)
        return key not in value if negated else key in value

    def forward(operation, /, *args, **kwargs):
        bound_method = type(operation) is MethodType
        function = operation.__func__ if bound_method else operation
        signature, accepted = (), set()
        if type(function) is FunctionType and function.__code__.co_filename.startswith(root):
            filename, start = function.__code__.co_filename[len(root):], function.__code__.co_firstlineno
            signature, accepted = trusted.get((filename, start), ((), set()))
        if bound_method:
            signature = signature[1:]
        for position, value in enumerate(args):
            if position >= len(signature) or signature[position] not in accepted:
                use(value, "opaque")
        for name, value in kwargs.items():
            if name not in accepted:
                use(value, "opaque")
        return operation(*args, **kwargs)

    def helper(name: str, args: list[ast.expr], node: ast.AST, keywords=()) -> ast.Call:
        return ast.copy_location(ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=list(keywords)), node)

    class Instrument(ast.NodeTransformer):
        def __init__(self, path: str, tree: ast.Module):
            self.path = path
            nodes = tuple(ast.walk(tree))
            self.parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
            self.declarations = {
                statement: origin for origin, statement in declarations(tree, _module_name(path)).items()
                if statement is not None and isinstance(statement.value, (ast.Dict, ast.Set))
            }
            self.forward_calls = set()
            self.operations = set()
            self.modes = {}
            for node in nodes:
                field = field_load(node, fields)
                if field is not None:
                    recorder.sites.add(_site(path, node, field))
                    self.modes[node] = self.mode(node)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = (*node.args.posonlyargs, *node.args.args)
                    accepted = {
                        arg.arg for arg in (*params, *node.args.kwonlyargs)
                        if _safe_uses(node, arg.arg, self.parents)
                    }
                    if accepted:
                        first = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
                        trusted[path, first] = tuple(arg.arg for arg in params), accepted
                        self.operations.update(use for name in accepted for use in _uses(node, name))

        def mode(self, node):
            mode = _direct(node, self.parents)
            if mode != "opaque":
                if mode == "operation":
                    self.operations.add(node)
                return mode
            parent = self.parents.get(node)
            if isinstance(parent, ast.Assign) and parent.value is node and _safe_alias(parent, self.parents):
                scope = _function(parent, self.parents)
                self.operations.update(
                    use for use in _uses(scope, parent.targets[0].id) if _direct(use, self.parents) == "operation"
                )
                return "transfer"
            call = self.parents.get(parent) if isinstance(parent, ast.keyword) else parent
            if isinstance(call, ast.Call) and (node in call.args or parent in call.keywords):
                self.forward_calls.add(call)
                return "transfer"
            return "opaque"

        def visit_Name(self, node):
            if node in self.modes and self.modes[node] in {"keys", "opaque"}:
                return helper(_USE, [node, ast.Constant(self.modes[node])], node)
            return node

        def visit_Attribute(self, node):
            self.generic_visit(node)
            if node in self.modes and self.modes[node] in {"keys", "opaque"}:
                return helper(_USE, [node, ast.Constant(self.modes[node])], node)
            return node

        def visit_Assign(self, node):
            origin = self.declarations.get(node)
            self.generic_visit(node)
            if origin is not None:
                self.finish_declaration(node, origin)
            return node

        visit_AnnAssign = visit_Assign

        def finish_declaration(self, node, origin):
            dictionary = node.value
            if isinstance(dictionary, ast.Dict):
                for index, key in enumerate(dictionary.keys):
                    label = key_label(key)
                    if label is not None:
                        dictionary.keys[index] = helper(_KEY, [ast.Constant(origin), ast.Constant(label), key], key)
            node.value = helper(_REG, [ast.Constant(origin), dictionary], dictionary)

        def visit_Call(self, node):
            method = node.func.attr if isinstance(node.func, ast.Attribute) else None
            bind_method = method in _METHODS and node.func.value in self.operations
            self.generic_visit(node)
            if node in self.forward_calls:
                return helper(_FORWARD, [node.func, *node.args], node, node.keywords)
            if bind_method and isinstance(node.func, ast.Attribute):
                node.func = helper(_BOUND, [node.func.value, ast.Constant(method)], node.func)
            return node

        def visit_Subscript(self, node):
            observed = node.value in self.operations
            self.generic_visit(node)
            if observed and isinstance(node.ctx, ast.Load) and not any(isinstance(child, ast.Slice) for child in ast.walk(node.slice)):
                return helper(_SUB, [node.value, node.slice], node)
            return node

        def visit_Compare(self, node):
            observed = len(node.comparators) == 1 and node.comparators[0] in self.operations
            self.generic_visit(node)
            if observed and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
                return helper(_MEMBER, [node.left, node.comparators[0], ast.Constant(isinstance(node.ops[0], ast.NotIn))], node)
            return node

    def transform(path, tree):
        return Instrument(path, tree).visit(tree)

    return transform, {
        _REG: register, _KEY: reg_key, _USE: use, _BOUND: bound, _SUB: subscript,
        _MEMBER: member, _FORWARD: forward,
    }


class MappingWitnesses:
    def __init__(self, source: Path, payload: Mapping, numbers_by_id: Mapping[str, int]):
        self.source = source
        self.numbers_by_id = numbers_by_id
        self.files_by_number = {value: node.partition("::")[0] for node, value in numbers_by_id.items()}
        self.impact = MappingImpact(source)
        self.recorded = MappingRecorder()
        self.recorded.merge(payload)
        reliable = payload.get("reliable")
        self.unobserved = None if not isinstance(reliable, list) else set(numbers_by_id.values()) - {
            numbers_by_id[node] for node in reliable if node in numbers_by_id
        }
        self.cached_census = payload.get("census")
        self.site_census = None
        self.test_trees = {}

    @staticmethod
    def _digest(value) -> str:
        return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()

    def _source_digest(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(mapping_source_files(self.source)):
            digest.update(path.relative_to(self.source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def _restore_census(self):
        saved = self.cached_census
        if not isinstance(saved, dict) or saved.get("version") != _CENSUS_VERSION:
            return None
        try:
            contents = saved["fields"]
            if (
                saved["contents"] != self._digest(contents)
                or saved["source"] != self._source_digest()
                or saved["sites"] != self._digest(sorted(self.recorded.sites))
            ):
                return None
            return {field: {tuple(site) for site in sites} for field, sites in contents.items()}
        except (OSError, AttributeError, KeyError, TypeError, ValueError):
            return None

    def build_census(self) -> dict:
        contents = {field: sorted(sites) for field, sites in self._census().items()}
        if self.recorded.unresolved:
            raise ValueError("Cannot index incomplete mapping source")
        return {
            "version": _CENSUS_VERSION,
            "source": self._source_digest(),
            "sites": self._digest(sorted(self.recorded.sites)),
            "fields": contents,
            "contents": self._digest(contents),
        }

    def _census(self):
        if self.site_census is not None:
            return self.site_census
        restored = self._restore_census()
        if restored is not None:
            self.site_census = restored
            return restored
        fields = frozenset(site[3] for site in self.recorded.sites)
        census = defaultdict(set)
        for path in mapping_source_files(self.source):
            relative = path.relative_to(self.source).as_posix()
            tree = self.impact._tree(relative)
            if tree is None:
                self.recorded.unresolved = True
                break
            for node in ast.walk(tree):
                field = field_load(node, fields)
                if field is not None:
                    census[field].add(_site(relative, node, field))
                if isinstance(node, ast.ImportFrom) and any(alias.asname and alias.name in fields and alias.asname != alias.name for alias in node.names):
                    # Aliasing a field into an unrelated lexical name can
                    # hide subsequent non-key consumption from the tracer.
                    for alias in node.names:
                        if alias.asname and alias.name in fields and alias.asname != alias.name:
                            census[alias.name].add((relative, node.lineno, node.col_offset, "?alias"))
        self.site_census = census
        return census

    def _test_reads(self, filename, field, key):
        if filename not in self.test_trees:
            result = None
            if filename.startswith("tests/") and ".." not in PurePosixPath(filename).parts:
                tree = self.impact._tree(filename)
                if tree is not None:
                    nodes = tuple(ast.walk(tree))
                    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
                    reflection = tuple(
                        _reflective_accesses(filename, nodes, parents, _reflection_aliases(tree), test_sources=True)
                    )
                    result = nodes, parents, reflection
            self.test_trees[filename] = result
        result = self.test_trees[filename]
        if result is None:
            return True
        nodes, parents, reflection = result
        if any(access.field_pattern is None or access.field_pattern == field for access in reflection):
            return True
        for node in nodes:
            if isinstance(node, ast.Constant) and type(node.value) is str and node.value == field:
                return True
            if field_load(node, frozenset({field})) is None:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                label = key_label(parent.slice)
                if label is not None and label != key and label.startswith("Constant(") and key.startswith("Constant("):
                    continue
            return True
        return False

    def select_patch(self, path, lines, edits, *, attributes, mentions):
        """Return mapped readers; the caller retains ``unobserved`` and source readers."""
        targets = self.impact.changes(path, lines, edits)
        if not targets or self.recorded.unresolved or self.unobserved is None:
            return None
        if attributes is None:
            return None
        result = set()
        for target in targets:
            origin, field, key, kind = target.origin, target.field, target.key, target.kind
            if (
                origin not in self.recorded.declared or origin in self.recorded.opaque
                or (kind == "set" and origin in self.recorded.keys_only)
            ):
                return None
            receiver = origin.rpartition(".")[0]
            if receiver == _module_name(path):
                # A module-global field can be retrieved reflectively without
                # exposing a tracked class receiver. That evidence is not yet
                # available to this proof path.
                return None
            labels = self.recorded.labels.get((origin, key), set())
            if not labels and key.startswith("Constant("):
                try:
                    labels = {_token(ast.literal_eval(ast.parse(key, mode="eval").body.keywords[0].value))}
                except (AttributeError, IndexError, SyntaxError, TypeError, ValueError):
                    return None
            if len(labels) != 1 or None in labels:
                return None
            token = next(iter(labels))
            observed = self.recorded.reads.get((origin, token), set())
            if not observed and kind == "dict" and token.startswith("str:"):
                # Key-only tries can resolve a longer spelling through an
                # existing shorter key. Preserve those actual readers too
                # when the changed value itself has no witness.
                words = token.removeprefix("str:").split(" ")
                if len(words) > 1 and all(words):
                    prefixes = ("str:" + " ".join(words[:end]) for end in range(1, len(words)))
                    observed = set().union(*(self.recorded.reads.get((origin, prefix), set()) for prefix in prefixes))
            if not observed or None in observed:
                return None
            witnessed = {self.numbers_by_id[node] for node in observed if node in self.numbers_by_id}
            if not witnessed:
                return None
            required = self._census().get(field, set())
            if self.recorded.unresolved or not required or not required <= self.recorded.sites:
                return None
            attributes._load()
            if (
                attributes.recorder.unresolved
                or not any(receiver in names for _, names, _ in attributes.by_field.get(field, ()))
                or attributes._has_dynamic_read(receiver, field)
            ):
                return None
            result.update(witnessed)
            by_file = defaultdict(set)
            for number in mentions({field, "__dict__"} | _REFLECTIVE_READS):
                by_file[self.files_by_number.get(number, "")].add(number)
            for filename, numbers in by_file.items():
                if self._test_reads(filename, field, key):
                    result.update(numbers)
        return result
