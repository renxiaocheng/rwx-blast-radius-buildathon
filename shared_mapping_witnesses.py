"""Baseline namespace observations for module-global mapping edits."""

import ast
import builtins
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import inspect
from importlib.util import resolve_name
from pathlib import Path
import types

from attribute_witnesses import _METHOD_ACCESS, _dictionary_field, _literal_name, _local_binding, _name
from dialect_impact import _factory_binding, _module_name as source_module


_VERSION = 1
_CALL = "__rwx_module_mapping_call__"
_ATTR = "__rwx_module_mapping_attribute__"
_CODE_FUNCTION = "__rwx_module_mapping_code_function__"
_BUILTINS = {name: getattr(builtins, name) for name in ("getattr", "hasattr", "setattr", "delattr", "vars")}
_INSPECT = {
    name: getattr(inspect, name)
    for name in ("getattr_static", "getmembers", "getmembers_static", "getclosurevars", "getmodule")
}
_DIRECT = {"__dict__", "__globals__", "f_globals", "f_locals"}
_LOCAL = {"globals", "locals"}
_CODE = {"eval": builtins.eval, "exec": builtins.exec}
_FUNCTIONS = _BUILTINS | _INSPECT | _CODE
_CALL_NAMES = set(_FUNCTIONS) | _METHOD_ACCESS | _LOCAL | {"attrgetter"}
Site = tuple[str, int, int, str]


def namespace_source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for package in ("sqlglot", "tests"):
        for path in sorted((source / package).rglob("*.py")):
            digest.update(path.relative_to(source).as_posix().encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class Access:
    site: Site
    field: str | None
    mode: str


def _aliases(tree):
    return {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in {"builtins", "inspect", "operator"}
        for alias in node.names
        if alias.asname and alias.name in _CALL_NAMES
    }


def _access(path, node, parents, aliases) -> Access | None:
    if isinstance(node, ast.Call):
        symbol = _name(node.func)
        symbol = aliases.get(symbol, symbol)
        if symbol not in _CALL_NAMES:
            return None
        if symbol in _CODE and not isinstance(node.func, ast.Name):
            return None
        site = path, node.lineno, node.col_offset, symbol
        if symbol in _LOCAL or (symbol == "vars" and not node.args and not node.keywords):
            return Access(site, None, "local")
        if symbol == "attrgetter":
            return Access(site, None, "escape")
        if symbol in _CODE:
            return Access(site, None, "code")
        if symbol in _METHOD_ACCESS and len(node.args) == 1:
            field = _literal_name(node.args[0])
        elif symbol in {"getattr", "getattr_static", "hasattr", "setattr", "delattr"} and len(node.args) >= 2:
            field = _literal_name(node.args[1])
        else:
            field = None
        return Access(site, None if field in _DIRECT else field, "call")
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in _DIRECT:
        return Access((path, node.lineno, node.col_offset, node.attr), _dictionary_field(node, parents), "attribute")
    if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
        symbol = _name(node)
        symbol = aliases.get(symbol, symbol)
        parent = parents.get(node)
        if (
            symbol in _CALL_NAMES
            and (symbol not in _CODE or isinstance(node, ast.Name))
            and (not isinstance(parent, ast.Call) or parent.func is not node)
            and (not isinstance(node, ast.Name) or not _local_binding(node, parents))
        ):
            return Access((path, node.lineno, node.col_offset, symbol), None, "escape")
    return None


def namespace_accesses(path: str, tree: ast.Module):
    nodes = tuple(ast.walk(tree))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    aliases = _aliases(tree)
    return {node: access for node in nodes if (access := _access(path, node, parents, aliases)) is not None}


def _module_name(receiver) -> str | None:
    if type(receiver) is types.ModuleType:
        module = types.ModuleType.__getattribute__(receiver, "__name__")
    elif type(receiver) is types.FunctionType:
        namespace = receiver.__globals__
        module = namespace.get("__name__")
    elif type(receiver) is types.MethodType:
        namespace = receiver.__func__.__globals__
        module = namespace.get("__name__")
    elif type(receiver) is types.FrameType:
        module = receiver.f_globals.get("__name__")
    else:
        return None
    return module if type(module) is str and (module == "sqlglot" or module.startswith("sqlglot.")) else None


class ModuleMappingRecorder:
    def __init__(self):
        self.sites: set[Site] = set()
        self.unknown: set[Site] = set()
        self.reads = defaultdict(set)
        self.dynamic_code: set[str | None] = set()
        self.unresolved = False

    def export(self) -> dict:
        return {
            "version": _VERSION,
            "sites": sorted(self.sites),
            "unknown": sorted(self.unknown),
            "reads": [
                [list(site), module, field, sorted(node for node in nodes if node is not None), None in nodes]
                for (site, module, field), nodes in sorted(self.reads.items())
            ],
            "dynamic_code": sorted(node for node in self.dynamic_code if node is not None),
            "shared_code": None in self.dynamic_code,
            "unresolved": self.unresolved,
        }

    def merge(self, data: Mapping) -> None:
        if data.get("version") != _VERSION:
            self.unresolved = True
            return
        self.sites.update(tuple(site) for site in data["sites"])
        self.unknown.update(tuple(site) for site in data["unknown"])
        self.dynamic_code.update(data["dynamic_code"])
        if data["shared_code"]:
            self.dynamic_code.add(None)
        for site, module, field, nodes, shared in data["reads"]:
            self.reads[tuple(site), module, field].update(nodes)
            if shared:
                self.reads[tuple(site), module, field].add(None)
        self.unresolved |= data["unresolved"]


def module_mapping_instrumentation(source: Path, recorder: ModuleMappingRecorder, current_node: Callable[[], str | None]):
    """Instrument module namespace reads without replacing the namespace."""
    def record(receiver, field, site, *, namespace=False):
        cls = type(receiver)
        if issubclass(cls, types.ModuleType) and cls is not types.ModuleType:
            recorder.unknown.add(site)
            return
        if cls is not types.ModuleType and not namespace:
            return
        module = _module_name(receiver)
        if module is not None:
            recorder.reads[site, module, field or "*"].add(current_node())

    def invoke(operation, path, line, column, symbol, /, *args, **kwargs):
        site = path, line, column, symbol
        if symbol in _FUNCTIONS and operation is _FUNCTIONS[symbol] and args:
            receiver = args[0]
            field = args[1] if symbol in {"getattr", "getattr_static", "hasattr", "setattr", "delattr"} and len(args) >= 2 else None
        elif symbol in _METHOD_ACCESS and type(operation) in {
            types.MethodType, types.BuiltinMethodType, types.MethodWrapperType,
        }:
            receiver = operation.__self__
            field = args[0] if args else None
        elif symbol in _METHOD_ACCESS and isinstance(operation, types.WrapperDescriptorType) and args:
            receiver = args[0]
            field = args[1] if len(args) >= 2 else None
        else:
            recorder.unknown.add(site)
            return operation(*args, **kwargs)
        field = field if type(field) is str else None
        namespace = field in _DIRECT or symbol in {"getmodule", "getclosurevars"}
        record(receiver, None if namespace else field, site, namespace=namespace)
        return operation(*args, **kwargs)

    def attribute(receiver, path, line, column, symbol, /):
        record(receiver, None, (path, line, column, symbol), namespace=symbol != "__dict__")
        return getattr(receiver, symbol)

    def code_function(operation, path, line, column, symbol, /):
        if operation is _CODE[symbol]:
            recorder.dynamic_code.add(current_node())
        else:
            recorder.unknown.add((path, line, column, symbol))
        # Let eval/exec run in its original caller's frame.
        return operation

    def helper(name, node, site, expressions, keywords=()):
        args = [*expressions, *(ast.Constant(value=value) for value in site)]
        if name == _CALL:
            args.extend(node.args)
        return ast.copy_location(ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=list(keywords)), node)

    def transform(path, tree):
        accesses = namespace_accesses(path, tree)

        class Instrument(ast.NodeTransformer):
            def visit_Call(self, node):
                access = accesses.get(node)
                self.generic_visit(node)
                if access is None or access.mode not in {"call", "code"}:
                    return node
                recorder.sites.add(access.site)
                if access.mode == "code":
                    node.func = helper(_CODE_FUNCTION, node.func, access.site, [node.func])
                    return node
                return helper(_CALL, node, access.site, [node.func], node.keywords)

            def visit_Attribute(self, node):
                access = accesses.get(node)
                self.generic_visit(node)
                if access is None or access.mode != "attribute":
                    return node
                recorder.sites.add(access.site)
                return helper(_ATTR, node, access.site, [node.value])

        return Instrument().visit(tree)

    return transform, {_CALL: invoke, _ATTR: attribute, _CODE_FUNCTION: code_function}


def _plain_helper(tree, statement, relevant):
    if (
        not isinstance(statement, ast.FunctionDef) or statement.decorator_list or getattr(statement, "type_params", ())
        or _factory_binding(tree, statement.name) is not statement
    ):
        return False
    body = {node for item in statement.body for node in ast.walk(item)}
    return bool(body & relevant) and not (set(ast.walk(statement)) - body) & relevant


def _stdlib_test_class(tree, owner):
    if owner not in tree.body or owner.decorator_list or owner.keywords or len(owner.bases) != 1:
        return False
    base = owner.bases[0]
    if not isinstance(base, ast.Attribute) or base.attr != "TestCase" or not isinstance(base.value, ast.Name):
        return False
    name = base.value.id
    bindings = [
        (node, alias) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names if (alias.asname or alias.name.partition(".")[0]) == name or alias.name == "*"
    ]
    if len(bindings) != 1:
        return False
    statement, alias = bindings[0]
    if (
        not isinstance(statement, ast.Import) or statement not in tree.body or statement.lineno >= owner.lineno
        or alias.name != "unittest"
    ):
        return False
    return not any(
        isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load)
        or isinstance(node, ast.arg) and node.arg == name
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler,
                             ast.MatchAs, ast.MatchStar)) and node.name == name
        or isinstance(node, ast.MatchMapping) and node.rest == name
        or isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
        for node in ast.walk(tree)
    )


def _ordinary_test(tree, method):
    if not isinstance(method, ast.FunctionDef) or not method.name.startswith("test_"):
        return False
    for decorator in method.decorator_list:
        if (
            not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name)
            or len(decorator.args) != 1 or decorator.keywords or _literal_name(decorator.args[0]) is None
        ):
            return False
        binding = _factory_binding(tree, decorator.func.id)
        if (
            not isinstance(binding, ast.ImportFrom) or binding.module != "unittest.mock" or binding.level
            or binding.lineno >= decorator.lineno
            or not any((alias.asname or alias.name) == decorator.func.id and alias.name == "patch" for alias in binding.names)
        ):
            return False
    return True


class ModuleMappingWitnesses:
    def __init__(self, source: Path, payload: Mapping, numbers_by_id: Mapping[str, int]):
        self.source = source
        self.numbers_by_id = numbers_by_id
        self.files = defaultdict(set)
        self.methods = defaultdict(set)
        for node, number in numbers_by_id.items():
            self.files[node.partition("::")[0]].add(number)
            self.methods[node.split("[", 1)[0]].add(number)
        self.recorded = ModuleMappingRecorder()
        self.recorded.merge(payload)
        reliable = payload.get("reliable")
        self.unobserved = None if not isinstance(reliable, list) else set(numbers_by_id.values()) - {
            numbers_by_id[node] for node in reliable if node in numbers_by_id
        }
        self.reader_census = payload.get("reader_census")
        self.census_verified = None
        self.census = None

    def _load(self, tree_for):
        if self.census is not None:
            return self.census
        result = {}
        for package in ("sqlglot", "tests"):
            for path in (self.source / package).rglob("*.py"):
                relative = path.relative_to(self.source).as_posix()
                tree = tree_for(relative)
                if tree is None:
                    self.recorded.unresolved = True
                    return {}
                result[relative] = tree, namespace_accesses(relative, tree)
        self.census = result
        return result

    @staticmethod
    def _imported_field(path, tree, module, field):
        current = source_module(path)
        package = current if path.endswith("/__init__.py") else current.rpartition(".")[0]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            name = node.module or ""
            if node.level:
                try:
                    name = resolve_name("." * node.level + name, package)
                except ImportError:
                    continue
            if name == module and any(alias.name in {field, "*"} for alias in node.names):
                return True
        return False

    @staticmethod
    def _test_exports(tree, relevant):
        names = set()
        for statement in tree.body:
            if not any(node in relevant for node in ast.walk(statement)):
                continue
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return None
            names.add(statement.name)
        return names

    def _scoped_helpers(self, path, tree, helper_exports, exposed, mentions):
        nodes = tuple(ast.walk(tree))
        parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        imports = set()
        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or statement.level:
                continue
            for alias in statement.names:
                name = alias.asname or alias.name
                if (
                    name in exposed and (statement.module, alias.name) in helper_exports
                    and _factory_binding(tree, name) is statement
                ):
                    imports.add(name)
        if not imports:
            return None
        if any(
            isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in imports
            or isinstance(node, ast.Constant) and type(node.value) is str and node.value in imports
            for node in nodes
        ):
            return None

        callers = set()
        owners = set()
        methods = set()
        declarations = set()
        for node in nodes:
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id not in imports:
                continue
            call = parents.get(node)
            if not isinstance(call, ast.Call) or call.func is not node:
                return None
            body = call
            method = parents.get(body)
            while method is not None and not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(method, (ast.ClassDef, ast.Lambda)):
                    return None
                body, method = method, parents.get(method)
            if not _ordinary_test(tree, method) or body not in method.body:
                return None
            owner = parents.get(method)
            if owner is tree:
                qualified = f"{path}::{method.name}"
            elif isinstance(owner, ast.ClassDef) and _stdlib_test_class(tree, owner):
                qualified = f"{path}::{owner.name}::{method.name}"
                owners.add(owner.name)
                declarations.add(owner)
            else:
                return None
            matched = self.methods.get(qualified)
            if not matched:
                return None
            callers.update(matched)
            methods.add(method.name)
            declarations.add(method)
        local = self.files[path]
        visible = owners | methods
        if not callers or mentions(visible) - local or mentions(exposed - imports - visible) & local:
            return None
        if any(
            isinstance(node, (ast.Name, ast.Attribute)) and _name(node) in visible
            or isinstance(node, ast.Constant) and type(node.value) is str and node.value in visible
            or isinstance(node, ast.alias) and node.name in visible
            or isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in visible
            and node not in declarations
            for node in nodes
        ):
            return None
        return callers

    def _test_readers(self, module, field, census, mentions):
        result = set()
        exposed = set()
        helpers = set()
        support = []
        for path, (tree, accesses) in census.items():
            if not path.startswith("tests/"):
                continue
            relevant = {
                node for node, access in accesses.items()
                if access.mode != "local" and access.field in {None, field}
            }
            direct = {
                node for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and node.value == field
                or isinstance(node, (ast.Name, ast.Attribute))
                and isinstance(node.ctx, ast.Load) and _name(node) == field
            }
            if self._imported_field(path, tree, module, field):
                return None
            if relevant or direct:
                result.update(self.files.get(path, ()))
            names = self._test_exports(tree, relevant | direct)
            if names is None:
                return None
            exposed.update(names)
            if path not in self.files:
                helpers.update(
                    (source_module(path), statement.name) for statement in tree.body
                    if _plain_helper(tree, statement, relevant | direct)
                )
            support.append((path, tree, accesses))
        while exposed:
            previous = set(exposed)
            for _, tree, _ in support:
                references = {
                    node for node in ast.walk(tree)
                    if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in exposed)
                    or (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in exposed)
                    or (isinstance(node, ast.alias) and node.name in exposed)
                }
                for statement in tree.body:
                    if isinstance(statement, ast.ImportFrom):
                        for alias in statement.names:
                            if alias.name in exposed and alias.asname is not None:
                                exposed.add(alias.asname)
                        references.difference_update(statement.names)
                names = self._test_exports(tree, references)
                if names is None:
                    return None
                exposed.update(names)
            if exposed == previous:
                break
        protected = mentions(exposed)
        for path, tree, accesses in support:
            local = self.files.get(path)
            if not local or not protected & local or result & local or accesses:
                continue
            callers = self._scoped_helpers(path, tree, helpers, exposed, mentions)
            if callers is not None:
                protected.difference_update(local)
                protected.update(callers)
        result.update(protected)
        return result

    def select(self, path: str, module: str, field: str, *, tree_for, mentions) -> set[int] | None:
        if self.recorded.unresolved or self.unobserved is None:
            return None
        if self.reader_census is not None:
            if self.census_verified is None:
                try:
                    self.census_verified = namespace_source_digest(self.source) == self.reader_census["source_digest"]
                except OSError:
                    self.census_verified = False
            origin = f"{module}.{field}"
            if self.census_verified and origin in self.reader_census["readers"]:
                readers = self.reader_census["readers"][origin]
                if readers is None:
                    return None
                return self.unobserved | {self.numbers_by_id[node] for node in readers if node in self.numbers_by_id}
        census = self._load(tree_for)
        if self.recorded.unresolved:
            return None
        for relative, (tree, accesses) in census.items():
            if not relative.startswith("sqlglot/"):
                continue
            if self._imported_field(relative, tree, module, field):
                return None
            for access in accesses.values():
                if access.field not in {None, field}:
                    continue
                if access.mode == "local":
                    if relative == path:
                        return None
                elif (
                    access.mode == "escape" or access.site not in self.recorded.sites
                    or access.site in self.recorded.unknown
                ):
                    return None
        readers = set(self.recorded.dynamic_code)
        for (_, receiver, actual_field), nodes in self.recorded.reads.items():
            if receiver == module and actual_field in {field, "*"}:
                readers.update(nodes)
        if None in readers:
            return None
        tests = self._test_readers(module, field, census, mentions)
        if tests is None:
            return None
        return tests | self.unobserved | {
            self.numbers_by_id[node] for node in readers if node in self.numbers_by_id
        }
