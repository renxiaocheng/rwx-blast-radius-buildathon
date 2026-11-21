"""Prove a mapping-derived membership set has no unobserved consumers."""

import ast
from collections import defaultdict
from importlib.util import resolve_name

from dialect_impact import _factory_binding, _module_name


def _imported_class(tree, path, node):
    if not isinstance(node, ast.Name):
        return None
    binding = _factory_binding(tree, node.id)
    if not isinstance(binding, ast.ImportFrom) or binding.lineno >= node.lineno:
        return None
    aliases = [alias for alias in binding.names if (alias.asname or alias.name) == node.id]
    if len(aliases) != 1:
        return None
    module = binding.module or ""
    if binding.level:
        origin = _module_name(path)
        package = origin if path.endswith("/__init__.py") else origin.rpartition(".")[0]
        try:
            module = resolve_name("." * binding.level + module, package)
        except ImportError:
            return None
    return f"{module}.{aliases[0].name}"


def _direct_bases(tree, path, origin):
    names = origin.removeprefix(_module_name(path) + ".").split(".")[:-1]
    scope = tree
    for name in names:
        matches = [node for node in scope.body if isinstance(node, ast.ClassDef) and node.name == name]
        if len(matches) != 1:
            return set()
        scope = matches[0]
    if not isinstance(scope, ast.ClassDef):
        return set()
    return {qualified for base in scope.bases if (qualified := _imported_class(tree, path, base)) is not None}


def _same_receiver(value, receiver):
    return isinstance(value, ast.Name) and value.id == receiver


def _set_derivation(node, parents, mapping):
    method = parents.get(node)
    if not isinstance(method, ast.Attribute) or method.value is not node or method.attr != "values":
        return None
    call = parents.get(method)
    star = parents.get(call)
    entries = parents.get(star)
    assignment = parents.get(entries)
    if (
        not isinstance(call, ast.Call) or call.func is not method or call.args or call.keywords
        or not isinstance(star, ast.Starred) or not isinstance(entries, ast.Set)
        or not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1
    ):
        return None
    target = assignment.targets[0]
    if (
        not isinstance(target, ast.Attribute) or target.attr == mapping or not isinstance(target.value, ast.Name)
        or not isinstance(node.value, ast.Name) or node.value.id != target.value.id
        or len(entries.elts) != 3 or not all(isinstance(item, ast.Starred) for item in entries.elts)
    ):
        return None
    receiver = node.value.id
    inherited = keys = None
    for item in entries.elts:
        value = item.value
        if isinstance(value, ast.Attribute) and value.attr == target.attr and _same_receiver(value.value, receiver):
            inherited = value
        elif (
            isinstance(value, ast.Call) and not value.args and not value.keywords
            and isinstance(value.func, ast.Attribute) and value.func.attr == "keys"
            and isinstance(value.func.value, ast.Attribute) and value.func.value.attr == mapping
            and _same_receiver(value.func.value.value, receiver)
        ):
            keys = value.func.value
    return (target.attr, inherited, keys) if inherited is not None and keys is not None else None


def _membership(node, parent):
    return (
        isinstance(parent, ast.Compare) and len(parent.ops) == 1
        and isinstance(parent.ops[0], (ast.In, ast.NotIn)) and parent.comparators[0] is node
    )


def _key_read(node, parents):
    parent = parents.get(node)
    if _membership(node, parent):
        return True
    if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
        return not any(isinstance(item, ast.Slice) for item in ast.walk(parent.slice))
    if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
        return True
    if (
        isinstance(parent, ast.Attribute) and parent.value is node
        and parent.attr in {"get", "pop", "__getitem__", "__contains__"}
    ):
        call = parents.get(parent)
        return isinstance(call, ast.Call) and call.func is parent
    return False


def _loads(field, tree):
    nodes = tuple(ast.walk(tree))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    matches = [
        node for node in nodes
        if isinstance(node, (ast.Attribute, ast.Name)) and isinstance(node.ctx, ast.Load)
        and (node.attr if isinstance(node, ast.Attribute) else node.id) == field
    ]
    return matches, parents


def derived_mapping_readers(path, target, *, required, tree_for, attributes, mentions):
    """Select consumers of a class set computed from unchanged keys and string values."""
    if target.kind != "dict" or not target.literal_string_value:
        return None
    origin = target.origin.rpartition(".")[0]
    bases = _direct_bases(tree_for(path), path, target.origin)
    if not bases:
        return None
    derivations = set()
    for relative in sorted({site[0] for site in required}):
        tree = tree_for(relative)
        nodes, parents = _loads(target.field, tree)
        derived_nodes = set()
        for node in nodes:
            if not isinstance(node, ast.Attribute):
                continue
            derived = _set_derivation(node, parents, target.field)
            if derived is None:
                continue
            field, inherited, keys = derived
            derivations.add((relative, inherited.lineno, inherited.col_offset, field))
            derived_nodes.update((node, keys))
            # Both key and value loads must match the same live family.
            for read in (node, keys):
                site = relative, read.lineno, target.field
                if not any(saved == site and origin in names for saved, names, _ in attributes.by_field[target.field]):
                    return None
        for node in nodes:
            if node in derived_nodes or _key_read(node, parents):
                continue
            if not isinstance(node, ast.Attribute):
                return None
            if _imported_class(tree, relative, node.value) not in bases:
                return None
    if not derivations:
        return None
    by_child = defaultdict(set)
    for relative, line, column, field in derivations:
        by_child[field].add((relative, line, column))
    selected = set()
    for field, inherited_by_site in by_child.items():
        explicit = attributes.explicit.get(field, set())
        shared_sites = {(relative, line, field) for relative, line, _ in inherited_by_site}
        if (
            not explicit or not explicit <= attributes.recorder.sites or field in attributes.bare_fields
            or not all(any(saved == site and origin in names for saved, names, _ in attributes.by_field[field]) for site in shared_sites)
            or attributes._has_dynamic_read(origin, field)
        ):
            return None
        for relative in sorted({site[0] for site in explicit}):
            tree = tree_for(relative)
            nodes, parents = _loads(field, tree)
            for node in nodes:
                if not isinstance(node, ast.Attribute):
                    return None
                if (
                    (relative, node.lineno, node.col_offset) in inherited_by_site
                    or _membership(node, parents.get(node))
                    or _imported_class(tree, relative, node.value) in bases
                ):
                    if (relative, node.lineno, field) in shared_sites and (
                        relative, node.lineno, node.col_offset
                    ) not in inherited_by_site:
                        return None
                    continue
                return None
        for site, names, nodes in attributes.by_field[field]:
            if origin not in names or site in shared_sites:
                continue
            if None in nodes:
                return None
            selected.update(attributes.numbers_by_id[node] for node in nodes if node in attributes.numbers_by_id)
        selected.update(attributes._extra_tests({field}, mentions))
    return selected
