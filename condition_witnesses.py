"""Record unmodified guard inputs and compare supported edits at scoring time."""

import ast
import builtins
import math
import operator
import sys
import symtable
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import GetSetDescriptorType, MemberDescriptorType

from patch_impact import PatchEdit, apply_edits, closure_signature


PROBE = "__rwx_record_condition__"
MISSING = object()
SCALAR_TYPES = frozenset(map(id, (type(None), bool, int, float, str, bytes)))
CONTAINERS = frozenset(map(id, (list, tuple, dict, str, bytes, set, frozenset, range)))
MAX_STATES = 6
MAX_HITS = 3000
RELATIONS = {
    ast.Eq: ("eq", operator.eq), ast.NotEq: ("ne", operator.ne),
    ast.Lt: ("lt", operator.lt), ast.LtE: ("le", operator.le),
    ast.Gt: ("gt", operator.gt), ast.GtE: ("ge", operator.ge),
    ast.Is: ("is", operator.is_), ast.IsNot: ("isnot", operator.is_not),
}
OPERATORS = {name: func for name, func in RELATIONS.values()}


def scalar(value):
    kind = type(value)
    if kind is type(None):
        return ("none", None)
    if kind is bool:
        return ("bool", value)
    if kind is int and value.bit_length() <= 80:
        return ("int", value)
    if kind is float and math.isfinite(value):
        return ("float", value.hex())
    if kind is str and len(value) <= 120:
        return ("str", value)
    if kind is bytes and len(value) <= 120:
        return ("bytes", value.hex())
    return None


def unpack(value):
    if value is None:
        return MISSING
    tag, raw = value
    if tag == "float":
        return float.fromhex(raw)
    if tag == "bytes":
        return bytes.fromhex(raw)
    return raw


def projection(node):
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Attribute):
        parent = projection(node.value)
        if parent is not None:
            return ("attribute", parent, node.attr)
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            operand = projection(node.args[0])
            if operand is not None:
                return ("length", operand)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            operand = projection(node.func.value)
            argument = node.args[0]
            if operand is not None and isinstance(argument, ast.Constant):
                key = scalar(argument.value)
                if key is not None:
                    return ("get", operand, key)
    return None


def plan(node):
    if isinstance(node, ast.Constant):
        value = scalar(node.value)
        if value is not None:
            return ("literal", value)
    if isinstance(node, ast.BoolOp):
        values = tuple(plan(value) for value in node.values)
        if None not in values:
            return ("and" if isinstance(node.op, ast.And) else "or", values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
        value = plan(node.operand)
        if value is not None:
            operation = {ast.Not: "not", ast.USub: "negative", ast.UAdd: "positive"}[type(node.op)]
            return (operation, value)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in RELATIONS:
        left, right = plan(node.left), plan(node.comparators[0])
        if left is not None and right is not None:
            return ("relation", RELATIONS[type(node.ops[0])][0], left, right)
    atom = projection(node)
    return ("atom", atom) if atom is not None else None


def atoms(expr):
    operation = expr[0]
    if operation == "atom":
        return {expr[1]}
    if operation == "literal":
        return set()
    if operation in {"not", "negative", "positive"}:
        return atoms(expr[1])
    if operation in {"and", "or"}:
        return set().union(*(atoms(item) for item in expr[1]))
    return atoms(expr[2]) | atoms(expr[3])


def raw_slot(cls, name):
    for owner in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        if name in namespace:
            return namespace[name]
    return MISSING


def raw_attribute(instance, name):
    if instance is MISSING:
        return MISSING
    cls = type(instance)
    if raw_slot(cls, "__getattribute__") is not object.__getattribute__:
        return MISSING
    member = raw_slot(cls, name)
    if type(member) is MemberDescriptorType:
        try:
            return member.__get__(instance, cls)
        except AttributeError:
            return MISSING
    if member is not MISSING and any(
        raw_slot(type(member), slot) is not MISSING for slot in ("__set__", "__delete__")
    ):
        return MISSING
    descriptor = raw_slot(cls, "__dict__")
    if type(descriptor) is GetSetDescriptorType:
        try:
            dictionary = descriptor.__get__(instance, cls)
        except AttributeError:
            dictionary = None
        if type(dictionary) is dict and name in dictionary:
            return dictionary[name]
    if member is MISSING or raw_slot(type(member), "__get__") is not MISSING:
        return MISSING
    return member


def load_name(frame, name):
    if name in frame.f_locals:
        return frame.f_locals[name]
    code = frame.f_code
    if name in code.co_varnames or name in code.co_cellvars or name in code.co_freevars:
        return MISSING
    return frame.f_globals.get(name, frame.f_builtins.get(name, MISSING))


def observe_atom(atom, frame):
    kind = atom[0]
    if kind == "name":
        return load_name(frame, atom[1])
    instance = observe_atom(atom[1], frame)
    if kind == "attribute":
        return raw_attribute(instance, atom[2])
    if kind == "length" and id(type(instance)) in CONTAINERS and load_name(frame, "len") is builtins.len:
        try:
            return len(instance)
        except OverflowError:
            return MISSING
    if kind == "get" and type(instance) is dict and len(instance) <= 128:
        if all(id(type(key)) in SCALAR_TYPES for key in instance):
            return instance.get(unpack(atom[2]))
    return MISSING


def evaluate(expr, observed):
    operation = expr[0]
    if operation == "literal":
        return unpack(expr[1])
    if operation == "atom":
        return unpack(observed.get(expr[1]))
    if operation in {"not", "negative", "positive"}:
        result = evaluate(expr[1], observed)
        if result is MISSING:
            return MISSING
        if operation == "not":
            return not result
        if type(result) in (bool, int, float):
            return -result if operation == "negative" else +result
        return MISSING
    if operation in {"and", "or"}:
        result = MISSING
        for part in expr[1]:
            result = evaluate(part, observed)
            if result is MISSING or bool(result) == (operation == "or"):
                break
        return result
    _, relation, lhs, rhs = expr
    left, right = evaluate(lhs, observed), evaluate(rhs, observed)
    if left is MISSING or right is MISSING:
        return MISSING
    # Independently serialized scalar values do not retain object identity.
    if relation in {"is", "isnot"} and not (
        type(left) in (type(None), bool) or type(right) in (type(None), bool)
    ):
        return MISSING
    try:
        return OPERATORS[relation](left, right)
    except (TypeError, ValueError, OverflowError):
        return MISSING


@dataclass(frozen=True)
class Replacement:
    site: str
    first: int
    last: int
    original: tuple
    modified: tuple


def guard_site(path, node):
    return f"{path}:{node.lineno}:{node.col_offset}"


class Guards(ast.NodeTransformer):
    def __init__(self, path, *, instrument=None, touched=()):
        self.path = path
        self.instrument = instrument
        self.touched = touched
        self.matches = []
        self.function_depth = 0

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self.function_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        original_depth = self.function_depth
        self.function_depth = 0
        try:
            return self.generic_visit(node)
        finally:
            self.function_depth = original_depth

    def visit_If(self, node):
        test = node.test
        relevant = self.function_depth and isinstance(test, (ast.BoolOp, ast.Compare))
        if self.instrument is not None and relevant:
            expression = plan(test)
            if expression is not None:
                ordered = tuple(sorted(atoms(expression), key=repr))
                if 0 < len(ordered) <= 6:
                    key = guard_site(self.path, test)
                    if self.instrument(key, ordered):
                        def branch(value):
                            return ast.Call(ast.Name(PROBE, ast.Load()), [ast.Constant(key), ast.Constant(value)], [])

                        # An if-expression keeps the original guard in a truth
                        # context, so Python does not call a custom __bool__ twice.
                        node.test = ast.copy_location(ast.IfExp(test, branch(True), branch(False)), test)
        elif relevant and any(test.lineno <= line <= test.end_lineno for line in self.touched):
            expression = plan(test)
            if expression is not None:
                self.matches.append((test, expression))
                node.test = ast.Constant(None)
        return self.generic_visit(node)

    visit_While = visit_If


class ConditionRecorder:
    def __init__(self, audit, important):
        self.audit = audit
        self.important = frozenset(important)
        self.registered = {}
        self.shared = set()
        self.tests = {}
        self.hits = {}

    def transform(self, path, tree):
        def register(site, projected):
            self.registered[site] = projected
            return True

        if self.important:
            return Guards(path, instrument=register).visit(tree)
        return tree

    def __call__(self, site, actual):
        if not self.audit.active:
            return actual
        node = self.audit.node_id
        if not self.audit.in_call or node is None:
            self.shared.add(site)
        elif node in self.important and site not in self.shared:
            key = (site, node)
            if key not in self.tests:
                self.tests[key] = set()
            states = self.tests[key]
            if states is not None:
                count = self.hits.get(key, 0) + 1
                if count > MAX_HITS:
                    self.tests[key] = None
                else:
                    self.hits[key] = count
                    frame = sys._getframe(1)
                    values = tuple(scalar(observe_atom(atom, frame)) for atom in self.registered[site])
                    states.add((actual, values))
                    if len(states) > MAX_STATES:
                        self.tests[key] = None
        return actual

    def merge(self, data):
        self.shared.update(data.get("shared", ()))
        for site, tests in data.get("sites", {}).items():
            for node, rows in tests.items():
                key = (site, node)
                previous = self.tests.get(key, set())
                if previous is None or rows is None:
                    self.tests[key] = None
                    continue
                previous.update(
                    (actual, tuple(None if value is None else tuple(value) for value in values))
                    for actual, values in rows
                )
                self.tests[key] = previous if len(previous) <= MAX_STATES else None

    def export(self):
        grouped = {}
        for (site, node), rows in self.tests.items():
            if site not in self.shared:
                grouped.setdefault(site, {})[node] = None if rows is None else sorted(rows, key=repr)
        return {"shared": sorted(self.shared), "sites": grouped}


def replacements(name: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> list[Replacement]:
    changed = apply_edits(lines, edits)
    if changed is None:
        return []
    patched, before, after = changed
    # Pure insert/delete sites need stronger proof of unchanged scopes.
    if before.gaps or after.gaps or not before.lines or not after.lines:
        return []
    try:
        old_source = "\n".join(lines) + "\n"
        new_source = "\n".join(patched) + "\n"
        old_tree = ast.parse(old_source, name, type_comments=True)
        new_tree = ast.parse(new_source, name, type_comments=True)
        old = Guards(name, touched=before.lines)
        new = Guards(name, touched=after.lines)
        old.visit(old_tree)
        new.visit(new_tree)
        if not old.matches or len(old.matches) != len(new.matches) or ast.dump(old_tree) != ast.dump(new_tree):
            return []
        if closure_signature(symtable.symtable(old_source, name, "exec")) != closure_signature(
            symtable.symtable(new_source, name, "exec")
        ):
            return []
    except (SyntaxError, ValueError, RecursionError):
        return []
    for changes, matches in ((before, old.matches), (after, new.matches)):
        if not all(any(test.lineno <= line <= test.end_lineno for test, _ in matches) for line in changes.lines):
            return []
    changes = []
    for (test, original), (_, modified) in zip(old.matches, new.matches):
        if not atoms(modified).issubset(atoms(original)):
            return []
        if original != modified:
            changes.append(Replacement(guard_site(name, test), test.lineno, test.end_lineno, original, modified))
    return changes


def equivalent_for(change, rows):
    if not rows:
        return False
    keys = sorted(atoms(change.original), key=repr)
    for actual, values in rows:
        if len(values) != len(keys):
            return False
        observed = dict(zip(keys, values))
        old = evaluate(change.original, observed)
        new = evaluate(change.modified, observed)
        if old is MISSING or new is MISSING or bool(old) != actual or bool(new) != actual:
            return False
    return True
