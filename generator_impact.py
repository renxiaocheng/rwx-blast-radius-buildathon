"""Observe deferred calls to the pinned SQLGlot rename_func factory."""

import ast
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
import tokenize

from dialect_impact import _comments, _factory_binding, _module_name
from mapping_impact import declarations
from patch_impact import PatchEdit, apply_edits


FACTORY_PATH = "sqlglot/dialects/dialect.py"
FACTORY_MODULE = "sqlglot.dialects.dialect"
FACTORY_NAME = "rename_func"
FACTORY_DIGEST = "76e8553773db08405642d8ea65cd680322d94a7f1a94b2efdba4015728116bda"
GENERATOR_DIGEST = "d8d75b37fd8dff73f2a8bb0de075e49e37d2576a6ac0468a9154e70fa8e24735"
TRANSFORMS_DIGEST = "53e846a2e8285367564477839baf69a2309f087cd7fc1b9459c302fa8885876d"
HOOK = "__rwx_observe_rename_func__"
VERSION = 1
REFLECTIVE_READS = {
    "rename_func", "TRANSFORMS", "_dispatch", "_DISPATCH_CACHE", "__closure__",
    "__code__", "getclosurevars", "getsource", "getsourcelines", "cell_contents", "co_freevars",
}


@dataclass
class GeneratorRecorder:
    available: bool = False
    unsupported: bool = False
    reads: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    shared: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)
    unknown_shared: bool = False

    def observe(self, value, node: str | None) -> None:
        if type(value) is not str:
            if node is None:
                self.unknown_shared = True
            else:
                self.unknown.add(node)
        elif node is None:
            self.shared.add(value)
        else:
            self.reads[value].add(node)

    def export(self) -> dict:
        return {
            "version": VERSION, "available": self.available, "unsupported": self.unsupported,
            "reads": {name: sorted(nodes) for name, nodes in sorted(self.reads.items())},
            "shared": sorted(self.shared), "unknown": sorted(self.unknown),
            "unknown_shared": self.unknown_shared,
        }

    def merge(self, evidence: Mapping | None) -> None:
        if not evidence or evidence.get("version") != VERSION or not evidence.get("available"):
            self.unsupported = True
            return
        self.available = True
        self.unsupported |= evidence.get("unsupported", False)
        self.shared.update(evidence.get("shared", ()))
        self.unknown.update(evidence.get("unknown", ()))
        self.unknown_shared |= evidence.get("unknown_shared", False)
        for name, nodes in evidence.get("reads", {}).items():
            self.reads[name].update(nodes)


def pinned_factory(source: Path) -> bool:
    try:
        return sha256((source / FACTORY_PATH).read_bytes()).hexdigest() == FACTORY_DIGEST
    except OSError:
        return False


def generator_instrumentation(source: Path, recorder: GeneratorRecorder, current_node: Callable[[], str | None]):
    """Add a pre-call observation without wrapping or replacing the handler."""
    source = Path(source)

    def read(name):
        recorder.observe(name, current_node())

    def transform(path: str, tree: ast.Module) -> ast.Module:
        if path != FACTORY_PATH:
            return tree
        if not pinned_factory(source):
            recorder.unsupported = True
            return tree
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == FACTORY_NAME]
        if len(functions) != 1:
            recorder.unsupported = True
            return tree
        factory = functions[0]
        if len(factory.body) != 1 or not isinstance(factory.body[0], ast.Return) or not isinstance(factory.body[0].value, ast.Lambda):
            recorder.unsupported = True
            return tree
        deferred = factory.body[0].value
        original = deferred.body
        event = ast.Call(
            func=ast.Name(id=HOOK, ctx=ast.Load()),
            args=[ast.Name(id="name", ctx=ast.Load())],
            keywords=[],
        )
        # The recorder returns None, so the original expression is evaluated
        # and returned exactly once. A raising handler is still observed.
        deferred.body = ast.copy_location(ast.BoolOp(op=ast.Or(), values=[event, original]), original)
        recorder.available = True
        return tree

    return transform, {HOOK: read}


def literal_call(node: ast.AST) -> tuple[str, ast.Constant] | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords or len(node.args) != 1:
        return None
    operand = node.args[0]
    if isinstance(operand, ast.Constant) and type(operand.value) is str:
        return node.func.id, operand
    return None


def rename_delta(path: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> str | None:
    """Detect a single allowed transform replacement, independent of labels."""
    filename = PurePosixPath(path)
    if filename.suffix != ".py" or filename.parts[:2] not in {("sqlglot", "generators"), ("sqlglot", "dialects")}:
        return None
    edits = tuple(edits)
    if not edits or any(not edit.removed or len(edit.removed) != len(edit.added) for edit in edits):
        return None
    patched = apply_edits(lines, edits)
    if patched is None:
        return None
    old_text, new_text = ("\n".join(file) + "\n" for file in (lines, patched[0]))
    try:
        if list(_comments(old_text)) != list(_comments(new_text)):
            return None
        old = ast.parse(old_text, path, type_comments=True)
        new = ast.parse(new_text, path, type_comments=True)
    except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
        return None
    module = _module_name(path)
    old_assignments, new_assignments = declarations(old, module), declarations(new, module)
    for origin, assignment in old_assignments.items():
        updated = new_assignments.get(origin)
        if (
            not origin.endswith(".TRANSFORMS") or origin.removesuffix(".TRANSFORMS") == module
            or assignment is None or updated is None
            or not isinstance(assignment.value, ast.Dict) or not isinstance(updated.value, ast.Dict)
        ):
            continue
        for key, previous, candidate in zip(assignment.value.keys, assignment.value.values, updated.value.values):
            if key is None or ast.dump(previous) == ast.dump(candidate):
                continue
            prior_call, changed_call = literal_call(previous), literal_call(candidate)
            if prior_call is None or changed_call is None or prior_call[0] != changed_call[0]:
                continue
            name, previous_literal = prior_call
            if previous_literal.value == changed_call[1].value or not all(
                previous_literal.lineno <= edit.start <= edit.start + len(edit.removed) - 1 <= previous_literal.end_lineno
                for edit in edits
            ):
                continue
            binding = _factory_binding(old, name)
            if (
                not isinstance(binding, ast.ImportFrom) or binding.level or binding.module != FACTORY_MODULE
                or binding.lineno >= assignment.lineno
                or not any(alias.name == FACTORY_NAME and (alias.asname or alias.name) == name for alias in binding.names)
            ):
                continue
            candidate.args[0] = previous_literal
            if ast.dump(old) == ast.dump(new):
                return previous_literal.value
            candidate.args[0] = changed_call[1]
    return None


def audited_copies(source: Path) -> bool:
    """Reject consumers beyond copied entries, key checks, and pinned calls."""
    known = {
        FACTORY_PATH: FACTORY_DIGEST,
        "sqlglot/generator.py": GENERATOR_DIGEST,
        "sqlglot/transforms.py": TRANSFORMS_DIGEST,
    }
    try:
        if any(sha256((source / path).read_bytes()).hexdigest() != digest for path, digest in known.items()):
            return False
        for path in (source / "sqlglot").rglob("*.py"):
            if path.relative_to(source).as_posix() in known:
                continue
            raw = path.read_bytes()
            if b"TRANSFORMS" not in raw:
                continue
            tree = ast.parse(raw, filename=str(path))
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "TRANSFORMS" and isinstance(node.ctx, ast.Load):
                    return False
                if isinstance(node, ast.Constant) and node.value == "TRANSFORMS":
                    return False
                if isinstance(node, ast.Attribute) and node.attr == "TRANSFORMS" and isinstance(node.ctx, ast.Load):
                    if not copied_or_key_only(node, parents):
                        return False
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def copied_or_key_only(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, ast.Dict):
        return any(key is None and value is node for key, value in zip(parent.keys, parent.values))
    if isinstance(parent, ast.Compare):
        return len(parent.ops) == 1 and isinstance(parent.ops[0], (ast.In, ast.NotIn)) and parent.comparators[0] is node
    if not isinstance(parent, ast.Attribute) or parent.value is not node or parent.attr != "items":
        return False
    call = parents.get(parent)
    loop = parents.get(call)
    target = parents.get(loop)
    if (
        not isinstance(call, ast.Call) or call.func is not parent or call.args or call.keywords
        or not isinstance(loop, ast.comprehension) or loop.iter is not call or loop.is_async
        or not isinstance(target, ast.DictComp) or len(target.generators) != 1
        or not isinstance(loop.target, (ast.Tuple, ast.List)) or len(loop.target.elts) != 2
    ):
        return False
    key, value = loop.target.elts
    return (
        isinstance(key, ast.Name) and isinstance(value, ast.Name)
        and isinstance(target.key, ast.Name) and target.key.id == key.id
        and isinstance(target.value, ast.Name) and target.value.id == value.id
        and not any(
            isinstance(item, ast.Name) and item.id == value.id
            for condition in loop.ifs for item in ast.walk(condition)
        )
    )


class GeneratorImpact:
    def __init__(self, source: Path, evidence: GeneratorRecorder | Mapping, numbers_by_id: Mapping[str, int]):
        self.source = Path(source)
        self.evidence = evidence if isinstance(evidence, GeneratorRecorder) else GeneratorRecorder()
        if not isinstance(evidence, GeneratorRecorder):
            self.evidence.merge(evidence)
        self.numbers = numbers_by_id
        self.pin = None

    def select(
        self, path: str, lines: Sequence[str], edits: Iterable[PatchEdit], *,
        mentions: Callable[[set[str]], set[int]] | None = None,
        complete: set[str] | frozenset[str] = frozenset(),
    ) -> set[int] | None:
        evidence = self.evidence
        if not complete or mentions is None or not evidence.available or evidence.unsupported or evidence.unknown_shared:
            return None
        delta = rename_delta(path, lines, edits)
        if delta is None or delta in evidence.shared:
            return None
        if self.pin is None:
            self.pin = audited_copies(self.source)
        if not self.pin:
            return None
        nodes = evidence.unknown | evidence.reads.get(delta, set()) | (self.numbers.keys() - complete)
        return {self.numbers[node] for node in nodes if node in self.numbers} | mentions(REFLECTIVE_READS)
