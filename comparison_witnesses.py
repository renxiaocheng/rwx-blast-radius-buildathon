"""Collect alternate-comparison witnesses without invoking custom equality hooks."""

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import EnumType
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
import operator
from pathlib import Path
import re
import sys
from typing import Protocol


Site = tuple[str, int, int, str]
Sample = tuple[str, str, bool]

OPERATIONS = {"Is": operator.is_, "IsNot": operator.is_not, "Eq": operator.eq, "NotEq": operator.ne}
ALTERNATIVES = {"Is": "Eq", "IsNot": "NotEq", "Eq": "Is", "NotEq": "IsNot"}
OPERATOR_TOKEN = {
    "Is": re.compile(rb"\bis\b"),
    "IsNot": re.compile(rb"\bis[ \t]+not\b"),
    "Eq": re.compile(rb"=="),
    "NotEq": re.compile(rb"!="),
}

COMPARISON_HELPER = "__rwx_witness_compare__"
ATTRIBUTE_HELPER = "__rwx_witness_attribute__"
MAX_VALUE_LENGTH = 64
MAX_DISTINCT_SAMPLES = 12


class Edit(Protocol):
    start: int
    removed: list[str]
    added: list[str]


def comparison_operation(node: ast.Compare) -> str | None:
    if len(node.ops) != 1:
        return None
    op = type(node.ops[0]).__name__
    return op if op in OPERATIONS else None


def raw_method(cls: type, name: str):
    """Read special-method definitions without binding descriptors."""
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return None


def snapshot(value) -> str | None:
    """Encode bounded original values or a raw special-method category."""
    cls = type(value)
    if value is None:
        return "n"
    if value is Ellipsis:
        return "e"
    if value is NotImplemented:
        return "x"
    if cls is bool:
        return "b1" if value else "b0"
    if cls is int:
        return "i" + str(value) if value.bit_length() <= 64 else None
    if cls is float:
        return "f" + value.hex()
    if cls is complex:
        return "c" + value.real.hex() + "," + value.imag.hex()
    if cls is str:
        return "s" + value if len(value) <= MAX_VALUE_LENGTH else None
    if cls is bytes:
        return "y" + value.hex() if len(value) <= MAX_VALUE_LENGTH else None
    meta = type(cls)
    if meta is not type and meta is not EnumType:
        return None
    equality, inequality = raw_method(cls, "__eq__"), raw_method(cls, "__ne__")
    if equality is object.__eq__ and inequality is object.__ne__:
        return "o"
    if equality is str.__eq__ and inequality is str.__ne__:
        if any(base is str for base in type.__getattribute__(cls, "__mro__")):
            text = str.__str__(value)
            return "u" + text if len(text) <= MAX_VALUE_LENGTH else None
    return None


def compare_with_witness(audit, path, line, column, op, left, right):
    wants_comparison = getattr(audit, "wants_comparison", None)
    if wants_comparison is None or wants_comparison():
        lhs, rhs = snapshot(left), snapshot(right)
        sample = (lhs, rhs, left is right) if lhs is not None and rhs is not None else None
        audit.record_comparison(path, line, column, op, sample)
    # The pinned program's own comparison still runs exactly once. The probe
    # does not perform or predict a competing operation during collection.
    return OPERATIONS[op](left, right)


def reveal_attribute(audit, owner, field, path, line):
    audit.record_attribute(owner, field, path, line)
    return getattr(owner, field)


class Instrument(ast.NodeTransformer):
    def __init__(self, path: str, audit, comparisons: bool, attribute_names: frozenset[str]):
        self.path = path
        self.comparisons = comparisons
        self.attribute_names = attribute_names
        self.register_comparison = getattr(audit, "register_comparison_site", None)
        self.register_attribute = getattr(audit, "register_attribute_site", None)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        op = comparison_operation(node) if self.comparisons else None
        if op is None:
            return node
        site = (self.path, node.lineno, node.col_offset, op)
        if self.register_comparison is not None:
            self.register_comparison(*site)
        return ast.copy_location(ast.Call(
            func=ast.Name(id=COMPARISON_HELPER, ctx=ast.Load()),
            args=[*(ast.Constant(value=value) for value in site), node.left, node.comparators[0]],
            keywords=[],
        ), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load) or node.attr not in self.attribute_names:
            return node
        if self.register_attribute is not None:
            self.register_attribute(self.path, node.lineno, node.attr)
        return ast.copy_location(ast.Call(
            func=ast.Name(id=ATTRIBUTE_HELPER, ctx=ast.Load()),
            args=[node.value, ast.Constant(value=node.attr), ast.Constant(value=self.path), ast.Constant(value=node.lineno)],
            keywords=[],
        ), node)


class WitnessLoader(SourceFileLoader):
    def __init__(self, fullname, filename, relative_path, finder):
        super().__init__(fullname, filename)
        self.relative_path = relative_path
        self.finder = finder

    def exec_module(self, module):
        module.__dict__.update(self.finder.helpers)
        super().exec_module(module)

    def get_code(self, fullname):
        source = self.get_data(self.path)
        tree = compile(source, self.path, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        if self.finder.extra_transformer is not None:
            tree = self.finder.extra_transformer(self.relative_path, tree)
        tree = Instrument(
            self.relative_path, self.finder.audit, self.finder.comparisons, self.finder.attribute_names
        ).visit(tree)
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)


class WitnessFinder(MetaPathFinder):
    def __init__(self, root, audit, comparisons, attribute_names, extra_transformer, extra_helpers):
        self.root = root.resolve()
        self.audit = audit
        self.comparisons = comparisons
        self.attribute_names = attribute_names
        self.extra_transformer = extra_transformer
        self.helpers = dict(extra_helpers)
        if comparisons:
            self.helpers[COMPARISON_HELPER] = lambda *args: compare_with_witness(audit, *args)
        if attribute_names:
            self.helpers[ATTRIBUTE_HELPER] = lambda *args: reveal_attribute(audit, *args)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, SourceFileLoader):
            return None
        filename = Path(spec.origin).resolve()
        try:
            relative = filename.relative_to(self.root).as_posix()
        except ValueError:
            return None
        if not relative.startswith("sqlglot/"):
            return None
        spec.loader = WitnessLoader(fullname, spec.origin, relative, self)
        return spec


@contextmanager
def trace_witnesses(
    source: Path,
    audit,
    *,
    comparisons: bool = True,
    attribute_names: Iterable[str] = (),
    extra_transformer: Callable[[str, ast.Module], ast.Module] | None = None,
    extra_helpers: Mapping[str, Callable] | None = None,
):
    """Instrument imported SQLGlot source from the pinned checkout only."""
    finder = WitnessFinder(
        Path(source), audit, comparisons, frozenset(attribute_names), extra_transformer, extra_helpers or {}
    )
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


@dataclass
class ComparisonEvidence:
    """Store sampled pinned operands; any shared observation prevents narrowing."""

    by_site: dict[Site, dict[str, set[Sample] | None]] = field(default_factory=dict)
    shared: set[Site] = field(default_factory=set)

    def register(self, path: str, line: int, column: int, op: str) -> None:
        self.by_site.setdefault((path, line, column, op), {})

    def observe(self, node_id: str | None, path: str, line: int, column: int, op: str, sample: Sample | None) -> None:
        site = (path, line, column, op)
        records = self.by_site.setdefault(site, {})
        if node_id is None:
            self.shared.add(site)
        else:
            self._record(records, node_id, sample)

    @staticmethod
    def _record(records, node, sample):
        samples = records.setdefault(node, set())
        if samples is None:
            return
        if sample is None:
            records[node] = None
        else:
            samples.add(sample)
            if len(samples) > MAX_DISTINCT_SAMPLES:
                records[node] = None

    def export(self) -> list:
        return [
            [*site, site in self.shared, {node: sorted(samples) if samples else None for node, samples in records.items()}]
            for site, records in sorted(self.by_site.items())
        ]

    def merge(self, rows: Iterable[Sequence]) -> None:
        for path, line, column, op, shared, records in rows:
            site = (path, line, column, op)
            combined = self.by_site.setdefault(site, {})
            if shared:
                self.shared.add(site)
            for node, samples in records.items():
                if samples is None:
                    self._record(combined, node, None)
                else:
                    for left, right, identity in samples:
                        self._record(combined, node, (left, right, identity))


def decode_scalar(atom: str):
    tag, value = atom[0], atom[1:]
    if tag == "n":
        return None
    if tag == "e":
        return Ellipsis
    if tag == "x":
        return NotImplemented
    if tag == "i":
        return int(value)
    if tag == "b":
        return value == "1"
    if tag == "f":
        return float.fromhex(value)
    if tag == "c":
        real, imaginary = value.split(",", 1)
        return complex(float.fromhex(real), float.fromhex(imaginary))
    if tag == "y":
        return bytes.fromhex(value)
    if tag in {"s", "u"}:
        return value
    raise ValueError(f"Unknown comparison atom: {tag}")


def confirms_equivalence(sample: Sample, old_op: str) -> bool:
    """Check the received identity/equality replacement against raw observations."""
    left, right, identity = sample
    tags = {left[0], right[0]}
    # A trusted str subclass is safe only paired with the same str slot.
    if "u" in tags and not tags <= {"s", "u"}:
        return False
    if "o" in tags:
        equality = identity
        inequality = not identity
    else:
        first, second = decode_scalar(left), decode_scalar(right)
        equality = first == second
        inequality = first != second
    if old_op in {"Is", "Eq"}:
        return identity is equality
    return (not identity) is inequality


def _edited_lines(lines: Sequence[str], edits: Iterable[Edit]) -> list[str] | None:
    updated = list(lines)
    previous_end = 0
    for edit in edits:
        start = edit.start - 1
        end = start + len(edit.removed)
        if start < previous_end or end > len(lines) or len(edit.removed) != len(edit.added):
            return None
        if updated[start:end] != edit.removed:
            return None
        updated[start:end] = edit.added
        previous_end = end
    return updated


def _mask_tokens(text: str, comparisons: list[tuple[ast.Compare, str]]) -> bytes | None:
    raw = text.encode("utf-8")
    offsets = [0, *(match.end() for match in re.finditer(b"\n", raw))]
    masks = []
    for expression, op in comparisons:
        left, right = expression.left, expression.comparators[0]
        start = offsets[left.end_lineno - 1] + left.end_col_offset
        end = offsets[right.lineno - 1] + right.col_offset
        hits = list(OPERATOR_TOKEN[op].finditer(raw, start, end))
        if len(hits) != 1:
            return None
        masks.append(hits[0].span())
    for start, end in sorted(masks, reverse=True):
        raw = raw[:start] + b"<comparison_operator>" + raw[end:]
    return raw


def comparison_changes(name: str, lines: Sequence[str], edits: Iterable[Edit]) -> tuple[Site, ...] | None:
    """Accept only is/equality changes with otherwise unchanged source bytes."""
    edits = tuple(edits)
    removed = "\n".join(line for edit in edits for line in edit.removed).encode("utf-8", errors="surrogatepass")
    added = "\n".join(line for edit in edits for line in edit.added).encode("utf-8", errors="surrogatepass")
    if not any(
        OPERATOR_TOKEN[old].search(removed) and OPERATOR_TOKEN[new].search(added)
        for old, new in ALTERNATIVES.items()
    ):
        return None
    patched = _edited_lines(lines, edits)
    if patched is None:
        return None
    original = "\n".join(lines) + "\n"
    modified = "\n".join(patched) + "\n"
    try:
        before = ast.parse(original, name, type_comments=True)
        after = ast.parse(modified, name, type_comments=True)
        sites = []
        old_ops, new_ops = [], []
        for old, new in zip(ast.walk(before), ast.walk(after)):
            if not isinstance(old, ast.Compare) or not isinstance(new, ast.Compare):
                continue
            old_op = comparison_operation(old)
            new_op = comparison_operation(new)
            if old_op is None or new_op != ALTERNATIVES[old_op]:
                continue
            sites.append((name, old.lineno, old.col_offset, old_op))
            old_ops.append((old, old_op))
            new_ops.append((new, new_op))
            new.ops[0] = old.ops[0]
        if not sites or ast.dump(before) != ast.dump(after):
            return None
        old_mask = _mask_tokens(original, old_ops)
        if old_mask is None or old_mask != _mask_tokens(modified, new_ops):
            return None
        return tuple(sites)
    except (SyntaxError, ValueError, UnicodeError, RecursionError):
        return None


def comparison_tests(sites: Iterable[Site], candidates: set, evidence: ComparisonEvidence) -> set | None:
    """Keep any candidate without positive equivalence for every changed site."""
    sites = tuple(sites)
    if not sites or any(site in evidence.shared or site not in evidence.by_site for site in sites):
        return None
    proven = set(candidates)
    for site in sites:
        records = evidence.by_site[site]
        proven = {
            node for node in proven
            if (samples := records.get(node)) and all(confirms_equivalence(sample, site[3]) for sample in samples)
        }
    return candidates - proven
