"""Pinned tokenizer input observations for edits to existing escape entries."""

import ast
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from unittest.mock import patch

from patch_impact import PatchEdit, apply_edits


FIELD = "NUMERIC_ESCAPES"
PREFIX = re.compile(r"(?<=\\).", flags=re.DOTALL)
OCTAL_CHARACTERS = frozenset("01234567")
VERSION = 1

# These pinned consumers establish that unchanged keys and non-emptiness
# preserve the eagerly built generator formats and every generator use. The
# I/O values only flow through Tokenizer._init_core into TokenizerCore. The
# core can consult a value only for a character following a backslash, with
# octal digits normalized to key 0. Reject source that differs from this pin.
CONSUMERS = {
    "sqlglot/tokens.py": "93ea4dd996a6ffb9b61f77cc7900658351ca1d9b1d6cd40efddd233720d8e2be",
    "sqlglot/tokenizer_core.py": "142463b972f35299140172de19eb72bf0c32742867eb251e06c5592413f8083c",
    "sqlglot/dialects/dialect.py": "76e8553773db08405642d8ea65cd680322d94a7f1a94b2efdba4015728116bda",
    "sqlglot/generator.py": "d8d75b37fd8dff73f2a8bb0de075e49e37d2576a6ac0468a9154e70fa8e24735",
}


def escape_keys(sql: str) -> set[str]:
    """Overapproximate escape reads, including overlaps and nonliteral SQL."""
    return {
        "0" if match[0] in OCTAL_CHARACTERS else match[0]
        for match in PREFIX.finditer(sql)
    }


@dataclass
class TokenizerEvidence:
    """Store baseline SQL prefix observations without storing SQL or labels."""

    available: bool = False
    unsupported: bool = False
    owners: dict[str, set[str]] = field(default_factory=dict)
    reads: dict[str, dict[str, set[str]]] = field(default_factory=lambda: defaultdict(dict))
    shared: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    unknown: set[str] = field(default_factory=set)
    unknown_shared: bool = False

    def register(self, owner: str, keys: Iterable[str]) -> None:
        self.owners.setdefault(owner, set()).update(keys)

    def mark_unknown(self, node_id: str | None) -> None:
        if node_id is None:
            self.unknown_shared = True
        else:
            self.unknown.add(node_id)

    def observe(self, owner: str, node_id: str | None, sql: str) -> None:
        if type(sql) is not str or owner not in self.owners:
            self.mark_unknown(node_id)
        elif "\\" in sql:
            self.add_keys(owner, node_id, escape_keys(sql))

    def add_keys(self, owner: str, node_id: str | None, keys: set[str]) -> None:
        observed = keys & self.owners[owner]
        if not observed:
            return
        if node_id is None:
            self.shared[owner].update(observed)
        else:
            self.reads[owner].setdefault(node_id, set()).update(observed)

    def export(self) -> dict:
        return {
            "version": VERSION,
            "available": self.available,
            "unsupported": self.unsupported,
            "owners": {
                owner: [
                    sorted(keys),
                    sorted(self.shared.get(owner, ())),
                    {node: sorted(seen) for node, seen in sorted(self.reads.get(owner, {}).items())},
                ]
                for owner, keys in sorted(self.owners.items())
            },
            "unknown": sorted(self.unknown),
            "unknown_shared": self.unknown_shared,
        }

    def merge(self, data: Mapping | None) -> None:
        if not data or data.get("version") != VERSION or not data.get("available"):
            self.unsupported = True
            return
        self.available = True
        self.unsupported |= data.get("unsupported", False)
        self.unknown_shared |= data.get("unknown_shared", False)
        self.unknown.update(data.get("unknown", ()))
        for owner, (keys, shared, reads) in data.get("owners", {}).items():
            self.register(owner, keys)
            self.shared[owner].update(shared)
            for node, observed in reads.items():
                self.reads[owner].setdefault(node, set()).update(observed)


def class_qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@contextmanager
def trace_tokenizer_inputs(source: Path, audit):
    """Record actual strings entering the pinned Python tokenizer core."""
    from sqlglot import tokenizer_core, tokens

    evidence = audit.tokenizer_inputs
    root = Path(source).resolve()
    if (
        tokens.SQLGLOTC_INSTALLED
        or Path(tokenizer_core.__file__).resolve() != root / "sqlglot/tokenizer_core.py"
        or Path(tokens.__file__).resolve() != root / "sqlglot/tokens.py"
        or tokens.TokenizerCore is not tokenizer_core.TokenizerCore
    ):
        evidence.unsupported = True
        yield
        return

    evidence.available = True
    addresses = {}
    initialize = tokens.Tokenizer._init_core
    tokenize = tokenizer_core.TokenizerCore.tokenize

    def create_core(tokenizer, *args, **kwargs):
        core = initialize(tokenizer, *args, **kwargs)
        if audit.active:
            mapping = core.numeric_escapes
            if type(mapping) is dict and mapping:
                owners = tuple(
                    class_qualified_name(base)
                    for base in type(tokenizer).__mro__
                    if vars(base).get(FIELD) is mapping
                )
                if owners:
                    # Keep the object alive so ids cannot be recycled while
                    # observed. Core calls verify the same mapping identity.
                    previous = addresses.get(id(mapping))
                    if previous is not None:
                        owners = tuple(dict.fromkeys((*previous[1], *owners)))
                    addresses[id(mapping)] = (mapping, owners)
                    for owner in owners:
                        evidence.register(owner, mapping)
        return core

    def read_sql(core, sql, *args, **kwargs):
        if audit.active:
            node = audit.node_id if audit.in_call else None
            mapping = core.numeric_escapes
            if type(sql) is not str or type(mapping) is not dict:
                evidence.mark_unknown(node)
            elif mapping and "\\" in sql:
                known = addresses.get(id(mapping))
                if known is None or known[0] is not mapping:
                    evidence.mark_unknown(node)
                else:
                    keys = escape_keys(sql)
                    for owner in known[1]:
                        evidence.add_keys(owner, node, keys)
        return tokenize(core, sql, *args, **kwargs)

    with (
        patch.object(tokens.Tokenizer, "_init_core", create_core),
        patch.object(tokenizer_core.TokenizerCore, "tokenize", read_sql),
    ):
        yield


@dataclass(frozen=True)
class NumericEscapeEdit:
    owner: str
    keys: frozenset[str]
    original_keys: frozenset[str]


def tokenizer_definitions(tree: ast.Module) -> dict[str, ast.Assign | ast.AnnAssign]:
    # The patch can only modify an existing literal assignment inside an
    # undecorated nested dialect Tokenizer class.
    definitions = {}
    for dialect in tree.body:
        if not isinstance(dialect, ast.ClassDef) or dialect.decorator_list or dialect.keywords:
            continue
        for tokenizer in dialect.body:
            if (
                not isinstance(tokenizer, ast.ClassDef) or tokenizer.name != "Tokenizer"
                or tokenizer.decorator_list or tokenizer.keywords or len(tokenizer.bases) != 1
            ):
                continue
            contents = []
            for stmt in tokenizer.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                elif isinstance(stmt, ast.AnnAssign) and stmt.simple and stmt.value is not None:
                    target = stmt.target
                else:
                    continue
                if isinstance(target, ast.Name) and target.id == FIELD:
                    contents.append(stmt)
            if len(contents) == 1:
                definitions[f"{dialect.name}.{tokenizer.name}"] = contents[0]
    return definitions


def settings(value: ast.AST) -> dict[str, tuple[int, int, int, int]] | None:
    if not isinstance(value, ast.Dict):
        return None
    try:
        payload = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError):
        return None
    if not payload or len(payload) != len(value.keys) or not all(
        type(key) is str and len(key) == 1 and type(spec) is tuple and len(spec) == 4
        and all(type(item) is int for item in spec)
        for key, spec in payload.items()
    ):
        return None
    return payload


def numeric_escape_edit(path: str, lines: Sequence[str], edits: Iterable[PatchEdit]) -> NumericEscapeEdit | None:
    name = PurePosixPath(path)
    if len(name.parts) != 3 or name.parts[:2] != ("sqlglot", "dialects") or name.suffix != ".py":
        return None
    edits = tuple(edits)
    if not edits or any(not edit.removed or len(edit.removed) != len(edit.added) for edit in edits):
        return None
    result = apply_edits(lines, edits)
    if result is None:
        return None
    patched, _, _ = result
    try:
        before = ast.parse("\n".join(lines) + "\n", path, type_comments=True)
        after = ast.parse("\n".join(patched) + "\n", path, type_comments=True)
    except (SyntaxError, ValueError, RecursionError):
        return None
    current = tokenizer_definitions(before)
    modified = tokenizer_definitions(after)
    for name_suffix, original in current.items():
        changed = modified.get(name_suffix)
        if changed is None or any(
            not (original.lineno <= edit.start <= edit.start + len(edit.removed) - 1 <= original.end_lineno)
            for edit in edits
        ):
            continue
        old, new = settings(original.value), settings(changed.value)
        if old is None or new is None or tuple(old) != tuple(new):
            return None
        keys = frozenset(key for key in old if old[key] != new[key])
        if not keys:
            return None
        changed.value = original.value
        if ast.dump(before) != ast.dump(after):
            return None
        module = ".".join(name.with_suffix("").parts)
        return NumericEscapeEdit(f"{module}.{name_suffix}", keys, frozenset(old))
    return None


class TokenizerImpact:
    """Map a new literal value edit to baseline input readers."""

    def __init__(self, source: Path, evidence: TokenizerEvidence | Mapping, numbers_by_id: Mapping[str, int]):
        self.source = Path(source)
        if isinstance(evidence, TokenizerEvidence):
            self.evidence = evidence
        else:
            self.evidence = TokenizerEvidence()
            self.evidence.merge(evidence)
        self.numbers = numbers_by_id
        self.valid_pin = None

    def select(
        self,
        path: str,
        lines: Sequence[str],
        edits: Iterable[PatchEdit],
        *,
        mentions: Callable[[set[str]], set[int]] | None = None,
    ) -> set[int] | None:
        evidence = self.evidence
        if not evidence.available or evidence.unsupported or evidence.unknown_shared:
            return None
        delta = numeric_escape_edit(path, lines, edits)
        if delta is None or not delta.original_keys <= evidence.owners.get(delta.owner, set()):
            return None
        if delta.keys & evidence.shared.get(delta.owner, set()):
            return None
        if self.valid_pin is None:
            try:
                self.valid_pin = all(
                    sha256((self.source / name).read_bytes()).hexdigest() == digest
                    for name, digest in CONSUMERS.items()
                )
            except OSError:
                self.valid_pin = False
        if not self.valid_pin or mentions is None:
            return None
        nodes = evidence.unknown | {
            node for node, observed in evidence.reads.get(delta.owner, {}).items()
            if delta.keys & observed
        }
        selected = {self.numbers[node] for node in nodes if node in self.numbers}
        return selected | mentions({
            FIELD, "numeric_escapes", "TokenizerCore", "_scan_numeric_escape",
            "_read_byte_escape", "_read_numeric_escape",
        })
