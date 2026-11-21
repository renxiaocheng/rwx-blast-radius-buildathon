"""Select SQLGlot tests from a coverage map collected at the pinned revision."""

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path, PurePosixPath
import re
from types import CodeType
from typing import Iterable

from attribute_witnesses import AttributeWitnesses
from branch_witnesses import BranchEvidence, branch_changes, branch_tests
from comparison_witnesses import ComparisonEvidence, comparison_changes, comparison_tests, unpack_comparison_row
from condition_witnesses import equivalent_for, replacements
from dialect_impact import DialectImpact
from generator_impact import GeneratorImpact
from mapping_witnesses import MappingWitnesses
from patch_impact import statement_impact
from position_witnesses import PositionWitnesses
from source_impact import changed_test_methods, compiled_equivalent, expression_arg_types_impact
from tokenizer_impact import TokenizerImpact


HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$")
SOURCE_POSITION_NAMES = {
    "__code__", "__firstlineno__", "__traceback__", "co_code", "co_consts", "co_filename", "co_firstlineno",
    "co_linetable", "co_lnotab", "co_lines", "co_positions", "disassemble", "f_lineno", "getframeinfo",
    "getsource", "getsourcefile", "getsourcelines", "linecache", "tb_lineno", "traceback",
}


@dataclass
class Edit:
    start: int
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FunctionBody:
    start: int
    end: int
    indent: int


@dataclass(frozen=True)
class SourceFile:
    lines: list[str]
    bodies: list[FunctionBody]
    code: CodeType


@dataclass
class CoverageIndex:
    lines: dict[str, dict[int, set[int]]]
    unmeasured: set[int]
    collection: dict[str, set[int]]
    fixtures: dict[str, set[int]]
    shared_fixtures: set[str]
    subprocess_coverage: bool
    class_uses: dict[str, set[int]]
    shared_class_uses: set[str]
    source_reads: dict[str, set[int]]
    shared_source_reads: set[str]
    observes_source: bool
    dedicated_importers: dict[str, set[int]]
    isolated_import_tests: set[int]


class TestSourceIndex:
    def __init__(self, source: Path, catalog: list[dict]):
        by_file = defaultdict(set)
        self.by_method = defaultdict(set)
        for test in catalog:
            by_file[test["file"]].add(test["number"])
            method = test["id"].rsplit("::", 1)[-1].split("[", 1)[0]
            self.by_method[method].add(test["number"])
        self.file_names = []
        for path, numbers in by_file.items():
            try:
                text = (source / path).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                self.file_names.append((None, numbers))
            else:
                self.file_names.append((set(re.findall(r"\b[A-Za-z_]\w*\b", text)), numbers))

    def mentioning(self, names: set[str]) -> set[int]:
        return set().union(*(
            numbers for words, numbers in self.file_names if words is None or not words.isdisjoint(names)
        ))

    def named_methods(self, names: set[str]) -> set[int] | None:
        # A catalog item may inherit a test method from another module.
        if any(name not in self.by_method for name in names):
            return None
        return set().union(*(self.by_method[name] for name in names))


class ExecutionWitnesses:
    def __init__(self, path: Path, source: Path, numbers_by_id: dict[str, int]):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        self.numbers_by_id = numbers_by_id
        self.ids_by_number = {number: node for node, number in numbers_by_id.items()}
        self.reliable = set(data["reliable"])
        self.unobserved = {
            number for node, number in numbers_by_id.items() if node not in self.reliable
        }
        self.comparison_rows = defaultdict(list)
        for row in data.get("comparison_witnesses", ()):
            self.comparison_rows[tuple(row[:4])].append(row)
        self.comparison_ids = data.get("comparison_node_ids", ())
        self.comparison_complete = frozenset(data.get("comparison_complete", ()))
        self.conditions = data.get("condition_witnesses", {}).get("sites", {})
        self.branches = None
        if "branch_witnesses" in data:
            self.branches = BranchEvidence()
            self.branches.merge(data["branch_witnesses"])
            self.branch_files = {site.split(":", 1)[0] for site in self.branches.sites}
            self.branch_complete = frozenset(data.get("branch_complete", ()))
        self.comparison_files = {site[0] for site in self.comparison_rows}
        self.condition_files = {site.split(":", 1)[0] for site in self.conditions}
        self.attributes = None
        if "attribute_witnesses" in data:
            self.attributes = AttributeWitnesses(source, data["attribute_witnesses"], numbers_by_id)
        self.tokenizer = None
        if "tokenizer_witnesses" in data:
            self.tokenizer = TokenizerImpact(source, data["tokenizer_witnesses"], numbers_by_id)
            complete = set(data.get("tokenizer_complete", ()))
            self.tokenizer_unobserved = {number for node, number in numbers_by_id.items() if node not in complete}
        self.mapping = None
        if "mapping_witnesses" in data:
            self.mapping = MappingWitnesses(
                source, data["mapping_witnesses"], numbers_by_id, namespaces=data.get("module_mapping_witnesses"),
            )
        self.generator = None
        if "generator_witnesses" in data:
            self.generator = GeneratorImpact(source, data["generator_witnesses"], numbers_by_id)
            self.generator_complete = frozenset(data.get("generator_complete", ()))
        self.positions = None
        if "position_witnesses" in data:
            self.positions = PositionWitnesses(source, data["position_witnesses"], numbers_by_id)

    def expression_tests(self, name, lines, edits, candidates, source_readers, *, single_file=False):
        eligible = {self.ids_by_number[number] for number in candidates - source_readers - self.unobserved}
        if not eligible:
            return candidates
        sites = comparison_changes(name, lines, edits) if name in self.comparison_files else None
        if sites:
            evidence = ComparisonEvidence()
            evidence.merge(
                unpack_comparison_row(row, self.comparison_ids)
                for site in sites for row in self.comparison_rows.get(site, ())
            )
            required = comparison_tests(sites, eligible, evidence, self.comparison_complete)
            if required is not None and required != eligible:
                return candidates - {self.numbers_by_id[node] for node in eligible - required}
        changes = replacements(name, lines, edits) if name in self.condition_files else ()
        if changes and all(change.site in self.conditions for change in changes):
            proven = {
                node for node in eligible
                if all(equivalent_for(change, self.conditions[change.site].get(node)) for change in changes)
            }
            candidates = candidates - {self.numbers_by_id[node] for node in proven}
            eligible -= proven
        if single_file and self.branches is not None and name in self.branch_files:
            eligible &= self.branch_complete
            changes = branch_changes(name, lines, edits) if eligible else ()
            if changes:
                required = branch_tests(changes, eligible, self.branches, complete=self.branch_complete)
                if required is not None and required != eligible:
                    candidates = (candidates - {self.numbers_by_id[node] for node in eligible - required}) | source_readers
        return candidates


def parse_patch(patch: str) -> list[tuple[str, list[Edit]]] | None:
    """Return changed blocks at old-file line positions for ordinary Git diffs."""
    rows = patch.split("\n")
    files = []
    cursor = 0
    while cursor < len(rows) and rows[cursor].startswith("diff --git "):
        cursor += 1
        if cursor < len(rows) and rows[cursor].startswith("index "):
            cursor += 1
        if cursor + 1 >= len(rows):
            return None

        old, new = rows[cursor : cursor + 2]
        if not old.startswith("--- a/") or not new.startswith("+++ b/"):
            return None
        path = old[6:]
        if path != new[6:] or not path or ".." in PurePosixPath(path).parts:
            return None
        cursor += 2

        edits = []
        while cursor < len(rows) and rows[cursor].startswith("@@ "):
            header = HUNK_HEADER.fullmatch(rows[cursor])
            if header is None:
                return None
            old_line = int(header[1])
            old_remaining = int(header[2] or "1")
            new_remaining = int(header[4] or "1")
            if old_remaining == 0:
                old_line += 1
            cursor += 1

            active_edit = None
            while old_remaining or new_remaining:
                if cursor >= len(rows) or not rows[cursor]:
                    return None
                row = rows[cursor]
                cursor += 1
                if row == "\\ No newline at end of file":
                    continue

                prefix = row[0]
                if prefix not in " +-":
                    return None
                if prefix == " ":
                    active_edit = None
                else:
                    if active_edit is None:
                        active_edit = Edit(old_line)
                        edits.append(active_edit)
                    changed_lines = active_edit.removed if prefix == "-" else active_edit.added
                    changed_lines.append(row[1:])
                if prefix != "+":
                    old_line += 1
                    old_remaining -= 1
                if prefix != "-":
                    new_remaining -= 1
                if old_remaining < 0 or new_remaining < 0:
                    return None

            if cursor < len(rows) and rows[cursor] == "\\ No newline at end of file":
                cursor += 1

        if not edits:
            return None
        files.append((path, edits))

    if any(rows[cursor:]):
        return None
    return files or None


def covered_lines(spans: Iterable[int | list[int]]) -> Iterable[int]:
    for span in spans:
        if isinstance(span, int):
            yield span
        else:
            yield from range(span[0], span[1] + 1)


def read_coverage(path: Path, numbers_by_id: dict[str, int]) -> CoverageIndex:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        coverage = json.load(stream)
    test_numbers = set(numbers_by_id.values())
    # Old local maps use catalog numbers. Bundled maps carry the matching
    # node IDs so a renumbered scoring catalog still selects the right tests.
    if "test_ids" in coverage:
        number_map = {
            number: numbers_by_id[node_id]
            for number, node_id in coverage["test_ids"].items()
            if node_id in numbers_by_id
        }
    else:
        number_map = {str(number): number for number in test_numbers}
    by_line = defaultdict(lambda: defaultdict(set))
    measured_tests = set()
    if "line_test_sets" in coverage:
        mapped = {int(number): target for number, target in number_map.items()}
        measured_tests.update(mapped[number] for number in coverage["measured"] if number in mapped)
        groups = [{mapped[number] for number in group if number in mapped} for group in coverage["line_test_sets"]]
        by_line.update({
            filename: {line: groups[group] for group, spans in entries for line in covered_lines(spans)}
            for filename, entries in coverage["line_index"].items()
        })
    else:
        for number_text, files in coverage["coverage"].items():
            number = number_map.get(number_text)
            if number is None:
                continue
            measured_tests.add(number)
            for filename, spans in files.items():
                for line in covered_lines(spans):
                    by_line[filename][line].add(number)

    unknown = {
        number_map[str(number)] for number in coverage.get("unmeasured", []) if str(number) in number_map
    }
    unmeasured_tests = (test_numbers - measured_tests) | unknown
    collection = {
        name: set(covered_lines(spans)) for name, spans in coverage.get("collection", {}).items()
    }
    def remap(key):
        return {
            name: {number_map[str(number)] for number in numbers if str(number) in number_map}
            for name, numbers in coverage.get(key, {}).items()
        }
    recorded_import = coverage.get("lazy_import_evidence", {})
    import_test = numbers_by_id.get(recorded_import.get("test_id"))
    dedicated_importers = {
        name: {import_test} for name in recorded_import.get("imported_files", ())
    } if import_test is not None else {}
    isolated_import_tests = set()
    if (
        import_test is not None and dedicated_importers
        and recorded_import.get("source_pin") == "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
        and recorded_import.get("class_attribute_reads") == [["sqlglot.dialects.dialect.Dialect", "_classes"]]
        and recorded_import.get("source_reads") == []
    ):
        isolated_import_tests.add(import_test)
    return CoverageIndex(
        by_line,
        unmeasured_tests,
        collection,
        remap("fixtures"),
        set(coverage.get("fixture_collection", [])),
        coverage.get("subprocess_coverage") is True,
        remap("class_uses"),
        set(coverage.get("shared_class_uses", [])),
        remap("source_reads"),
        set(coverage.get("shared_source_reads", [])),
        "source_reads" in coverage,
        dedicated_importers,
        isolated_import_tests,
    )


def read_source(path: Path) -> SourceFile:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    bodies = [
        FunctionBody(node.body[0].lineno, node.end_lineno, node.body[0].col_offset)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    bodies.sort(key=lambda body: body.end - body.start)
    return SourceFile(source.splitlines(), bodies, compile(tree, str(path), "exec", dont_inherit=True))


def compiled_patch(name: str, lines: list[str], edits: list[Edit]) -> tuple[list[str], CodeType] | None:
    patched = lines.copy()
    previous_start = len(lines) + 1
    for edit in reversed(edits):
        start = edit.start - 1
        end = start + len(edit.removed)
        if start < 0 or end > len(lines) or end > previous_start:
            return None
        if lines[start:end] != edit.removed:
            return None
        patched[start:end] = edit.added
        previous_start = start
    try:
        return patched, compile("\n".join(patched) + "\n", name, "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, OverflowError):
        return None


def uses_subprocesses(path: Path) -> bool:
    """Treat a test module as unmeasured if it can execute code in child processes."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError):
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.partition(".")[0] in {"multiprocessing", "subprocess"}
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.partition(".")[0] in {"multiprocessing", "subprocess"}:
                return True
            if module == "concurrent.futures" and any(
                alias.name == "ProcessPoolExecutor" for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Attribute) and node.attr == "ProcessPoolExecutor":
            return True
    return False


def impacted_tests(
    edit: Edit,
    by_line: dict[int, set[int]],
    bodies: list[FunctionBody],
    collection_lines: set[int],
) -> set[int]:
    if edit.removed:
        first, last = edit.start, edit.start + len(edit.removed) - 1
    else:
        # An added line has no old coverage. Require it to sit inside a
        # function on both sides, and use the coverage of the whole body.
        first, last = edit.start - 1, edit.start

    body = next((body for body in bodies if body.start <= first <= last <= body.end), None)
    if body is None:
        return set()
    for line in edit.added:
        if line.strip() and len(line) - len(line.lstrip()) < body.indent:
            return set()
    body_lines = range(body.start, body.end + 1)
    if any(line in collection_lines for line in body_lines):
        return set()
    # A changed branch can also affect callers that did not execute that
    # particular line in the pinned revision.
    return set().union(*(by_line.get(line, ()) for line in body_lines))


def statement_tests(name, lines, edits, by_line, collection):
    spans = statement_impact(name, lines, edits)
    if spans is None or any(line in collection for span in spans for line in span):
        return None
    selected = set()
    for span in spans:
        tests = set().union(*(by_line.get(line, ()) for line in span))
        if not tests:
            return None
        selected.update(tests)
    return selected


def class_tests(
    classes: set[str], coverage: CoverageIndex, tests: TestSourceIndex, source: Path
) -> set[int] | None:
    if not classes or classes & coverage.shared_class_uses:
        return None
    if any(not coverage.class_uses.get(name) for name in classes):
        return None
    names = {name.rsplit(".", 1)[-1] for name in classes}
    selection = tests.mentioning(names).union(*(coverage.class_uses[name] for name in classes))
    names_pattern = "|".join(map(re.escape, sorted(names)))
    # A library can read class metadata before instantiating the class. Map
    # explicit reads back to their executable lines and defer if one is shared.
    access = re.compile(rf"\b(?:{names_pattern})\s*\.\s*(?:arg_types|required_args)\b")
    for filename in coverage.lines.keys() | coverage.collection.keys():
        if not filename.startswith("sqlglot/") or not filename.endswith(".py"):
            continue
        try:
            contents = (source / filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        for read in access.finditer(contents):
            start = contents.count("\n", 0, read.start()) + 1
            stop = contents.count("\n", 0, read.end()) + 1
            for line in range(start, stop + 1):
                if line in coverage.collection.get(filename, ()):
                    return None
                selection.update(coverage.lines.get(filename, {}).get(line, ()))
    return selection


def select_tests(
    patch: str,
    coverage: CoverageIndex,
    source: Path,
    source_cache: dict[str, SourceFile | None],
    dialects: DialectImpact | None = None,
    test_sources: TestSourceIndex | None = None,
    witnesses: ExecutionWitnesses | None = None,
) -> set[int] | None:
    files = parse_patch(patch)
    if files is None:
        return None

    selected = set(coverage.unmeasured)
    for name, edits in files:
        if name.startswith("tests/fixtures/") and name.endswith(".sql"):
            consumers = coverage.fixtures.get(name)
            # Fixtures opened during setup or collection can be shared by tests
            # that never open the file themselves.
            if not consumers or name in coverage.shared_fixtures:
                return None
            selected.update(consumers)
            continue
        if not name.startswith(("sqlglot/", "tests/")) or not name.endswith(".py"):
            return None
        if name not in source_cache:
            try:
                source_cache[name] = read_source(source / name)
            except (OSError, SyntaxError, UnicodeError):
                source_cache[name] = None
        original = source_cache[name]
        if original is None or (compiled := compiled_patch(name, original.lines, edits)) is None:
            return None
        if name.startswith("tests/"):
            methods = changed_test_methods(name, original.lines, edits)
            if not methods or test_sources is None:
                return None
            method_tests = test_sources.named_methods(methods)
            if method_tests is None:
                return None
            selected.update(method_tests)
            continue
        patched, code = compiled
        position_readers = set() if test_sources is None else test_sources.mentioning(SOURCE_POSITION_NAMES)
        if len(files) == 1 and witnesses is not None and witnesses.positions is not None:
            position_readers = witnesses.positions.refine(name, position_readers)
        if coverage.observes_source and test_sources is not None and compiled_equivalent(
            original.lines, patched, edits, original.code, code,
        ):
            if name in coverage.shared_source_reads:
                return None
            selected.update(coverage.source_reads.get(name, set()))
            selected.update(position_readers)
            continue
        if name not in coverage.lines:
            return None
        source_readers = None
        if witnesses is not None and coverage.observes_source and test_sources is not None:
            if name not in coverage.shared_source_reads:
                source_readers = coverage.source_reads.get(name, set()) | position_readers
        collection_lines = coverage.collection.get(name, set())
        narrowed = statement_tests(name, original.lines, edits, coverage.lines[name], collection_lines)
        if narrowed:
            if source_readers is not None:
                narrowed = source_readers | witnesses.expression_tests(
                    name, original.lines, edits, narrowed, source_readers, single_file=len(files) == 1,
                )
            selected.update(narrowed)
            continue
        file_tests = set()
        for edit in edits:
            tests = impacted_tests(edit, coverage.lines[name], original.bodies, collection_lines)
            if not tests:
                break
            file_tests.update(tests)
        else:
            if source_readers is not None:
                file_tests = witnesses.expression_tests(
                    name, original.lines, edits, file_tests, source_readers, single_file=len(files) == 1,
                )
            selected.update(file_tests)
            continue
        # The isolated import test may inspect class data created by module code.
        selected.update(coverage.dedicated_importers.get(name, ()))
        if len(files) == 1 and source_readers is not None and witnesses.tokenizer is not None:
            readers = witnesses.tokenizer.select(name, original.lines, edits, mentions=test_sources.mentioning)
            if readers is not None:
                selected.update(readers, witnesses.tokenizer_unobserved - coverage.isolated_import_tests, source_readers)
                continue
        if len(files) == 1 and source_readers is not None and witnesses.mapping is not None:
            readers = witnesses.mapping.select_patch(
                name, original.lines, edits, attributes=witnesses.attributes, mentions=test_sources.mentioning,
            )
            if readers is not None:
                selected.update(readers, witnesses.mapping.unobserved - coverage.isolated_import_tests, source_readers)
                continue
        if len(files) == 1 and source_readers is not None and witnesses.generator is not None:
            observed_importers = {witnesses.ids_by_number[number] for number in coverage.isolated_import_tests}
            readers = witnesses.generator.select(
                name, original.lines, edits, complete=witnesses.generator_complete | observed_importers,
                mentions=test_sources.mentioning,
            )
            if readers is not None:
                selected.update(readers, source_readers)
                continue
        if source_readers is not None and witnesses.attributes is not None and dialects is not None:
            reads = witnesses.attributes.select_patch(
                name, original.lines, edits, dialect_targets=dialects.targets, mentions=test_sources.mentioning,
            )
            if reads is not None:
                selected.update(reads, witnesses.unobserved, source_readers)
                continue
        classes = expression_arg_types_impact(name, original.lines, edits)
        if classes and test_sources is not None:
            expression_tests = class_tests(classes, coverage, test_sources, source)
            if expression_tests is not None:
                selected.update(expression_tests)
                continue
        if dialects is not None:
            related = dialects.select(
                name, original.lines, edits,
                class_uses=coverage.class_uses, shared_class_uses=coverage.shared_class_uses,
            )
            if related is not None:
                selected.update(related)
                continue
        return None
    return selected


def schedule(tests: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Balance work using the scorer's durations and per-worker overhead."""
    ordered = sorted(tests, key=lambda test: (-test[1], test[0]))
    if not ordered:
        return []
    total = sum(milliseconds for _, milliseconds in ordered)
    largest = ordered[0][1]
    best = []
    best_cost = float("inf")
    for count in range(1, min(len(ordered), 8) + 1):
        overhead = 2000 * count
        if largest + overhead >= best_cost:
            break
        if (total + count - 1) // count + overhead >= best_cost:
            continue
        workers = [[] for _ in range(count)]
        loads = [0] * count
        for number, milliseconds in ordered:
            worker = min(range(count), key=lambda i: (loads[i], len(workers[i])))
            workers[worker].append(number)
            loads[worker] += milliseconds

        cost = max(loads) + overhead
        if cost < best_cost:
            best, best_cost = workers, cost
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--coverage", type=Path, required=True, help="Coverage map for the official test numbers")
    parser.add_argument("--source", type=Path, required=True, help="Checkout of the pinned SQLGlot revision")
    parser.add_argument("--witnesses", type=Path, help="Runtime observations from the pinned SQLGlot revision")
    parser.add_argument("--dialect-index", type=Path, help="Pinned source imports and test references")
    args = parser.parse_args()
    if not args.coverage.is_file():
        parser.error(f"coverage map not found: {args.coverage}")
    if not (args.source / "sqlglot").is_dir():
        parser.error(f"SQLGlot source directory not found: {args.source / 'sqlglot'}")

    catalog = json.loads((args.data / "tests.json").read_text(encoding="utf-8"))["tests"]
    durations = {test["number"]: round(test["seconds"] * 1000) for test in catalog}
    test_numbers = set(durations)
    numbers_by_id = {test["id"]: test["number"] for test in catalog}
    coverage = read_coverage(args.coverage, numbers_by_id)
    witnesses = ExecutionWitnesses(args.witnesses, args.source, numbers_by_id) if args.witnesses else None
    if not coverage.subprocess_coverage:
        subprocess_files = {
            path for path in {test["file"] for test in catalog} if uses_subprocesses(args.source / path)
        }
        coverage.unmeasured.update(test["number"] for test in catalog if test["file"] in subprocess_files)

    source_cache: dict[str, SourceFile | None] = {}
    dialects = DialectImpact(args.source, coverage.lines, catalog, index_path=args.dialect_index)
    test_sources = TestSourceIndex(args.source, catalog)
    full_schedule = schedule(durations.items())
    for path in sorted((args.data / "commits").glob("*.patch")):
        patch = path.read_text(encoding="utf-8", errors="surrogateescape")
        selection = select_tests(patch, coverage, args.source, source_cache, dialects, test_sources, witnesses)
        workers = full_schedule
        if selection is not None and selection != test_numbers:
            workers = schedule((number, durations[number]) for number in selection)
        print(json.dumps({"commit": path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
