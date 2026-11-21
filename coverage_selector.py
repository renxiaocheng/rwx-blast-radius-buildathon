"""Select SQLGlot tests from a coverage map collected at the pinned revision."""

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable


HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*$")


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


@dataclass
class CoverageIndex:
    lines: dict[str, dict[int, set[int]]]
    unmeasured: set[int]
    collection: dict[str, set[int]]
    fixtures: dict[str, set[int]]
    shared_fixtures: set[str]
    subprocess_coverage: bool


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
    fixtures = {
        name: {number_map[str(number)] for number in numbers if str(number) in number_map}
        for name, numbers in coverage.get("fixtures", {}).items()
    }
    return CoverageIndex(
        by_line,
        unmeasured_tests,
        collection,
        fixtures,
        set(coverage.get("fixture_collection", [])),
        coverage.get("subprocess_coverage") is True,
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
    return SourceFile(source.splitlines(), bodies)


def patch_compiles(name: str, lines: list[str], edits: list[Edit]) -> bool:
    patched = lines.copy()
    previous_start = len(lines) + 1
    for edit in reversed(edits):
        start = edit.start - 1
        end = start + len(edit.removed)
        if start < 0 or end > len(lines) or end > previous_start:
            return False
        if lines[start:end] != edit.removed:
            return False
        patched[start:end] = edit.added
        previous_start = start
    try:
        compile("\n".join(patched) + "\n", name, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return False
    return True


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


def select_tests(
    patch: str,
    coverage: CoverageIndex,
    source: Path,
    source_cache: dict[str, SourceFile | None],
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
        if not name.startswith("sqlglot/") or not name.endswith(".py") or name not in coverage.lines:
            return None
        if name not in source_cache:
            try:
                source_cache[name] = read_source(source / name)
            except (OSError, SyntaxError, UnicodeError):
                source_cache[name] = None
        original = source_cache[name]
        if original is None or not patch_compiles(name, original.lines, edits):
            return None
        collection_lines = coverage.collection.get(name, set())
        for edit in edits:
            tests = impacted_tests(edit, coverage.lines[name], original.bodies, collection_lines)
            if not tests:
                return None
            selected.update(tests)
    return selected


def schedule(tests: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Balance work using the scorer's durations and per-worker overhead."""
    ordered = sorted(tests, key=lambda test: (-test[1], test[0]))
    best = []
    best_cost = float("inf")
    for count in range(1, min(len(ordered), 8) + 1):
        workers = [[] for _ in range(count)]
        loads = [0] * count
        for number, milliseconds in ordered:
            worker = min(range(count), key=lambda i: (loads[i], len(workers[i])))
            workers[worker].append(number)
            loads[worker] += milliseconds

        cost = max(loads) + 2000 * count
        if cost < best_cost:
            best, best_cost = workers, cost
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--coverage", type=Path, required=True, help="Coverage map for the official test numbers")
    parser.add_argument("--source", type=Path, required=True, help="Checkout of the pinned SQLGlot revision")
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
    if not coverage.subprocess_coverage:
        subprocess_files = {
            path for path in {test["file"] for test in catalog} if uses_subprocesses(args.source / path)
        }
        coverage.unmeasured.update(test["number"] for test in catalog if test["file"] in subprocess_files)

    source_cache: dict[str, SourceFile | None] = {}
    full_schedule = schedule(durations.items())
    for path in sorted((args.data / "commits").glob("*.patch")):
        patch = path.read_text(encoding="utf-8", errors="surrogateescape")
        selection = select_tests(patch, coverage, args.source, source_cache)
        workers = full_schedule
        if selection is not None and selection != test_numbers:
            workers = schedule((number, durations[number]) for number in selection)
        print(json.dumps({"commit": path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
