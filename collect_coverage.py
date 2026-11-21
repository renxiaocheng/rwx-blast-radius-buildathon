"""Collect per-test SQLGlot line coverage on the pinned, unpatched checkout."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import coverage
import pytest


class TestCoverage:
    def __init__(self, tracer, numbers):
        self.tracer = tracer
        self.numbers = numbers
        self.passed = defaultdict(set)
        self.incomplete = set()

    def pytest_collection_modifyitems(self, items):
        items[:] = [item for item in items if item.nodeid in self.numbers]

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.tracer.switch_context(item.nodeid)
        try:
            return (yield)
        finally:
            self.tracer.switch_context("")

    def pytest_runtest_logreport(self, report):
        if report.passed:
            self.passed[report.nodeid].add(report.when)
        else:
            self.incomplete.add(report.nodeid)

    def export(self, source, catalog):
        complete = {
            node_id: str(self.numbers[node_id])
            for node_id, phases in self.passed.items()
            if phases == {"setup", "call", "teardown"} and node_id not in self.incomplete
        }
        measured = {number: {} for number in complete.values()}
        collection = {}
        data = self.tracer.get_data()
        for filename in sorted(data.measured_files()):
            path = Path(filename).relative_to(source).as_posix()
            by_test = defaultdict(list)
            collected_lines = []
            for line, contexts in sorted(data.contexts_by_lineno(filename).items()):
                if "" in contexts:
                    collected_lines.append(line)
                for context in contexts:
                    number = complete.get(context)
                    if number is not None:
                        by_test[number].append(line)
            for number, lines in by_test.items():
                measured[number][path] = contiguous_ranges(lines)
            if collected_lines:
                collection[path] = contiguous_ranges(collected_lines)
        unmeasured = [entry["number"] for entry in catalog if str(entry["number"]) not in measured]
        return {
            "coverage": measured,
            "collection": collection,
            "unmeasured": unmeasured,
            "test_ids": {str(entry["number"]): entry["id"] for entry in catalog},
        }


def contiguous_ranges(lines):
    ranges = []
    for line in lines:
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1][1] = line
        else:
            ranges.append([line, line])
    return [start if start == end else [start, end] for start, end in ranges]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned SQLGlot checkout")
    parser.add_argument("--test-catalog", type=Path, required=True, help="Official tests.json")
    parser.add_argument("--output", type=Path, required=True, help="Per-test coverage JSON path")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    catalog = json.loads(args.test_catalog.read_text())["tests"]
    numbers = {entry["id"]: entry["number"] for entry in catalog}

    os.chdir(source)
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(source))
    tracer = coverage.Coverage(
        config_file=False, data_file=None, source=[str(source / "sqlglot")]
    )
    collector = TestCoverage(tracer, numbers)
    with tracer.collect():
        status = pytest.main(
            ["-q", "--continue-on-collection-errors", "-p", "no:cacheprovider", "tests"],
            plugins=[collector],
        )
    result = collector.export(source, catalog)
    if not result["coverage"]:
        raise SystemExit("No tests completed with coverage")
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        f"Measured {len(result['coverage'])}/{len(catalog)} tests; "
        f"{len(result['unmeasured'])} require fallback. Pytest exit status: {int(status)}."
    )


if __name__ == "__main__":
    main()
