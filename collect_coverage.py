"""Collect per-test SQLGlot line coverage on the pinned, unpatched checkout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import _process_worker
from contextlib import contextmanager
import json
from multiprocessing.process import BaseProcess
from multiprocessing.reduction import ForkingPickler
import os
from pickle import PickleError
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from weakref import WeakSet

import coverage
import pytest

from attribute_witnesses import candidate_attribute_names
from coverage_worker import AuditTrace, WorkerTask, initialize_worker, run_task, trace_observations


class TestCoverage:
    def __init__(self, tracer, numbers, source, directory, witness_fields=(), important=(), witness_mode="full"):
        self.tracer = tracer
        self.numbers = numbers
        self.source = source
        self.directory = Path(directory)
        self.audit = AuditTrace(source, witness_fields, important, witness_mode)
        self.passed = defaultdict(set)
        self.incomplete = set()
        self.traced_pools = WeakSet()
        self.pool_tasks = []

    def pytest_collection_modifyitems(self, items):
        items[:] = [item for item in items if item.nodeid in self.numbers]

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.audit.node_id = item.nodeid
        self.tracer.switch_context(item.nodeid)
        try:
            return (yield)
        finally:
            self.tracer.switch_context("")
            self.audit.node_id = None

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self.audit.in_call = True
        try:
            return (yield)
        finally:
            self.audit.in_call = False

    def pytest_runtest_logreport(self, report):
        if report.passed:
            self.passed[report.nodeid].add(report.when)
        else:
            self.incomplete.add(report.nodeid)

    @contextmanager
    def subprocess_tracing(self):
        original_init = ProcessPoolExecutor.__init__
        original_submit = ProcessPoolExecutor.submit
        original_start = BaseProcess.start

        def init_pool(executor, max_workers=None, mp_context=None, initializer=None, initargs=(), **options):
            if mp_context is not None and mp_context.get_start_method() == "spawn":
                try:
                    payload = bytes(ForkingPickler.dumps((initializer, initargs)))
                except (TypeError, ValueError, AttributeError, PickleError):
                    # Let multiprocessing handle its own validation for an
                    # initializer we cannot wrap. The test remains unknown.
                    pass
                else:
                    original_init(
                        executor, max_workers, mp_context, initialize_worker,
                        (str(self.source), str(self.directory), payload,
                         self.audit.witness_fields, self.audit.important, self.audit.witness_mode), **options
                    )
                    self.traced_pools.add(executor)
                    return
            original_init(executor, max_workers, mp_context, initializer, initargs, **options)

        def submit_task(executor, fn, /, *args, **kwargs):
            if executor not in self.traced_pools:
                self.audit.record_process()
                return original_submit(executor, fn, *args, **kwargs)
            task_id = len(self.pool_tasks)
            # Setup and teardown can belong to a shared pytest fixture, so
            # attribute their worker activity to collection instead.
            node_id = self.audit.node_id if self.audit.in_call else None
            task = WorkerTask(task_id, node_id, fn, args, kwargs)
            self.pool_tasks.append(node_id)
            future = original_submit(executor, run_task, task)

            def note_failure(result):
                if result.cancelled() or result.exception() is not None:
                    if node_id is None:
                        self.audit.unknown_shared = True
                    else:
                        self.audit.unknown_tests.add(node_id)

            future.add_done_callback(note_failure)
            return future

        def start_process(process):
            is_our_worker = (
                getattr(process, "_target", None) is _process_worker
                and len(process._args) > 2
                and process._args[2] is initialize_worker
            )
            if not is_our_worker:
                self.audit.record_process()
            return original_start(process)

        sys.addaudithook(self.audit.audit)
        self.audit.active = True
        try:
            with (
                patch.object(ProcessPoolExecutor, "__init__", init_pool),
                patch.object(ProcessPoolExecutor, "submit", submit_task),
                patch.object(BaseProcess, "start", start_process),
            ):
                yield
        finally:
            self.audit.active = False

    def combine_workers(self):
        exported = set()
        data = self.tracer.get_data()
        for receipt in sorted(self.directory.glob("worker-*.json")):
            try:
                metadata = json.loads(receipt.read_text(encoding="utf-8"))
                path = receipt.with_suffix(".cov")
                if not path.is_file():
                    continue
                worker_data = coverage.CoverageData(basename=str(path))
                worker_data.read()
                data.update(worker_data)
            except (OSError, ValueError, coverage.CoverageException) as error:
                print(f"Ignoring incomplete worker trace {receipt.name}: {error}", file=sys.stderr)
                continue
            task_ids = metadata["tasks"]
            exported.update(task_ids)
            self.audit.merge_fixtures(metadata["fixtures"], metadata["fixture_collection"])
            self.audit.merge_observations(metadata)
            self.audit.unknown_tests.update(metadata["unknown_tests"])
            self.audit.unknown_shared |= metadata["unknown_shared"]

        for task_id, node_id in enumerate(self.pool_tasks):
            if task_id not in exported:
                if node_id is None:
                    self.audit.unknown_shared = True
                else:
                    self.audit.unknown_tests.add(node_id)
        return len(exported)

    def export(self, catalog):
        traced_tasks = self.combine_workers()
        self.incomplete.update(self.audit.unknown_tests)
        if self.audit.unknown_shared:
            self.incomplete.update(self.numbers)
        complete = {
            node_id: str(self.numbers[node_id])
            for node_id, phases in self.passed.items()
            if phases == {"setup", "call", "teardown"} and node_id not in self.incomplete
        }
        measured = {number: {} for number in complete.values()}
        collection = {}
        data = self.tracer.get_data()
        for filename in sorted(data.measured_files()):
            path = Path(filename).relative_to(self.source).as_posix()
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
        print(f"Merged {traced_tasks}/{len(self.pool_tasks)} spawned pool tasks.")
        observations = self.audit.export_observations()
        for key in ("class_uses", "source_reads"):
            observations[key] = {
                name: sorted(self.numbers[node] for node in nodes if node in self.numbers)
                for name, nodes in sorted(observations[key].items())
            }
        return {
            "coverage": measured,
            "collection": collection,
            "unmeasured": unmeasured,
            "test_ids": {str(entry["number"]): entry["id"] for entry in catalog},
            "witness_mode": self.audit.witness_mode,
            "subprocess_coverage": True,
            "fixtures": {
                path: sorted(self.numbers[node] for node in nodes if node in self.numbers)
                for path, nodes in sorted(self.audit.fixtures.items())
            },
            "fixture_collection": sorted(self.audit.fixture_collection),
            **observations,
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
    witness_options = parser.add_mutually_exclusive_group()
    witness_options.add_argument("--witnesses", action="store_true", help="Collect value and class-read witnesses")
    witness_options.add_argument("--comparison-witnesses", action="store_true", help="Trace comparisons for every selected test")
    witness_options.add_argument("--tokenizer-witnesses", action="store_true", help="Trace keys in pinned tokenizer inputs")
    witness_options.add_argument("--mapping-witnesses", action="store_true", help="Trace key access in pinned shared mappings")
    witness_options.add_argument("--generator-witnesses", action="store_true", help="Trace calls to pinned name-generating closures")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    catalog = json.loads(args.test_catalog.read_text())["tests"]
    numbers = {entry["id"]: entry["number"] for entry in catalog}
    witness_fields = candidate_attribute_names(source) if args.witnesses else ()
    important = ()
    if args.comparison_witnesses:
        important = numbers.keys()
    elif args.witnesses:
        important = {entry["id"] for entry in catalog if entry["seconds"] >= 0.1}
    witness_mode = "full"
    if args.comparison_witnesses:
        witness_mode = "comparisons"
    elif args.tokenizer_witnesses:
        witness_mode = "tokenizer"
    elif args.mapping_witnesses:
        witness_mode = "mapping"
    elif args.generator_witnesses:
        witness_mode = "generator"

    os.chdir(source)
    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(source))
    tracer = coverage.Coverage(
        config_file=False, data_file=None, source=[str(source / "sqlglot")]
    )
    with TemporaryDirectory(prefix="sqlglot-worker-coverage-") as workers:
        collector = TestCoverage(tracer, numbers, source, workers, witness_fields, important, witness_mode)
        with tracer.collect(), collector.subprocess_tracing(), trace_observations(collector.audit):
            status = pytest.main(
                ["-q", "--continue-on-collection-errors", "-p", "no:cacheprovider", "tests"],
                plugins=[collector],
            )
        result = collector.export(catalog)
    if not result["coverage"]:
        raise SystemExit("No tests completed with coverage")
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        f"Measured {len(result['coverage'])}/{len(catalog)} tests; "
        f"{len(result['unmeasured'])} require fallback. Pytest exit status: {int(status)}."
    )


if __name__ == "__main__":
    main()
