"""Spawn-safe Python coverage instrumentation for SQLGlot's test pools."""

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
from multiprocessing.process import BaseProcess
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import Finalize
import os
from pathlib import Path
import sys
from uuid import uuid4

import coverage

from attribute_witnesses import AttributeRecorder, dynamic_instrumentation, mark_preloaded_modules
from branch_witnesses import BranchEvidence, branch_instrumentation
from comparison_witnesses import ComparisonEvidence, trace_witnesses
from condition_witnesses import ConditionRecorder, PROBE
from generator_impact import GeneratorRecorder, generator_instrumentation
from mapping_witnesses import MappingRecorder, mapping_instrumentation
from shared_mapping_witnesses import ModuleMappingRecorder, module_mapping_instrumentation
from tokenizer_impact import TokenizerEvidence, trace_tokenizer_inputs


PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.exec"}


class AuditTrace:
    def __init__(self, source, witness_fields=(), important=(), witness_mode="full"):
        self.source = str(source) + os.sep
        self.witness_fields = tuple(witness_fields)
        self.important = frozenset(important)
        self.witness_mode = witness_mode
        self.comparisons = ComparisonEvidence()
        self.attributes = AttributeRecorder()
        self.conditions = ConditionRecorder(self, important if witness_mode == "full" else ())
        self.tokenizer_inputs = TokenizerEvidence() if witness_mode == "tokenizer" else None
        self.mapping = MappingRecorder() if witness_mode == "mapping" else None
        self.generator = GeneratorRecorder() if witness_mode == "generator" else None
        self.branches = BranchEvidence() if witness_mode == "impact" else None
        self.module_mappings = ModuleMappingRecorder() if witness_mode == "impact" else None
        self.fixture_root = self.source + os.path.join("tests", "fixtures") + os.sep
        self.active = False
        self.node_id = None
        self.in_call = False
        self.fixtures = defaultdict(set)
        self.fixture_collection = set()
        self.source_reads = defaultdict(set)
        self.shared_source_reads = set()
        self.class_uses = defaultdict(set)
        self.shared_class_uses = set()
        self._classes_by_context = defaultdict(set)
        self.unknown_tests = set()
        self.unknown_shared = False

    def current_test(self):
        return self.node_id if self.in_call else None

    def wants_comparison(self):
        return self.active and (not self.in_call or self.node_id in self.important)

    def register_comparison_site(self, path, line, column, op):
        self.comparisons.register(path, line, column, op)

    def record_comparison(self, path, line, column, op, sample):
        if self.wants_comparison():
            self.comparisons.observe(self.current_test(), path, line, column, op, sample)

    def register_attribute_site(self, path, line, field):
        self.attributes.register_site(path, line, field)

    def record_attribute(self, receiver, field, path, line):
        if self.active:
            self.attributes.record(receiver, field, path, line, self.current_test())

    def record_process(self):
        if self.node_id is not None:
            if self.in_call:
                self.unknown_tests.add(self.node_id)
            else:
                self.unknown_shared = True
            return
        # Interpreter and test runner probes can spawn unrelated utilities.
        # A collection-time child launched by the checkout is unknown instead.
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename.startswith(self.source):
                self.unknown_shared = True
                return
            frame = frame.f_back

    def audit(self, event, args):
        if not self.active:
            return
        if event in PROCESS_EVENTS:
            self.record_process()
        elif event == "open" and isinstance(args[0], (str, bytes)):
            candidate = os.path.abspath(os.fsdecode(args[0]))
            if candidate.startswith(self.fixture_root):
                relative = candidate[len(self.source):].replace(os.sep, "/")
                if self.in_call and self.node_id is not None:
                    self.fixtures[relative].add(self.node_id)
                else:
                    self.fixture_collection.add(relative)
            elif candidate.startswith(self.source + "sqlglot" + os.sep) and candidate.endswith(".py"):
                caller = sys._getframe(1)
                if caller.f_code.co_filename == "<frozen importlib._bootstrap_external>":
                    return
                relative = candidate[len(self.source):].replace(os.sep, "/")
                if self.in_call and self.node_id is not None:
                    self.source_reads[relative].add(self.node_id)
                else:
                    self.shared_source_reads.add(relative)

    def merge_fixtures(self, fixtures, collection):
        for name, nodes in fixtures.items():
            self.fixtures[name].update(nodes)
        self.fixture_collection.update(collection)

    def record_class(self, instance):
        if self.active:
            test = self.node_id if self.in_call else None
            self._classes_by_context[test].add(type(instance))

    def export_observations(self):
        for context, classes in self._classes_by_context.items():
            for cls in classes:
                for base in cls.__mro__:
                    if not base.__module__.startswith("sqlglot."):
                        continue
                    name = f"{base.__module__}.{base.__qualname__}"
                    if context is None:
                        self.shared_class_uses.add(name)
                    else:
                        self.class_uses[name].add(context)
        observations = {
            "class_uses": {name: sorted(nodes) for name, nodes in self.class_uses.items()},
            "shared_class_uses": sorted(self.shared_class_uses),
            "source_reads": {path: sorted(nodes) for path, nodes in self.source_reads.items()},
            "shared_source_reads": sorted(self.shared_source_reads),
            "comparison_witnesses": self.comparisons.export(),
            "attribute_witnesses": self.attributes.export(),
            "condition_witnesses": self.conditions.export(),
        }
        if self.tokenizer_inputs is not None:
            observations["tokenizer_witnesses"] = self.tokenizer_inputs.export()
        if self.mapping is not None:
            observations["mapping_witnesses"] = self.mapping.export()
        if self.generator is not None:
            observations["generator_witnesses"] = self.generator.export()
        if self.branches is not None:
            observations["branch_witnesses"] = self.branches.export()
            observations["module_mapping_witnesses"] = self.module_mappings.export()
        return observations

    def merge_observations(self, data):
        for name, nodes in data.get("class_uses", {}).items():
            self.class_uses[name].update(nodes)
        self.shared_class_uses.update(data.get("shared_class_uses", []))
        for name, nodes in data.get("source_reads", {}).items():
            self.source_reads[name].update(nodes)
        self.shared_source_reads.update(data.get("shared_source_reads", []))
        self.comparisons.merge(data.get("comparison_witnesses", ()))
        if "attribute_witnesses" in data:
            self.attributes.merge(data["attribute_witnesses"])
        self.conditions.merge(data.get("condition_witnesses", {}))
        if self.tokenizer_inputs is not None:
            self.tokenizer_inputs.merge(data.get("tokenizer_witnesses"))
        if self.mapping is not None:
            self.mapping.merge(data["mapping_witnesses"])
        if self.generator is not None:
            self.generator.merge(data["generator_witnesses"])
        if self.branches is not None:
            self.branches.merge(data["branch_witnesses"])
            self.module_mappings.merge(data["module_mapping_witnesses"])


def impact_instrumentation(source, audit):
    paths = {"sqlglot/parser.py", "sqlglot/tokenizer_core.py", "sqlglot/generator.py", "sqlglot/expressions/core.py"}
    for package in ("optimizer", "executor"):
        paths.update(path.relative_to(source).as_posix() for path in (source / "sqlglot" / package).glob("*.py"))
    branch_transform, branch_helpers = branch_instrumentation(
        source, audit.branches, audit.current_test, paths=paths,
    )
    namespace_transform, namespace_helpers = module_mapping_instrumentation(source, audit.module_mappings, audit.current_test)

    def transform(path, tree):
        return namespace_transform(path, branch_transform(path, tree))

    return transform, branch_helpers | namespace_helpers


@contextmanager
def trace_observations(audit):
    source = Path(audit.source)
    if audit.witness_mode in {"mapping", "generator", "impact"}:
        if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
            raise RuntimeError("Load the key and dispatch tracer before SQLGlot")
        if audit.mapping is not None:
            transform, helpers = mapping_instrumentation(source, audit.mapping, audit.current_test)
        elif audit.generator is not None:
            transform, helpers = generator_instrumentation(source, audit.generator, audit.current_test)
        else:
            transform, helpers = impact_instrumentation(source, audit)
        with trace_witnesses(source, audit, comparisons=False, extra_transformer=transform, extra_helpers=helpers), trace_class_uses(audit):
            yield
        return
    if audit.witness_mode == "tokenizer":
        with trace_tokenizer_inputs(source, audit), trace_class_uses(audit):
            yield
        return
    if not audit.witness_fields and not audit.important:
        with trace_class_uses(audit):
            yield
        return
    if audit.witness_mode == "comparisons":
        if any(name == "sqlglot" or name.startswith("sqlglot.") for name in sys.modules):
            raise RuntimeError("Load the comparison tracer before SQLGlot")
        with trace_witnesses(source, audit), trace_class_uses(audit):
            yield
        return
    mark_preloaded_modules(source, audit.attributes)
    attributes, helpers = dynamic_instrumentation(audit.attributes, audit.current_test)
    helpers[PROBE] = audit.conditions

    def transform(path, tree):
        tree = audit.conditions.transform(path, tree)
        return attributes(path, tree)

    with trace_witnesses(
        source, audit, attribute_names=audit.witness_fields,
        extra_transformer=transform, extra_helpers=helpers,
    ), trace_class_uses(audit):
        yield


@contextmanager
def trace_class_uses(audit):
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.expressions.core import Expr
    from sqlglot.generator import Generator
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer

    def traced(original):
        @wraps(original)
        def call(*args, **kwargs):
            if args:
                audit.record_class(args[0])
            return original(*args, **kwargs)

        return call

    methods = (
        (Dialect, "__init__"), (Dialect, "generate"), (Dialect, "parse"), (Dialect, "tokenize"),
        (Expr, "__init__"), (Generator, "__init__"), (Generator, "generate"),
        (Parser, "__init__"), (Parser, "parse"),
        (Tokenizer, "__init__"), (Tokenizer, "tokenize"),
    )
    originals = []
    try:
        for owner, name in methods:
            original = owner.__dict__.get(name)
            if original is not None:
                originals.append((owner, name, original))
                setattr(owner, name, traced(original))
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


class WorkerTask:
    def __init__(self, task_id, node_id, fn, args, kwargs):
        self.task_id = task_id
        self.node_id = node_id
        self.call = fn, args, kwargs

    def __reduce__(self):
        # The queue's feeder pickles this in the parent. Keeping the original
        # arguments opaque until the restore function runs also traces worker
        # input deserialization, including SQLGlot expression loading.
        encoded = bytes(ForkingPickler.dumps(self.call))
        return restore_task, (self.task_id, self.node_id, encoded)


class WorkerTrace:
    def __init__(self, source, directory, witness_fields, important, witness_mode):
        self.base = Path(directory) / f"worker-{os.getpid()}-{uuid4().hex}"
        self.tracer = coverage.Coverage(
            config_file=False, data_file=str(self.base.with_suffix(".cov")), source=[str(Path(source) / "sqlglot")]
        )
        self.audit = AuditTrace(source, witness_fields, important, witness_mode)
        self.task_ids = []
        self.tracer.start()
        sys.addaudithook(self.audit.audit)
        self.audit.active = True
        Finalize(None, self.save, exitpriority=0)

    def save(self):
        self.audit.active = False
        self.tracer.stop()
        self.tracer.save()
        self.base.with_suffix(".json").write_text(json.dumps({
            "tasks": self.task_ids,
            "fixtures": {path: sorted(nodes) for path, nodes in self.audit.fixtures.items()},
            "fixture_collection": sorted(self.audit.fixture_collection),
            "unknown_tests": sorted(self.audit.unknown_tests),
            "unknown_shared": self.audit.unknown_shared,
            **self.audit.export_observations(),
        }), encoding="utf-8")


_worker_trace = None


def initialize_worker(source, directory, encoded_initializer, witness_fields=(), important=(), witness_mode="full"):
    global _worker_trace
    _worker_trace = WorkerTrace(source, directory, witness_fields, important, witness_mode)
    tracing = trace_observations(_worker_trace.audit)
    tracing.__enter__()
    Finalize(None, tracing.__exit__, (None, None, None), exitpriority=-1)
    original_start = BaseProcess.start

    def start_nested_process(process):
        _worker_trace.audit.record_process()
        return original_start(process)

    BaseProcess.start = start_nested_process
    initializer, initargs = ForkingPickler.loads(encoded_initializer)
    if initializer is not None:
        initializer(*initargs)


def restore_task(task_id, node_id, encoded):
    _worker_trace.audit.node_id = node_id
    _worker_trace.audit.in_call = node_id is not None
    _worker_trace.tracer.switch_context(node_id or "")
    _worker_trace.task_ids.append(task_id)
    # Leave the context set until the next task's arguments arrive, so the
    # executor's result and exception pickling stays attributed to this test.
    return ForkingPickler.loads(encoded)


def run_task(call):
    fn, args, kwargs = call
    return fn(*args, **kwargs)
