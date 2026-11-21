"""Spawn-safe Python coverage instrumentation for SQLGlot's test pools."""

from collections import defaultdict
import json
from multiprocessing.process import BaseProcess
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import Finalize
import os
from pathlib import Path
import sys
from uuid import uuid4

import coverage


PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.exec"}


class AuditTrace:
    def __init__(self, source):
        self.source = str(source) + os.sep
        self.fixture_root = self.source + os.path.join("tests", "fixtures") + os.sep
        self.active = False
        self.node_id = None
        self.in_call = False
        self.fixtures = defaultdict(set)
        self.fixture_collection = set()
        self.unknown_tests = set()
        self.unknown_shared = False

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

    def merge_fixtures(self, fixtures, collection):
        for name, nodes in fixtures.items():
            self.fixtures[name].update(nodes)
        self.fixture_collection.update(collection)


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
    def __init__(self, source, directory):
        self.base = Path(directory) / f"worker-{os.getpid()}-{uuid4().hex}"
        self.tracer = coverage.Coverage(
            config_file=False, data_file=str(self.base.with_suffix(".cov")), source=[str(Path(source) / "sqlglot")]
        )
        self.audit = AuditTrace(source)
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
        }), encoding="utf-8")


_worker_trace = None


def initialize_worker(source, directory, encoded_initializer):
    global _worker_trace
    _worker_trace = WorkerTrace(source, directory)
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
