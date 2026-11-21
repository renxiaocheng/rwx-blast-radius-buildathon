"""Replay the pinned lazy-load test and trace its one real Python child."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import coverage
import pytest


TARGET = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"
PROGRAM = "import sqlglot; assert len(sqlglot.Dialect._classes) == 1; print('Success')"


STARTUP = '''
import atexit
import json
import os
from pathlib import Path
import platform
import sys

import coverage

source = Path(os.environ['RWX_LAZY_SOURCE'])
root_prefix = str(source) + os.sep
prefix = str(source / 'sqlglot') + os.sep
output = Path(os.environ['RWX_LAZY_TRACE'])
source_reads = set()
data_reads = set()
class_uses = set()
active = True
tracer = coverage.Coverage(config_file=False, data_file=None, source=[str(source / 'sqlglot')])
tracer.start()


def audit(event, args):
    if active and event == 'open' and isinstance(args[0], (str, bytes)):
        candidate = os.path.abspath(os.fsdecode(args[0]))
        if not candidate.startswith(root_prefix):
            return
        caller = sys._getframe(1)
        if caller.f_code.co_filename == '<frozen importlib._bootstrap_external>':
            return
        relative = Path(candidate).relative_to(source).as_posix()
        if candidate.startswith(prefix) and candidate.endswith('.py'):
            source_reads.add(relative)
        else:
            data_reads.add(relative)


def record_class(cls):
    if not isinstance(cls, type):
        return
    for base in type.__getattribute__(cls, '__mro__'):
        module = type.__getattribute__(base, '__module__')
        if module.startswith('sqlglot.'):
            class_uses.add(module + '.' + type.__getattribute__(base, '__qualname__'))


def profile(frame, event, result):
    if not frame.f_code.co_filename.startswith(prefix):
        return
    name = frame.f_code.co_name
    if event == 'call' and name in {'__init__', '__init_subclass__', 'generate', 'parse', 'tokenize'}:
        values = frame.f_locals
        if 'self' in values:
            record_class(type(values['self']))
        if 'cls' in values:
            record_class(values['cls'])
    elif event == 'return' and name == '__new__':
        record_class(result)


def write_trace():
    global active
    active = False
    sys.setprofile(None)
    tracer.stop()
    data = tracer.get_data()
    payload = {
        'identity': print_environment(),
        'lines': {
            Path(path).relative_to(source).as_posix(): sorted(data.lines(path) or ())
            for path in sorted(data.measured_files())
            if data.lines(path)
        },
        'source_reads': sorted(source_reads),
        'data_reads': sorted(data_reads),
        'class_uses': sorted(class_uses),
        # This is the direct class-level lookup in the pinned -c program,
        # which is classified separately from profiler-visible constructors.
        'class_attribute_reads': [['sqlglot.dialects.dialect.Dialect', '_classes']],
    }
    staging = output.with_suffix('.tmp')
    staging.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    staging.replace(output)


def print_environment():
    return {'python': platform.python_version(), 'platform': sys.platform, 'machine': platform.machine()}


sys.addaudithook(audit)
sys.setprofile(profile)
atexit.register(write_trace)
'''


class Replay:
    def __init__(self, source, support, tracer):
        self.source = source
        self.support = support
        self.tracer = tracer
        self.output = support / "child.json"
        self.in_call = False
        self.launches = []
        self.untracked = False
        self.passed = set()

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self.in_call = item.nodeid == TARGET
        self.tracer.switch_context(TARGET if self.in_call else "")
        try:
            return (yield)
        finally:
            self.tracer.switch_context("")
            self.in_call = False

    def pytest_runtest_logreport(self, report):
        if report.nodeid == TARGET and report.passed:
            self.passed.add(report.when)

    @contextmanager
    def trace_child(self):
        original_init = subprocess.Popen.__init__

        def init(process, command, *args, **kwargs):
            if self.in_call:
                if (
                    tuple(command) != ("python", "-c", PROGRAM)
                    or args or kwargs.get("shell") or kwargs.get("executable")
                    or kwargs.get("cwd") is not None
                ):
                    self.untracked = True
                else:
                    environment = dict(kwargs.get("env") or os.environ)
                    previous_path = environment.get("PYTHONPATH")
                    environment["PYTHONPATH"] = os.pathsep.join(
                        (str(self.support), previous_path) if previous_path else (str(self.support),)
                    )
                    environment["RWX_LAZY_SOURCE"] = str(self.source)
                    environment["RWX_LAZY_TRACE"] = str(self.output)
                    kwargs["env"] = environment
                    self.launches.append(process)
            original_init(process, command, *args, **kwargs)

        with patch.object(subprocess.Popen, "__init__", init):
            yield

    def evidence(self):
        if self.untracked or len(self.launches) != 1 or self.launches[0].returncode != 0:
            raise RuntimeError("Expected exactly one successful recognized Python subprocess")
        if self.passed != {"setup", "call", "teardown"} or not self.output.is_file():
            raise RuntimeError("The selected test did not finish cleanly with a trace receipt")
        result = json.loads(self.output.read_text(encoding="utf-8"))
        identity = {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine()}
        if result["identity"] != identity:
            raise RuntimeError("The child compared a different Python or host environment")
        data = self.tracer.get_data()
        for filename in data.measured_files():
            lines = {line for line, contexts in data.contexts_by_lineno(filename).items() if TARGET in contexts}
            if lines:
                path = Path(filename).relative_to(self.source).as_posix()
                result["lines"][path] = sorted(lines.union(result["lines"].get(path, ())))
        if "sqlglot/__init__.py" not in result["lines"] or not result["lines"]["sqlglot/__init__.py"]:
            raise RuntimeError("The subprocess trace did not observe the package import")
        return {"test_id": TARGET, "passed": True, "child_subprocesses": len(self.launches), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    # Use the exact same Python for the actual test's bare `python` command.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tracer = coverage.Coverage(config_file=False, data_file=None, source=[str(source / "sqlglot")])
    with TemporaryDirectory(prefix="rwx-lazy-support-") as directory:
        support = Path(directory)
        (support / "sitecustomize.py").write_text(STARTUP, encoding="utf-8")
        replay = Replay(source, support, tracer)
        with tracer.collect(), replay.trace_child():
            status = pytest.main(["-q", "-p", "no:cacheprovider", TARGET], plugins=[replay])
        if status != pytest.ExitCode.OK:
            raise SystemExit(f"The unmodified pinned Pytest case failed: {int(status)}")
        result = replay.evidence()
        output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"Confirmed one real child; observed {sum(map(len, result['lines'].values()))} lines in {len(result['lines'])} files.")


if __name__ == "__main__":
    main()
