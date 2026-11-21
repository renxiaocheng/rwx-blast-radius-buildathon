"""Observe the pinned test that compares SQLGlot source and code objects."""

import argparse
from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from types import CodeType, FunctionType, MethodType


TEST_FILE = "tests/test_typing.py"
TEST_NODE = f"{TEST_FILE}::TestTyping::test_no_redundant_expression_metadata"
TEST_DIGEST = "b2366d808c1b0368be23bed1f827aff6075c61544d2a301c0f95470c027c0ee9"
SCORER_IDENTITY = {"platform": "linux", "machine": "x86_64", "python": "3.13.18"}
VERSION = 1
PROCESS_EVENTS = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.fork", "os.exec"}


def source_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def observed_source_path(name: str) -> bool:
    path = PurePosixPath(name)
    return path.parts[:1] == ("sqlglot",) and path.suffix == ".py" and ".." not in path.parts


class Recorder:
    def __init__(self, source: Path):
        self.source_prefix = str(source) + os.sep
        self.target_file = str(source / TEST_FILE)
        self.active = False
        self.complete = True
        self.calls = 0
        self.files_read = set()
        self.code_files = set()

    def relative_path(self, filename: str) -> str | None:
        if filename.startswith("<"):
            self.complete = False
            return None
        resolved = os.path.abspath(filename)
        if resolved.startswith(self.source_prefix):
            name = resolved[len(self.source_prefix):].replace(os.sep, "/")
            return name if observed_source_path(name) else None
        return None

    def audit(self, event, args):
        if not self.active:
            return
        if event in PROCESS_EVENTS:
            self.complete = False
        elif event == "open" and isinstance(args[0], (str, bytes)):
            caller = sys._getframe(1)
            if caller.f_code.co_filename != "<frozen importlib._bootstrap_external>":
                relative = self.relative_path(os.fsdecode(args[0]))
                if relative is not None:
                    self.files_read.add(relative)

    def code(self, code: CodeType) -> None:
        name = self.relative_path(code.co_filename)
        if name is not None:
            self.code_files.add(name)
        for constant in code.co_consts:
            if type(constant) is CodeType:
                self.code(constant)

    def profile(self, frame, event, result):
        if event != "call" or frame.f_code.co_name != "_same_code" or frame.f_code.co_filename != self.target_file:
            return
        self.calls += 1
        for slot in ("f", "g"):
            function = frame.f_locals[slot]
            if type(function) is MethodType:
                function = function.__func__
            if type(function) is FunctionType:
                self.code(function.__code__)
            else:
                self.complete = False


def capture(source: Path) -> dict:
    """Replay the actual pinned test once in a fresh interpreter."""
    import pytest

    source = source.resolve()
    if source_digest(source / TEST_FILE) != TEST_DIGEST:
        raise ValueError("Unexpected test source fingerprint")
    recorder = Recorder(source)
    phases = set()

    class Plugin:
        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_runtest_call(self, item):
            if item.nodeid != TEST_NODE:
                return (yield)
            previous = sys.getprofile()
            callback = recorder.profile
            recorder.active = True
            sys.setprofile(callback)
            try:
                return (yield)
            finally:
                if sys.getprofile() is not callback:
                    recorder.complete = False
                sys.setprofile(previous)
                recorder.active = False

        def pytest_runtest_logreport(self, report):
            if report.nodeid == TEST_NODE and report.passed:
                phases.add(report.when)

    original_cwd = Path.cwd()
    sys.addaudithook(recorder.audit)
    os.chdir(source)
    sys.path.insert(0, str(source))
    try:
        outcome = pytest.main(["-q", "-p", "no:cacheprovider", TEST_NODE], plugins=[Plugin()])
    finally:
        sys.path.pop(0)
        os.chdir(original_cwd)
    if outcome != pytest.ExitCode.OK or phases != {"setup", "call", "teardown"}:
        raise RuntimeError(f"The pinned source-position test did not pass cleanly (pytest status {int(outcome)})")
    files = recorder.files_read | recorder.code_files
    if not recorder.calls or not recorder.files_read or not recorder.code_files or not recorder.complete:
        raise RuntimeError("Incomplete source-position coverage")
    return {
        "version": VERSION,
        "test_node": TEST_NODE,
        "test_digest": TEST_DIGEST,
        "identity": {"platform": sys.platform, "machine": platform.machine(), "python": platform.python_version()},
        "comparison_calls": recorder.calls,
        "file_reads": sorted(recorder.files_read),
        "code_files": sorted(recorder.code_files),
        "digests": {name: source_digest(source / name) for name in sorted(files)},
    }


class PositionWitnesses:
    def __init__(self, source: Path, evidence: Mapping | None, numbers_by_id: Mapping[str, int]):
        self.source = Path(source)
        self.evidence = evidence or {}
        self.number = numbers_by_id.get(TEST_NODE)
        self.files = set()
        self.trusted = None

    def validate(self) -> bool:
        evidence = self.evidence
        if (
            evidence.get("version") != VERSION or evidence.get("test_node") != TEST_NODE
            or evidence.get("test_digest") != TEST_DIGEST or evidence.get("identity") != SCORER_IDENTITY
            or not evidence.get("comparison_calls")
        ):
            return False
        reads, code_files = evidence.get("file_reads"), evidence.get("code_files")
        hashes = evidence.get("digests")
        if not reads or not code_files or not isinstance(hashes, dict):
            return False
        paths = set(reads) | set(code_files)
        if paths != set(hashes) or any(not observed_source_path(name) for name in paths):
            return False
        try:
            if source_digest(self.source / TEST_FILE) != TEST_DIGEST:
                return False
            if any(source_digest(self.source / name) != hashes[name] for name in paths):
                return False
        except OSError:
            return False
        self.files = paths
        return True

    def refine(self, name: str, position_readers: set[int]) -> set[int]:
        if self.number not in position_readers or not observed_source_path(name):
            return position_readers
        if self.trusted is None:
            self.trusted = self.validate()
        if not self.trusted or name in self.files:
            return position_readers
        return position_readers - {self.number}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = capture(args.source)
    args.output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"Observed {len(receipt['file_reads'])} direct source files and {len(receipt['code_files'])} code files "
        f"across {receipt['comparison_calls']} comparisons."
    )


if __name__ == "__main__":
    main()
