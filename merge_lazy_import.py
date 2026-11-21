"""Overlay the fully traced pinned lazy-load test into a raw coverage map."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile


PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
PIN_ROOT = "sqlglot-" + PIN
TEST = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"
DECLARED_PYTHON = {"python": "3.13.18", "platform": "linux", "machine": "x86_64"}


def read_json(path):
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    return json.loads(content)


def line_ranges(lines):
    spans = []
    for value in sorted(lines):
        if spans and spans[-1][1] + 1 == value:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return [first if first == last else [first, last] for first, last in spans]


def expand(spans):
    for span in spans:
        if isinstance(span, int):
            yield span
        else:
            first, last = span
            yield from range(first, last + 1)


def source_bytes(archive, path):
    item = archive.getmember(path)
    if not item.isfile():
        raise ValueError(f"Coverage source is not an ordinary file: {path}")
    with archive.extractfile(item) as stream:
        return stream.read()


def validate(evidence, original, vendored):
    if evidence["identity"] != DECLARED_PYTHON or evidence["test_id"] != TEST:
        raise ValueError("Expected the declared Linux, Python, and test identity")
    if evidence["passed"] is not True or evidence["child_subprocesses"] != 1:
        raise ValueError("Expected a passing test and exactly one traced child")
    observed = evidence["lines"]
    if not observed.get("sqlglot/__init__.py"):
        raise ValueError("Missing child package initialization evidence")
    hashes = {}
    with tarfile.open(original, "r:gz") as source, tarfile.open(vendored, "r:gz") as packaged:
        for name, lines in observed.items():
            path = PurePosixPath(name)
            if path.parts[0] != "sqlglot" or path.suffix != ".py" or ".." in path.parts:
                raise ValueError(f"Invalid coverage path: {name}")
            upstream = source_bytes(source, f"{PIN_ROOT}/{name}")
            production = source_bytes(packaged, name)
            if production != upstream:
                raise ValueError(f"Packaged source differs from pinned source: {name}")
            physical_lines = len(upstream.splitlines())
            # An empty imported module can report an event on CPython's
            # synthetic line zero. Keep the import path, not this coordinate.
            if lines == [0] and not upstream.strip():
                hashes[name] = sha256(production).hexdigest()
                continue
            if not lines or any(type(line) is not int or not 1 <= line <= physical_lines for line in lines):
                raise ValueError(f"Invalid coverage line number: {name}")
            hashes[name] = sha256(production).hexdigest()
        test_file = TEST.partition("::")[0]
        if source_bytes(source, f"{PIN_ROOT}/{test_file}") != source_bytes(packaged, test_file):
            raise ValueError("Production test source differs from the pinned test")
    return hashes


def associate(mapping, entries, number):
    for entry in entries:
        numbers = set(mapping.get(entry, ()))
        numbers.add(number)
        mapping[entry] = sorted(numbers)


def merge(coverage, evidence, hashes):
    numbers = [int(number) for number, node in coverage["test_ids"].items() if node == TEST]
    if len(numbers) != 1:
        raise ValueError("Test ID does not map uniquely to the supplied catalog")
    number = numbers[0]
    measured = coverage["coverage"].setdefault(str(number), {})
    for name, current in evidence["lines"].items():
        physical = set(current).union(expand(measured.get(name, ()))) - {0}
        if physical:
            measured[name] = line_ranges(physical)
        else:
            measured.pop(name, None)
    coverage["unmeasured"] = [item for item in coverage["unmeasured"] if int(item) != number]
    associate(coverage.setdefault("class_uses", {}), evidence["class_uses"], number)
    associate(coverage.setdefault("source_reads", {}), evidence["source_reads"], number)
    data_reads = evidence.get("data_reads", [])
    associate(coverage.setdefault("fixtures", {}), (path for path in data_reads if path.startswith("tests/fixtures/")), number)
    coverage["lazy_import_evidence"] = {
        "number": number,
        "test_id": TEST,
        "source_pin": PIN,
        "imported_files": sorted(hashes),
        "class_attribute_reads": evidence["class_attribute_reads"],
        "source_reads": evidence["source_reads"],
        "data_reads": data_reads,
    }
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--pinned-archive", required=True, type=Path)
    parser.add_argument("--production-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    coverage, evidence = read_json(args.coverage), read_json(args.receipt)
    hashes = validate(evidence, args.pinned_archive, args.production_archive)
    number = merge(coverage, evidence, hashes)
    content = (json.dumps(coverage, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    args.output.write_bytes(gzip.compress(content, mtime=0) if args.output.suffix == ".gz" else content)
    print(f"Validated {len(hashes)} declared Python paths and line ranges; merged test {number}.")


if __name__ == "__main__":
    main()
