"""Index pinned SQLGlot coverage by line without changing the observations."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def covered_lines(spans):
    for span in spans:
        if isinstance(span, int):
            yield span
        else:
            yield from range(span[0], span[1] + 1)


def grouped_lines(lines):
    spans = []
    for number in sorted(lines):
        if spans and spans[-1][-1] == number - 1:
            spans[-1][-1] = number
        else:
            spans.append([number, number])
    return [start if start == end else [start, end] for start, end in spans]


def index_coverage(data):
    by_line = defaultdict(lambda: defaultdict(set))
    coverage = data.pop("coverage")
    for number, files in coverage.items():
        for path, spans in files.items():
            for line in covered_lines(spans):
                by_line[path][line].add(int(number))
    groups = {}
    index = {}
    for path, lines in sorted(by_line.items()):
        entries = defaultdict(list)
        for line, numbers in sorted(lines.items()):
            key = tuple(sorted(numbers))
            group = groups.setdefault(key, len(groups))
            entries[group].append(line)
        index[path] = [[group, grouped_lines(matches)] for group, matches in entries.items()]
    data.update(measured=sorted(map(int, coverage)), line_index=index, line_test_sets=list(groups))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.coverage.suffix == ".gz" else open
    with opener(args.coverage, "rt", encoding="utf-8") as stream:
        data = index_coverage(json.load(stream))
    with args.output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    print(f"{len(data['line_test_sets'])} distinct coverage sets in {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
