"""Combine runtime observations from completed tests on the pinned source."""

import argparse
import gzip
import json
from pathlib import Path

from attribute_witnesses import AttributeRecorder, AttributeWitnesses
from comparison_witnesses import ComparisonEvidence
from condition_witnesses import ConditionRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("observations", nargs="+", type=Path)
    parser.add_argument("--source", type=Path, help="Pinned source used to index field access sites")
    args = parser.parse_args()
    comparisons = ComparisonEvidence()
    attributes = AttributeRecorder()
    conditions = ConditionRecorder(None, ())
    reliable = set()
    for path in args.observations:
        raw = json.loads(path.read_text())
        complete = {raw["test_ids"][number] for number in raw["coverage"]}
        if not complete:
            raise ValueError(f"No completed tests in {path}")
        reliable.update(complete)
        comparisons.merge(
            [*site, {node: samples for node, samples in records.items() if node in complete}]
            for *site, records in raw["comparison_witnesses"]
            if site[-1] or any(node in complete for node in records)
        )
        attributes.merge(raw["attribute_witnesses"])
        guard_observations = raw["condition_witnesses"]
        conditions.merge({
            "shared": guard_observations["shared"],
            "sites": {
                site: {node: rows for node, rows in tests.items() if node in complete}
                for site, tests in guard_observations["sites"].items()
            },
        })
        print(f"{path.name}: {len(complete)} completed tests")
    attribute_payload = attributes.export()
    if args.source is not None:
        attribute_payload["census"] = AttributeWitnesses(args.source.resolve(), attribute_payload, {}).build_census()
    data = {
        "reliable": sorted(reliable),
        "comparison_witnesses": comparisons.export(),
        "attribute_witnesses": attribute_payload,
        "condition_witnesses": conditions.export(),
    }
    with args.output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    print(f"{args.output}: {args.output.stat().st_size:,} bytes for {len(reliable)} tests")


if __name__ == "__main__":
    main()
