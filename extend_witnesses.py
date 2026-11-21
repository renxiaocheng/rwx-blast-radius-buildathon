"""Add separate pinned-run observations to a baseline witness profile."""

import argparse
import gzip
import json
from pathlib import Path

from comparison_witnesses import ComparisonEvidence, unpack_comparison_row
from branch_witnesses import BranchEvidence
from coverage_selector import TestSourceIndex
from dialect_impact import _module_name
from generator_impact import GeneratorRecorder
from mapping_impact import MappingImpact, declarations
from mapping_witnesses import MappingRecorder, MappingWitnesses
from shared_mapping_witnesses import ModuleMappingRecorder, ModuleMappingWitnesses, namespace_source_digest
from tokenizer_impact import TokenizerEvidence


def load_profile(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def completed_tests(report: dict) -> set[str]:
    complete = {report["test_ids"][number] for number in report["coverage"]}
    if not complete:
        raise ValueError("No tests finished in the observation report")
    return complete


def supplement_comparisons(profile: dict, reports: list[Path]) -> None:
    evidence = ComparisonEvidence()
    node_ids = profile.pop("comparison_node_ids", ())
    evidence.merge(unpack_comparison_row(row, node_ids) for row in profile["comparison_witnesses"])
    complete = set(profile.get("comparison_complete", ()))
    for path in reports:
        report = load_profile(path)
        if report.get("witness_mode") != "comparisons":
            raise ValueError(f"Expected a complete comparison trace in {path}")
        finished = completed_tests(report)
        complete.update(finished)
        evidence.merge(
            [*site, {node: samples for node, samples in records.items() if node in finished}]
            for *site, records in report["comparison_witnesses"]
            if site[-1] or any(node in finished for node in records)
        )
    profile["comparison_witnesses"] = evidence.export()
    profile["comparison_complete"] = sorted(complete)


def compact_comparisons(profile: dict) -> None:
    if "comparison_node_ids" in profile:
        return
    rows = profile["comparison_witnesses"]
    identifiers = sorted({node for row in rows for node in row[-1]})
    numbers = {node: str(number) for number, node in enumerate(identifiers)}
    profile["comparison_node_ids"] = identifiers
    profile["comparison_witnesses"] = [
        [*row[:-1], {numbers[node]: samples for node, samples in row[-1].items()}] for row in rows
    ]


def supplement_tokenizer(profile: dict, reports: list[Path]) -> None:
    evidence = TokenizerEvidence()
    complete = set()
    for path in reports:
        report = load_profile(path)
        if report.get("witness_mode") != "tokenizer":
            raise ValueError(f"Expected a tokenizer trace in {path}")
        complete.update(completed_tests(report))
        evidence.merge(report["tokenizer_witnesses"])
    profile["tokenizer_witnesses"] = evidence.export()
    profile["tokenizer_complete"] = sorted(complete - evidence.unknown)


def supplement_mappings(profile: dict, reports: list[Path], source: Path | None) -> None:
    evidence = MappingRecorder()
    complete = set()
    for path in reports:
        report = load_profile(path)
        if report.get("witness_mode") != "mapping":
            raise ValueError(f"Expected a mapping trace in {path}")
        complete.update(completed_tests(report))
        evidence.merge(report["mapping_witnesses"])
    payload = evidence.export()
    payload["reliable"] = sorted(complete)
    if source is not None:
        payload["census"] = MappingWitnesses(source.resolve(), payload, {}).build_census()
    profile["mapping_witnesses"] = payload


def supplement_generators(profile: dict, reports: list[Path]) -> None:
    evidence = GeneratorRecorder()
    complete = set()
    for path in reports:
        report = load_profile(path)
        if report.get("witness_mode") != "generator":
            raise ValueError(f"Expected a generator trace in {path}")
        complete.update(completed_tests(report))
        evidence.merge(report["generator_witnesses"])
    profile["generator_witnesses"] = evidence.export()
    profile["generator_complete"] = sorted(complete)


def pack_namespace_readers(profile: dict, source: Path, numbers_by_id: dict[str, int]) -> None:
    """Precompute namespace observers of fields declared by unmodified SQLGlot."""
    payload = profile["module_mapping_witnesses"]
    witnesses = ModuleMappingWitnesses(source, payload, numbers_by_id)
    index = TestSourceIndex(source, [
        {"id": node, "number": number, "file": node.partition("::")[0]}
        for node, number in numbers_by_id.items()
    ])
    ids_by_number = {number: node for node, number in numbers_by_id.items()}
    impact = MappingImpact(source)
    observed = {origin for origin, *_ in profile["mapping_witnesses"]["reads"]}
    readers = {}
    for file in (source / "sqlglot").rglob("*.py"):
        path = file.relative_to(source).as_posix()
        module = _module_name(path)
        tree = impact._tree(path)
        if tree is None:
            raise ValueError(f"Cannot read pinned source {path}")
        for origin in declarations(tree, module).keys() & observed:
            receiver, _, field = origin.rpartition(".")
            if receiver == module:
                tests = witnesses.select(path, module, field, tree_for=impact._tree, mentions=index.mentioning)
                readers[origin] = None if tests is None else sorted(ids_by_number[number] for number in tests)
    payload["reader_census"] = {"source_digest": namespace_source_digest(source), "readers": readers}


def supplement_impact(profile: dict, reports: list[Path], source: Path | None = None) -> None:
    branches = BranchEvidence()
    namespaces = ModuleMappingRecorder()
    complete = set()
    numbers_by_id = {}
    for path in reports:
        report = load_profile(path)
        if report.get("witness_mode") != "impact":
            raise ValueError(f"Expected a branch and namespace trace in {path}")
        complete.update(completed_tests(report))
        branches.merge(report["branch_witnesses"])
        namespaces.merge(report["module_mapping_witnesses"])
        for number, node in report["test_ids"].items():
            if numbers_by_id.setdefault(node, int(number)) != int(number):
                raise ValueError(f"Inconsistent catalog numbers in {path}")
    profile["branch_witnesses"] = branches.export()
    profile["branch_complete"] = sorted(complete)
    profile["module_mapping_witnesses"] = namespaces.export()
    profile["module_mapping_witnesses"]["reliable"] = sorted(complete)
    if source is not None:
        pack_namespace_readers(profile, source, numbers_by_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, nargs="+", default=[])
    parser.add_argument("--tokenizer", type=Path, nargs="+", default=[])
    parser.add_argument("--mappings", type=Path, nargs="+", default=[])
    parser.add_argument("--positions", type=Path, help="Pinned test's source inspection observations")
    parser.add_argument("--generator", type=Path, nargs="+", default=[])
    parser.add_argument("--impact", type=Path, nargs="+", default=[])
    parser.add_argument("--compact-comparisons", action="store_true", help="Store comparison test IDs once")
    parser.add_argument("--source", type=Path, help="Pinned source for mapping and namespace censuses")
    args = parser.parse_args()
    profile = load_profile(args.baseline)
    if args.comparisons:
        supplement_comparisons(profile, args.comparisons)
    if args.tokenizer:
        supplement_tokenizer(profile, args.tokenizer)
    if args.mappings:
        supplement_mappings(profile, args.mappings, args.source)
    if args.positions is not None:
        profile["position_witnesses"] = load_profile(args.positions)
    if args.generator:
        supplement_generators(profile, args.generator)
    if args.impact:
        supplement_impact(profile, args.impact, args.source)
    if args.compact_comparisons:
        compact_comparisons(profile)
    with args.output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(json.dumps(profile, separators=(",", ":"), sort_keys=True).encode("utf-8"))


if __name__ == "__main__":
    main()
