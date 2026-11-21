# Blast Radius coverage-based entry

This entry selects tests using per-test line coverage collected on SQLGlot at
`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. It maps changed Python statements,
static dialect configuration, and SQL fixtures to baseline test observations.
It compares compiled code for behavior-preserving edits and retains observed
source readers and tests that inspect Python code locations. It keeps tests
whose coverage is incomplete and uses the full suite when the available
evidence cannot bound an edit. Worker count and grouping use the recorded test
times and per-worker overhead.

The compressed coverage in `vendor/coverage.json.gz` was collected in the
contest's Linux/amd64 Python 3.13 image using `collect_coverage.py` and
`coverage_worker.py`. The collector attributes work in spawned process pools
to the test that submitted it and records fixture reads, explicit Python-source
reads, and SQLGlot class use. Three optimizer tests that hit a DuckDB error
under x86 emulation were re-traced with the same dependency versions in a native
Linux/arm64 container. A separate Linux/amd64 replay traced the fresh Python
process launched by the lazy-import test. The compressed line index preserves
those test memberships and uses test IDs to match the scorer's catalog.
`vendor/dialect-index.json.gz` stores source imports, field access locations,
and references from the pinned tests. `vendor/witnesses.json.gz` stores raw
comparison operands, guard inputs, and class field reads from the same pinned
tests. The selector checks a received patch against these observations. Shared
observations, unsupported values, and incomplete traces preserve the safe
fallback. The source archive contains the pinned Python source and its upstream
license. All lookup data comes from the unmodified upstream source and tests.

For regeneration, `collect_coverage.py` accepts the pinned checkout and the
official test catalog. `capture_lazy_import.py` and `merge_lazy_import.py` record
and merge the isolated child's observations. `pack_coverage.py` converts the
merged coverage into the compressed line index. Run `dialect_impact.py --help`
for the static index generator. Run `collect_coverage.py --witnesses` and pass
the reports and pinned `--source` to `pack_witnesses.py` to regenerate the
runtime profile and its static field access index.

The selector can also read pinned observations of individual mapping keys and
numeric escape prefixes. Key-level selection applies when the patch keeps the
mapping's ordered keys and changes a supported value. The lookup trace follows
the loaded mapping by identity, and direct consumers, shared reads, and
unresolved uses constrain narrowing. A separate tokenizer trace records only
escape prefixes entering the actual tokenizer. Each profile tracks which tests
finished with that observer installed. Tests outside its coverage and any
unresolved edit retain their conservative selection.

`collect_coverage.py` provides `--comparison-witnesses`, `--mapping-witnesses`,
`--tokenizer-witnesses`, `--generator-witnesses`, and `--impact-witnesses`
for separate observations. The generator profile records calls to the pinned
`rename_func` closures. The impact profile tracks integer comparisons in
guards and reflective module-namespace reads. For supported guard edits, the
recorded operand ranges can show that a test would take the same branch.
For supported mapping edits, the selector combines key and namespace reads
with source checks for consumers of a derived membership set. Incomplete
observations keep the corresponding tests selected.

The isolated source inspection replay is in `position_witnesses.py`. Combine
the raw reports with the baseline using `extend_witnesses.py --help`. These
reports use the unmodified pinned checkout and the official test catalog.

## The problem

Your image gets a batch of code changes to [sqlglot](https://github.com/tobymao/sqlglot) at commit `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. For each change, print which tests to run and how to split them across up to 8 workers.

| | |
| --- | --- |
| `/data/commits/<id>.patch` | One unified diff against the pinned commit. Each stands alone: none builds on another. |
| `/data/tests.json` | Every test, numbered, with its file and the seconds it took us to run. |
| stdout | One JSON line a commit: `{"commit": "k7q2", "workers": [[12, 845, 901], [3, 77]]}` |

Each inner list is one worker's test numbers. `"workers": []` runs nothing, which is the right answer for a change that breaks nothing.

## Scoring

Lower is better. [SCORING.md](SCORING.md) has the exact rules; in short:

- A commit costs the recorded seconds of its busiest worker, plus 2 seconds for every worker.
- A commit where you left out a test that fails adds 300 seconds.
- The CPU seconds your program uses are added. Running sqlglot's tests while you decide is allowed, and you pay for the time.
- Your image runs 3 times and the middle total counts.

"Fails" means pytest reports a failure, an error or a timeout for the test, or the test's file cannot be imported. A test with subtests is one test: one failing subtest fails it.

## Try it

You need Docker, make and Python 3.9 or newer. The kit unpacks into the folder you are in, so unpack it into an empty one.

```sh
make sample   # build, run once on the 30 sample commits under the scorer's limits, and score
make compare  # run the four-worker full-suite baseline and the coverage entry
make check    # the output's format only
```

The submission repository omits the sample kit. Extract the official
[sample archive](https://www.joinbuilders.org/buildathon/tracks/blast-radius/files/blast-radius-sample.tar.gz)
in a separate directory and run `make compare SAMPLE=/path/to/official-kit/sample`
to test a fresh checkout.

`make sample` prints your simulated seconds and misses, which are exact for that output. The board adds the CPU seconds your container used and takes the middle of 3 runs, so expect it a little higher. Your program never sees `sample/answers.json`: only the patches and `tests.json` are mounted.

`sample/answers.json` lists the tests each sample commit breaks and the kind of change it is. The practice and final commits are hidden, and so is everything your program prints when it runs on them. A rejected submission tells you which rule it broke and nothing more, so reproduce problems here.

## Submit

Push to a public GitHub repository with the `Dockerfile` at its top, then enter the repository and the commit hash on the track page. The final scores the commit you pick. If you pick none, your best practice submission counts.

## Rules

- Submit your own team's repository. A submission with the same repository, commit or archive as another team's is disqualified, the later one.
- Your build may download only from public package registries: PyPI, npm, Docker Hub, GitHub releases at a fixed tag, and sqlglot’s own repository at the pinned commit. Nothing from an address your team controls. The final rebuilds every image after the cutoff, so anything else could change a submission after the deadline.
- No answers worked out ahead of time and looked up by patch: no table from a patch, or from a kind of edit, to the tests it breaks. Maps built from the pinned commit (coverage, imports, fixtures) are allowed. Top entries are read.
- Only the final counts. The practice board carries no prize.

## How your image runs

- The build has the network, 4 cores, 8 GB of memory and 10 minutes. It may clone sqlglot at the pinned commit, install it and run its tests to make coverage or import maps.
- Your repository may hold 20 MB. Git submodules and Git LFS files are not fetched: clone sqlglot in the `Dockerfile` and commit maps as plain files.
- The run has no network, 2 cores, 4 GB of memory and 3 minutes for the whole batch (4.5 in the final, which has about one and a half times as many commits).
- The disk is read-only. `/tmp` holds 64 MB and is empty at the start of every run.
- The container runs as user `nobody`, with `HOME=/tmp` and at most 256 processes.
- stdout may hold 2 MB. More rejects the run.

## Things that will catch you

- Logs go to stderr. Any line on stdout that is not an answer rejects the run.
- `score.py` names a broken rule by its code, as in `Rejected: not_json on line 1`. The table in SCORING.md says what each code means.
- Files your program needs must be readable by `nobody`. A map written by `root` during the build with mode 600 is not.
- pytest 9 prints a test as `PASSED` beside `SUBFAILED` lines when one of its subtests fails. That test failed.
- sqlglot skips its executor tests unless `SKIP_INTEGRATION=0` is set. We labeled with it set, so those tests can fail.
- `requirements.lock` is every package in the environment we labeled in, on the image the `Dockerfile` starts from. To run sqlglot's tests the way we did, `pip install -r requirements.lock` and put the sqlglot checkout on `PYTHONPATH`; sqlglot's own `pip install -e .[dev]` picks newer versions.
- The scorer is linux/amd64. On an Apple Silicon Mac, build and run with `--platform linux/amd64`, as the Makefile does, or your timings and wheels will differ.
- A change that only adds a comment is not promised to be harmless.
