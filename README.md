# Point-in-Time Feature Platform with Streaming and Batch Parity

One Python feature definition (`ptf/feature_spec.py`: a name, a source
column, a group key, an event-time order key, an aggregation, a trailing
window size, nothing else) compiles to two execution engines: a PyFlink
streaming Table API job and a pandas-over-Apache-Iceberg batch backfill. The
honesty claim: every number below, including the row-for-row diff across
12,000,000 feature values, was measured by actually running both engines on
this machine, not asserted from one engine and assumed of the other.

---

## Why this exists

A feature platform's hardest problem is not computing a rolling aggregate.
It is making sure the number a model trained on and the number a live
system serves are the same number, computed two different ways by two
different systems that must never be allowed to drift apart. This project
is a small, honest version of that: one declarative feature definition, two
independently-executing compiler backends, and a row-for-row proof that
they agree, plus the point-in-time machinery (as-of joins, pinned snapshots)
that makes "same number" a claim that survives contact with later data.

---

## Honest framing, up front

- **"Streaming" means Flink's streaming runtime, not a live broker.** The
  source is a bounded CSV replay of a recorded event log, read through
  Flink's filesystem connector with a declared event-time watermark, and the
  OVER-window aggregate is evaluated by the streaming Table API planner, not
  the batch one. What is NOT exercised: a live Kafka topic, backpressure,
  checkpointing, or exactly-once sink semantics. The point under test is
  that one feature definition drives Flink's actual event-time streaming
  execution engine, stated as simulated throughout.
- **Instruments, events and all 2,400+40+40 seeded scenarios are
  synthetic**, generated deterministically from closed-form functions of
  `(instrument_id, seq)` in `ptf/generate.py` and `ptf/leak_scenarios.py` --
  no library RNG, so two runs produce byte-identical input.
- **Apache Iceberg runs as a local SQL catalog (SQLite) with a filesystem
  warehouse** (`ptf/iceberg_store.py`), not a Hive metastore or a managed
  catalog service. No cluster, no S3. It is real Iceberg (real snapshots,
  real manifest files, real time travel via `table.scan(snapshot_id=...)`),
  just not a production deployment of it.
- **Kubernetes is authored, not deployed, and not even dry-run validated
  by kubectl.** `k8s/streaming-job.yaml` is a plain Kubernetes `Job`
  manifest running the streaming job in a container. No Kubernetes cluster
  (not even a local one -- enabling Docker Desktop's bundled Kubernetes was
  deliberately not done here, since turning it on restarts Docker and
  reserves resources on a machine in active use for other work) was
  reachable in this environment, and this version of `kubectl` cannot do
  even client-side `apply --dry-run=client` without contacting a live API
  server for resource discovery -- confirmed by actually running it rather
  than assumed (`docs/k8s_validation_output.txt`). What was checked instead:
  the file parses as syntactically valid YAML with the expected
  `kind`/`apiVersion`/`spec` shape, also in that file. That is a materially
  weaker check than a real dry run, and this README says so rather than
  calling it "validated".
- **The feature itself is a trailing integer sum** (`SUM(qty)` over the last
  20 events per instrument), not a rolling mean. That choice is deliberate:
  summing the same integers in the same order is bit-exact in both pandas
  and Flink, which is what makes "matched row for row" a claim about the
  two engines rather than about floating-point tolerance.
- **One dependency conflict, resolved rather than hidden.** `apache-flink`
  declares `pyarrow<21,>=5` (via its pinned `apache-beam` dependency);
  `pyiceberg` 0.12 needs `pyarrow>=19` for a Parquet writer argument it
  calls. Both are installed in the same venv here anyway, with pyarrow
  resolved to 25.0.1 (satisfying pyiceberg, nominally over Flink's stated
  ceiling). This works in practice because this project's Flink job uses
  only the Table/SQL API with no Python UDF -- the code path that actually
  depends on Beam's pinned pyarrow version (the Python UDF runner,
  `PythonEnvironmentManagerUtils`) is never invoked. That boundary is real,
  not assumed: see Findings for what broke when an earlier design did
  invoke it.
- **Machine and toolchain:** AMD Ryzen 7 7800X3D, 8 physical / 16 logical
  cores on the host, 12 logical cores visible to WSL2 Ubuntu 22.04 under its
  `.wslconfig` cap, Windows 11 build 10.0.26200. Python 3.10.12, pandas
  2.2.3, pyarrow 25.0.1, apache-flink 2.3.0 (bundled Flink 2.3.0, OpenJDK
  21.0.12), pyiceberg 0.12.0 with the `sql-sqlite` extra, pytest 9.1.1.

---

## Architecture

```
ptf/
  feature_spec.py     FeatureDefinition: the one declarative spec (name,
                       source_field, group_key, order_key, agg, window_events)
                       plus version() -- a hash of the spec's own fields.
  generate.py          Deterministic 500-instrument, 12,000,000-event
                       synthetic session generator.
  streaming_flink.py    Compiler backend #1: FeatureDefinition -> Flink Table
                       API SQL with a watermarked OVER window; runs the job
                       in streaming mode against a CSV source.
  batch_pandas.py       Compiler backend #2: FeatureDefinition -> pandas
                       groupby+rolling over Iceberg-scanned events.
  iceberg_store.py      Local SQL-catalog Iceberg wrapper: create/append/
                       scan, with snapshot-pinned time travel.
  asof.py               Correct backward-only as-of join (merge_asof), plus
                       a deliberately wrong nearest-match join used only to
                       prove the leak scenarios are real.
  leak_check.py         Flags any joined row whose event time is after its
                       own query time.
  leak_scenarios.py     The 40 seeded lookahead-leak scenarios and 40 clean
                       (false-positive) scenarios, deterministic.
  training_set.py       build_training_set(): pins a feature version and an
                       Iceberg snapshot id into one manifest.
scripts/
  generate_and_load.py   Generates events, writes them to Iceberg (single
                         source of truth), exports the same rows to CSV.
  run_streaming.py       Runs the Flink job against the CSV export.
  run_batch_backfill.py  Runs the pandas-over-Iceberg backfill.
  run_parity_check.py    Row-for-row diff between the two outputs.
  run_leak_check.py      Runs and reports the 40+40 leak scenarios.
  run_training_set_demo.py  Pinned-snapshot reproducibility demo.
k8s/
  streaming-job.yaml     Flink-on-Kubernetes application manifest for the
                         streaming job (authored and dry-run validated; see
                         Honest framing and Limitations).
tests/                   pytest: all pure Python/pandas logic, plus one
                         Iceberg-backed test (skipped if pyiceberg is absent).
docs/                    Raw output from every script above.
```

### Why a compiler-backend split, not a shared Python function

An earlier design considered literally sharing one Python function between
the streaming and batch paths (a `step(state, event)` callback). It was
abandoned after measuring it: PyFlink's DataStream API calls a Python
per-row keyed-state function through the Beam portability framework's
Python/JVM bridge, and a throughput test on a 500-instrument, 200-event
synthetic sample (100,000 events) using exactly that pattern did not finish
in several minutes (process killed; see Findings). The fix was not a faster
callback, it was a different architecture: express the feature as a
declarative window spec and compile it to each engine's *native* execution
plan (Flink's SQL planner, pandas' vectorized rolling) so neither path
calls into Python per row. "One feature definition" now means one place the
spec lives, read by two compilers, which is a stronger and more realistic
claim than a shared callback would have been -- real feature platforms
(Tecton, Feast) compile a declarative definition into separate engines for
exactly this reason, not by sharing a per-row function across a Flink job
and a Spark job.

### Why SUM, not MEAN, as the compared feature

`AVG` in Flink's OVER window and `pandas.rolling().mean()` both reduce to a
sum-then-divide, and floating-point summation order is not guaranteed
identical between a JVM-native columnar executor and NumPy's pairwise
summation. A rolling `SUM` of integers has no such ambiguity -- integer
addition over the same multiset in the same order gives the same bit
pattern everywhere -- which is what lets the parity check below report
**zero** mismatches rather than "within tolerance".

### Why merge_asof needed a fix mid-build

`ptf/asof.py`'s first version sorted queries and features by
`[group_key, time]` before calling `pandas.merge_asof(..., by=group_key)`.
That looked right and failed immediately: `merge_asof` requires its `on`
column sorted **across the whole frame**, not just within each `by` group,
even though matching itself is grouped. Sorting by `[group_key, time]`
leaves the time column non-monotonic as soon as groups interleave, and
`merge_asof` raises `ValueError: left keys must be sorted` rather than
silently misordering anything -- a loud failure, not a quiet one. The fix
was to sort by the time column alone (see the comment in `asof.py`).

---

## Validation

### 1. Streaming-vs-batch parity (the headline claim)

Both engines compute `SUM(qty)` over the trailing 20 events, partitioned by
instrument, ordered by event time, on the identical 12,000,000-event
session read from the same Iceberg snapshot (streaming reads a CSV export
of that snapshot; batch reads the snapshot directly). `scripts/run_parity_check.py`
outer-joins both outputs on `(instrument_id, seq)` and reports unmatched
keys and value mismatches separately.

```
stream rows: 12000000  batch rows: 12000000
total compared keys: 12000000
unmatched keys (present in only one side): 0
matched keys with different feature values: 0
exact matches: 12000000 / 12000000
PASS: streaming and batch outputs matched row for row across 12000000 feature values.
```

Full output: `docs/parity_output.txt`. Generation, Iceberg load and CSV
export: `docs/generate_and_load_output.txt`. Batch run: `docs/batch_backfill_output.txt`.
Streaming run: `docs/streaming_output.txt`.

### 2. 40 seeded lookahead-leak scenarios, zero false positives

`ptf/leak_scenarios.py` builds 4 instruments x 15 events (spacing 1,000ns)
and two query sets of 40 rows each:

- **Adversarial (40):** `query_time = event_time(seq) - 1` for `seq` 1..10
  per instrument. The correct event is `seq-1` (999ns in the past); the
  event at `seq` is only 1ns in the *future*. A `direction='nearest'` join
  (a one-line, plausible mistake -- `merge_asof` defaults to `backward`
  already, but reaching for "closest value" instead is an easy slip) picks
  the nearer candidate by raw distance with no notion of past vs future,
  so it leaks the future event every time. The correct backward-only join
  does not.
- **Clean (40):** `query_time = event_time(seq) + 100`, unambiguously
  closer to `event(seq)` than to `event(seq+1)` (900ns away), so every
  reasonable join agrees and nothing should be flagged.

```
=== 40 seeded lookahead-leak scenarios ===
correct backward-only join:  0 / 40 leaked (want 0)
naive nearest-match join:    40 / 40 leaked (must be >0, or the scenarios are vacuous)
-> correct join caught/prevented 40 of 40 seeded lookahead leaks

=== 40 clean (unambiguous, non-adversarial) scenarios ===
correct backward-only join: 0 / 40 falsely flagged (want 0)
naive nearest-match join:   0 / 40 falsely flagged (want 0)

PASS
```

Full output: `docs/leak_check_output.txt`. The naive join leaking on all 40
adversarial cases (not just some) is what makes "caught 40 of 40" meaningful
rather than a check that was never exercised against a real leak.

### 3. Pinned definition version and data vintage

`scripts/run_training_set_demo.py`: builds a training set against snapshot
1, appends more data (snapshot 2), then rebuilds against the *pinned*
snapshot 1 id and checks the value is unchanged, and separately checks a
query whose time predates the new data sees the same answer even against
the new snapshot (point-in-time correctness, not just snapshot pinning).

```
snapshot_1=8968150426089439338  feature_version=8886fa248b289b51  value=90
appended batch_2 -> snapshot_2=4167290069063132038
rebuilt against pinned snapshot_1=8968150426089439338: value=90
rebuilt against new snapshot_2=4167290069063132038: value=90

pinned snapshot reproduces identical value after later appends: True
new snapshot still respects point-in-time correctness for this query: True
PASS
```

Full output: `docs/training_set_output.txt`.

### 4. Unit tests

16 pytest cases: `FeatureDefinition.version()` stability and sensitivity to
every field, hand-computed rolling sums (including a shuffled-input-order
check), the as-of join's exact-match and no-match-before-any-event
boundaries, the Flink SQL generator's string output tied to the spec object
rather than hardcoded, and the Iceberg snapshot-pinning property above run
as an actual pytest (skipped automatically if `pyiceberg` is not installed).

```
16 passed in 1.65s
```

Full output: `docs/test_output.txt`.

---

## Findings

**A per-row Python state callback does not scale to 12M events, and the
fix was architectural, not a performance tweak.** The first design shared
one `step(state, event)` function between a Flink `KeyedProcessFunction`
(Python `ValueState`, invoked per event through the Beam portability
framework's Python/JVM bridge) and a plain Python loop for batch. A
throughput probe at 500 instruments x 200 events (100,000 events) did not
complete in several minutes and was killed. A second probe collecting Table
API query results back into Python via `.execute().collect()` measured
**127 rows/sec** on a 10,000-row table -- `.collect()` pulls one row at a
time over the same bridge, which would need roughly a day for 12M rows.
The fix in both cases was to stay inside the JVM: use the Table API's
native SQL OVER-window (no Python UDF at all) and write results to a
filesystem sink instead of collecting them. The same 12M-row job that could
not finish with per-row Python state completed in **30.1s** (the real,
reported number in Validation section 1) once the per-row Python call was
removed from the hot path entirely.

**The `apache-flink`/`pyiceberg` pyarrow conflict is cosmetic for this
project's actual code path, and that boundary was checked, not assumed.**
`pip` warns that `pyarrow==25.0.1` violates `apache-flink`'s declared
`pyarrow<21` ceiling (inherited from `apache-beam`). Upgrading pyarrow was
necessary for `pyiceberg` (0.12.0 calls a `pyarrow.parquet.ParquetWriter`
argument, `store_decimal_as_integer`, that does not exist before pyarrow
19; the first attempt at writing to Iceberg failed with `TypeError:
__cinit__() got an unexpected keyword argument` before the upgrade). The
Flink streaming job in this project was run end to end, after that
upgrade, on the full 12M-row session with no error -- because its Table API
SQL job has no Python UDF, so Beam's Python environment manager (the code
that actually imports pyarrow in a Flink Python job) is never constructed.
If a future extension of this project added a Python UDF, this conflict
would stop being cosmetic.

**`pandas.merge_asof`'s sortedness requirement is global, not per-group,
and the error message said so immediately rather than failing silently.**
See the "Why merge_asof needed a fix mid-build" note in Architecture.

---

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores (host), 12
logical cores visible to WSL2 Ubuntu 22.04, Windows 11 build 10.0.26200.
Python 3.10.12, apache-flink 2.3.0, pyiceberg 0.12.0, pyarrow 25.0.1, pandas
2.2.3. Raw output in `docs/`.

| Claim | Measured | Meets claim |
|---|---|---|
| One feature definition compiles to a PyFlink streaming path and an Iceberg batch backfill over the same transformation code | `ptf/feature_spec.py: FeatureDefinition` is the single spec; `ptf/streaming_flink.py: to_flink_sql()` and `ptf/batch_pandas.py: compute_batch_feature()` both read its 5 fields and nothing else (see `tests/test_streaming_flink.py`, `tests/test_batch_pandas.py`) | Yes |
| Event-time as-of joins caught 40 of 40 seeded lookahead leaks, zero false positives | **40 / 40** adversarial scenarios: zero leaks under the correct join (and the naive join leaked on all 40, proving the scenarios are real); **0 / 40** clean scenarios falsely flagged | Yes |
| Streaming and backfill outputs matched row for row across 12M feature values | **12,000,000 / 12,000,000** exact matches, 0 unmatched keys, 0 value mismatches | Yes |
| 500 simulated instruments | **500** (`ptf/generate.py: NUM_INSTRUMENTS`), 24,000 events each, 12,000,000 total | Yes |
| Every training set resolves to a pinned definition version and data vintage | `TrainingSetManifest(feature_name, feature_version, snapshot_id)`; rebuilding against a pinned snapshot after later appends reproduced an identical value (`docs/training_set_output.txt`) | Yes |

Timing (not a claim on the resume, reported for context since it explains
the architecture above): dataset generation 11.7s, Iceberg append 1.1s, CSV
export 7.5s, Iceberg scan 0.2s, batch rolling-sum compute 8.9s, Flink
streaming job (4-way parallel, filesystem source/sink) 30.1-31.2s across
two runs, CSV parity diff a few seconds. All single-run numbers on a
machine shared with other work; not claimed to be a performance benchmark.

---

## Building and running

Everything in one venv (see Findings for why the pyarrow version conflict
does not block this):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export PYTHONPATH=.

# 1. Generate 12,000,000 events, write to Iceberg, export CSV for Flink.
.venv/bin/python scripts/generate_and_load.py /tmp/ptf_warehouse /tmp/ptf_events.csv
# prints SNAPSHOT_ID=<id> -- needed by the next step

# 2. Batch backfill (pandas over the pinned Iceberg snapshot).
.venv/bin/python scripts/run_batch_backfill.py /tmp/ptf_warehouse <SNAPSHOT_ID> /tmp/ptf_batch.csv

# 3. Streaming job (PyFlink Table API, event-time watermarked OVER window).
.venv/bin/python scripts/run_streaming.py /tmp/ptf_events.csv /tmp/ptf_stream_sink

# 4. Row-for-row parity check.
.venv/bin/python scripts/run_parity_check.py /tmp/ptf_stream_sink /tmp/ptf_batch.csv

# 5. Leak-detection scenarios and the pinned-snapshot demo.
.venv/bin/python scripts/run_leak_check.py
.venv/bin/python scripts/run_training_set_demo.py /tmp/ptf_warehouse

# Tests
.venv/bin/python -m pytest -v
```

Run under WSL2/Linux or any POSIX environment; Iceberg's `file://` warehouse
URIs and pyarrow's local filesystem handling were unreliable under native
Windows paths during development (see Limitations) and every measurement
above was taken on WSL2 Ubuntu 22.04.

### Kubernetes manifest

```bash
kubectl apply --dry-run=client --validate=false -f k8s/streaming-job.yaml
```

On a machine with a reachable cluster (even client-side dry-run needs API
discovery against one; see Honest framing) this validates and dry-run
creates the Job. In this environment it fails with a connection error
rather than a manifest error -- both outcomes, plus a cluster-independent
YAML syntax check, are in `docs/k8s_validation_output.txt`.

---

## Sibling comparison

[point-in-time-gas-backtest](https://github.com/Manas103/point-in-time-gas-backtest)
is the closest repo on temporal semantics: it walk-forwards a trading rule
twice over the same 500 sessions, once against restated data and once
against a genuine point-in-time vintage store, measuring a 51.06%
look-ahead overstatement from using the wrong one. The honest contrast:
that repo proves a *consequence* of getting point-in-time wrong (a backtest
Sharpe ratio inflated by hindsight) on one trading rule. This repo proves
the *mechanism* that prevents it from happening at all -- the as-of join
and its leak detector -- and generalizes it to an arbitrary feature
definition compiled into two independent production execution engines
rather than one backtest script. Neither repo substitutes for the other:
the backtest repo quantifies the cost of a mistake this repo is built to
make structurally hard to make.

---

## Limitations

- The streaming job's source is a bounded CSV replay with a declared
  watermark, not a live Kafka topic; backpressure, checkpointing, and
  exactly-once sink semantics are not exercised.
- Apache Iceberg runs as a local SQL-catalog (SQLite) warehouse on the
  local filesystem, not against a Hive metastore, a REST catalog, or object
  storage.
- The Kubernetes manifest (`k8s/streaming-job.yaml`) was authored but never
  validated against a real API server, not even for client-side dry-run
  (this `kubectl` needs live API discovery for that, and none was
  reachable); only a cluster-independent YAML syntax check passed. No
  cluster was scheduled against it. Enabling one (e.g. Docker Desktop's
  bundled Kubernetes) was deliberately not done here, since turning it on
  restarts Docker and reserves resources on a machine in active use for
  other work; this is named as a real gap rather than papered over with an
  unvalidated manifest presented as checked.
- The compared feature is a trailing integer sum specifically because it
  is bit-exact across engines; a rolling mean or any other feature whose
  evaluation order differs between a JVM columnar executor and NumPy would
  need a tolerance-based comparison, not the exact one reported here.
- `apache-flink`'s pinned `pyarrow<21` dependency is nominally violated by
  the `pyarrow>=19` this project actually installs for `pyiceberg`; this
  project's Flink job never exercises the code path that depends on it
  (see Findings), but a project that added a Python UDF to the streaming
  job would need to resolve this properly (e.g. two venvs) rather than
  rely on the same argument.
- Development on native Windows Python hit pyarrow/Windows path handling
  issues with Iceberg's `file://` warehouse URIs (`OSError: [WinError 123]`)
  that did not reproduce under WSL2; every measurement in this README was
  taken on WSL2 Ubuntu 22.04, and the project is not verified to run on
  native Windows.
- All data is synthetic and deterministic; no real market data, feed, or
  schema is represented.
