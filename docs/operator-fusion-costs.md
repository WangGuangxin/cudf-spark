# Operator fusion cost study (2026-10-10)

The first prototype should investigate **partition + pack**, before changing partial
aggregate output or Parquet decoding. This is an implementation-risk recommendation,
not a measured fusion speedup. The measurements below describe existing paths; no new
production fusion, configuration, or JNI implementation was added.

Filter + PartialAggregate remains a useful second candidate. Its removable gather is
visible, especially across several string batches, but an external selection mask must
preserve the existing aggregation strategy and avoid allocating hash state for all input
rows when few survive. Full scan + filter + aggregate needs a separate decode study.

## Scope and baseline

- GPU: NVIDIA L4; driver 595.71.05; JDK 17; Spark 3.5.1 / Scala 2.12.21.
- Spark source baseline: `e9074507245c2bac6a3bb2875d6a841cd6d6d9a1`.
- JNI: `26.08.1-SNAPSHOT-cuda12`, source revision
  `260504b2b2f49d4fc56c6f5a69a1735ccb3d1961`. Artifact SHA-256:
  `9118791d9d7e22e65a99b794dd8d19780bb0e7c910f0aef4c83645faaa0d3729`.
- Resident GPU input: 131,072 total rows, split into 1 or 8 batches. No scan or decode.
- Keys: integer or fixed 32-byte strings; cardinality 16 or 131,072. Values are
  `row % 17`, with COUNT(value) and integer SUM(value). Fixtures contain no NULLs.
- Filter survival: 0%, 1%, 10%, 50%, 100%. A seeded random selector is independent of
  the grouping key; actual survivor counts are 0, 1,275, 12,916, 65,436, 131,072.
- Filter baseline uses the existing FilterProject evaluator, gathering only key and
  measure. Predicate-only columns are already excluded from materialized output.
- Coalescing target: 1 GiB. These small inputs coalesce into one nonempty aggregate
  input batch. The real partial iterator and its merge machinery are retained, but this
  geometry does **not** exercise multiple aggregate inputs, spilling, or split retry.
- The existing pre-project growth estimate and dynamic partial-sort heuristic are
  enabled. Low-cardinality cases use hash aggregation; unique-key cases use sort
  aggregation. Partial output skipping is permitted as in a normal Partial stage.
- Shuffle partitions: 1, 32, 256. Each output is checked against CPU Spark Murmur3
  hash / pmod, and every COUNT/SUM buffer is checked against a CPU-generated map.
- Three preparation paths: CPU Kudo, GPU Kudo, and uncompressed packed device buffers.
  Ordinary paths include D2H and serialization into a counting sink. The packed path
  stops at device buffers ready for caching. It does not register them in a catalog.
- No actual shuffle writer, file, compression, network, or commit time is measured.
  A plain local Spark session only initializes the environment needed by primitives.
- Formal timing uses ASYNC RMM, 512 MiB initial pool, no refcount debug or profiler,
  5 warmups and 21 samples per phase. The 112-case matrix runs in both forward and
  reverse order, in separate JVMs. Unpooled and refcount-debug runs are diagnostics.
- Nsight captures use ASYNC RMM and 3 measured repetitions per phase, separately from
  timing. Allocation requests are matched to printed epoch timestamps in the RMM log.
  Scoped peaks exclude resident input and retained comparison fixtures.

The input is deliberately small and can benefit from GPU cache residency. This study
does not establish a DRAM-bandwidth benefit at large working sets. Early JVM cases also
show substantial timing variation: for example the 1% integer/16-key pipeline medians
were 4.34 ms and 0.75 ms. Do not infer precise speedups from these wall-time results.

## Measurements

The following ranges are the two unprofiled run medians, in milliseconds. Separate
phase timings include their own synchronization and need not add to pipeline time.
`pack` includes D2H and host serialization for ordinary paths.

| String keys, one batch, 32 partitions | Partial output | Partition | Pack | Full pipeline |
|---|---:|---:|---:|---:|
| 16 keys, CPU Kudo | 16 rows / 836 B | 0.274–0.294 | 0.077–0.106 | 1.42–1.75 |
| 16 keys, GPU Kudo | 16 rows / 836 B | 0.299–0.361 | 0.135–0.153 | 1.46–1.48 |
| 16 keys, packed preparation | 16 rows / 836 B | 0.333–0.362 | 0.222–0.275 | 1.53–2.02 |
| Unique keys, CPU Kudo | 131,072 rows / 6.82 MB | 0.469–0.470 | 1.36–1.42 | 9.57–11.00 |
| Unique keys, GPU Kudo | 131,072 rows / 6.82 MB | 0.474–0.477 | 1.09–1.20 | 9.29–9.97 |
| Unique keys, packed preparation | 131,072 rows / 6.82 MB | 0.472–0.486 | 0.275–0.411 | 8.37–8.98 |

These rows have different endpoints and are **not an A/B comparison of shuffle
transports**. In particular, packed preparation avoids bulk D2H, while ordinary shuffle
still requires it. A future ordinary-shuffle fusion must compete with GPU Kudo as well
as CPU Kudo, and a caching fusion must compete with existing contiguous split.

Negative controls matter. With only 16 integer output rows and 256 partitions, CPU
pack took 0.053–0.209 ms, GPU Kudo 0.143–0.249 ms, and packed preparation
0.428–0.824 ms. Device packing has fixed metadata/allocation costs even for tiny output.
More partitions do not imply more removable payload traffic.

CUDA and RMM counts below are normalized per benchmark repetition. They come from
the separate profile run, not from subtracting wall times. Allocation bytes are the
sum of requests, not peak residency or physical upstream pool allocations.

| Isolated phase | Kernels | RMM requests | Requested bytes | Scoped peak |
|---|---:|---:|---:|---:|
| Integer gather, 1% survival, one batch | 4 | 4 | 541,891 | 539,588 |
| Integer gather, 50% survival, one batch | 4 | 4 | 1,311,823 | 1,309,520 |
| String gather, 10% survival, eight batches | 48 | 64 | 1,115,832 | 140,076 |
| Gather fast path, 100% survival, one batch | 0 | 0 | 0 | 0 |
| Unique string output, hash partition, 32 partitions | 17 | 21 | 13,661,634 | 11,035,349 |
| Same output, CPU Kudo pack | 0 | 0 | 0 | 0 |
| Same output, GPU Kudo pack | 14 | 8 | 6,835,204 | 6,828,807 |
| Same output, packed preparation | 13 | 43 | 6,848,521 | 6,841,859 |

CPU Kudo's zero GPU allocation/kernel counts do not imply zero cost: that phase copies
6,815,756 bytes D2H and performs host serialization. GPU Kudo copies 6,817,172 bytes
D2H, including its packed payload/offsets. Packed preparation copies 20,752 bytes D2H
for metadata, with the bulk payload remaining on device.

The eight-batch string gather contains 8 char-gather kernels, 8 offset-scan kernels,
8 row-selection kernels, 8 measure-gather kernels and 16 initialization kernels.
Together these consume about 148 microseconds of summed GPU kernel time in this
profile. The corresponding complete pipeline has 120 kernels and about 5.25 ms of
summed kernel time, largely in the existing string sort path. All 48 gather kernels
are an opportunity budget; a selection-aware implementation would still need work
to consume the selection and maintain aggregate semantics.

For unique string output, partition kernels sum to about 210 microseconds. Packed
preparation sums to about 41 microseconds, of which the final `copy_partitions` kernel
is about 11 microseconds. The full packed pipeline sums to about 6.58 ms. Eliminating
the second data copy cannot recover all partition/pack time: hashing, prefix/layout
calculation, string offsets, masks, metadata, and output buffers still have to exist.

CUDA memcpy counters do not include global-memory reads/writes inside gather, scatter,
or pack kernels. JNI call counts were not directly instrumented. Kernel counts, RMM
requests, and explicit copy bytes are the measured counters in this report.

## Implementation implications

1. **Prototype partition scatter into final packed buffers.** Keep aggregate input,
   output, and sort/hash selection unchanged. Preserve Spark hash seed 42, partition
   IDs, null/string semantics, and the current Kudo or packed-table format. The target
   is the intermediate partitioned table and its subsequent copy, not disk/network.
   Public `Table.partition` currently returns a reordered table, so a Scala wrapper
   alone cannot remove this boundary. GPU Kudo already combines split and serialization;
   its existing optimization must be part of the baseline.
2. **Only then consider partial aggregate emission directly into partitions.** This
   has a wider native integration surface. Aggregate spill buckets use a different hash
   and cannot be reused as Spark shuffle partition IDs. Gains scale with actual partial
   output M and width, not just original input N.
3. **Prototype Filter + PartialAggregate on a bounded eligible path.** Existing internal
   hash-groupby masks suggest an implementation foundation, but cuDF's public groupby
   API has no external predicate mask. The hash set/cache can be sized from original N;
   the current filtered baseline sizes work from surviving rows. Low-survival cases
   therefore need explicit capacity control. The mask must preserve real NULL key groups;
   `ignoreNullKeys` is not a substitute. A hash-only candidate must retain the current
   sort path when it wins rather than claiming an algorithm change as fusion savings.
4. **Delay full Parquet decode fusion.** Row-group pruning already exists. The current
   readers return materialized tables; this study contains no decode timing and cannot
   rank decode fusion quantitatively.

Neither route has demonstrated end-to-end acceleration yet. Before choosing a production
fusion, require an opt-in prototype A/B against the same manager/format and aggregation
strategy, with larger working sets, several aggregate input batches, NULLs, skew, wide
buffers, OOM/split/early-close checks, and real shuffle writer execution. Distributed
TPC-DS remains a later validation step.

## Reproduce

The benchmark is
[`OperatorFusionCostBenchmark.scala`](../tests/src/test/scala/com/nvidia/spark/rapids/OperatorFusionCostBenchmark.scala).
The checked-in matrix runner is [`operator_fusion_cost.py`](../tools/operator_fusion_cost.py).
It records the exact Java command and generated cases next to its log.

Build test sources using the repository build instructions, Spark 3.5.1 and the matching
JNI artifact. For example:

```bash
mvn verify -DskipTests -Dbuildver=351 \
  -Dspark-rapids-jni.version=26.08.1-SNAPSHOT -pl sql-plugin,tests -am
mvn -pl tests dependency:build-classpath -DincludeScope=test -Dbuildver=351 \
  -Dspark-rapids-jni.version=26.08.1-SNAPSHOT \
  -Dmdep.outputFile=/tmp/fusion-dependencies.txt
```

Create a classpath file containing compiled `tests/target/spark351/test-classes`, the
matching built SQL-plugin Spark351 JAR, and the dependency classpath. Use absolute paths
and a single colon-separated line. Run with JDK 17:

```bash
python3 tools/operator_fusion_cost.py --classpath-file /tmp/fusion-classpath.txt \
  --output-dir .cache/operator-fusion-cost/run --matrix full
python3 tools/operator_fusion_cost.py --classpath-file /tmp/fusion-classpath.txt \
  --output-dir .cache/operator-fusion-cost/run --matrix full --reverse
python3 tools/operator_fusion_cost.py --classpath-file /tmp/fusion-classpath.txt \
  --output-dir .cache/operator-fusion-cost/run --matrix breakdown
python3 tools/operator_fusion_cost.py --classpath-file /tmp/fusion-classpath.txt \
  --output-dir .cache/operator-fusion-cost/run --matrix profile --iterations 3
```

Use `--java /path/to/jdk17/bin/java` when needed. `--allocator NONE` is a diagnostic
control; do not mix its timings with ASYNC. The profile matrix requires `nsys`. Run GPU
measurements sequentially without other GPU work. Decode each profile using
`nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,nvtx_gpu_proj_sum`.

Local artifacts are under `.cache/operator-fusion-cost/2026-10-10/`:

- `async-forward.log` and `full-reverse.log`: 112 cases each, 480 phase timings each.
- `breakdown-forward.log`: predicate/gather separation for 10 controls.
- `profile-final/`: 30 captures, RMM log, exported SQLite/CSV reports and `summary.csv`.
- `timing-forward.log`, `timing-reverse.log`, `async-refcount-debug.log`: diagnostic
  runs with different allocator/debug settings; excluded from the formal timing tables.

Large raw artifacts remain in the ignored cache. The runner regenerates the matrices;
the tables above preserve the observations needed to review this recommendation.

Validation completed: Spark351 Maven `verify -DskipTests` succeeded (including test
compilation), the final source passed repository Scalastyle with zero warnings/errors,
and both full matrices, the breakdown matrix, and all 30 final captures completed
their CPU-result checks. Maven unit-test execution was skipped; these checks were
performed by the manual benchmark. JNI source/artifact remained unchanged.
