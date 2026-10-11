# Partial aggregate + packed shuffle fusion

Independent, default-off session switch (set before constructing the query plan):

```text
spark.rapids.shuffle.partialAggPack.enabled=true
```

This switch works independently of `spark.rapids.shuffle.partitionPack.enabled`.
Both modified Spark and JNI artifacts are required. The endpoint is RAPIDS GPU shuffle
with cuDF packed tables, including its existing compression codecs and spill/transport format.
For a controlled A/B, retain the same GPU shuffle manager and settings in both variants.
Set `spark.rapids.sql.agg.singlePassPartialSortEnabled=false` in both variants: this prototype
excludes aggregates eligible for that alternate execution path, even if they eventually
choose hashing at runtime. Then toggle only `spark.rapids.shuffle.partialAggPack.enabled`.

## Scope and data flow

The first implementation supports an adjacent GPU hash partial aggregate and hash exchange,
with one INT32/INT64 grouping key, default Spark Murmur3 hashing, and canonical buffer output.
Supported buffers are COUNT, non-ANSI INT32/INT64 SUM, and INT64 MIN/MAX. Computed grouping
and aggregate input expressions still use the existing preprocessing implementation.
DISTINCT, filtered aggregates, floating/decimal buffers, reduction-only aggregates, multiple
keys, sorted single-pass aggregation, GPU Kudo serialization, CPU shuffle, and LoRE dumping
use existing paths. The CPU Kudo serializer setting does not exclude the RAPIDS GPU
packed-table endpoint, which does not use that serializer.

The existing first pass and its preprocessing, early neighbour consolidation, spill handling,
and recursive hash repartitioning remain. The last neighbour consolidation is concatenated
without aggregation, and the terminal bucket merge is deferred to the shuffle consumer:

```text
OFF: canonical partial buffers -> cuDF merge -> output table -> partition + pack
 ON: canonical partial buffers -> hash state -> final per-partition packed buffers
```

Native open addressing stores immutable representative input row indices, INT64 accumulators,
and validity flags. Occupied slots receive positions inside their Spark hash partitions.
One output kernel writes keys, accumulators and validity directly into final packed allocations.
No compact aggregate output table or reordered payload table is constructed. Null grouping keys
use Spark's seed 42; SUM uses unsigned atomic addition to preserve non-ANSI wrapping. COUNT
merges counts, rather than counting buffer rows. All-null SUM/MIN/MAX remain NULL.

The fused terminal pass can reaggregate buffers that are already unique. The hash state uses
capacity rounded to a power of two at a maximum 50% load, so its additional allocation demand
can exceed cuDF's merge path for some shapes. Native merge inputs are limited to 2^28 rows;
larger inputs request split-and-retry before entering native code.
The switch remains off by default; performance must be measured for each target workload.
OOM retries rebuild state from the spillable input; splitting a retry may emit multiple partial
states for a group, which the downstream aggregate merges. No aggregate state survives an attempt.

This removes the terminal merge-output materialization and its reads for shuffle preparation.
It does not fuse the first aggregate pass, scan/filter processing, compression or transport.
It does not guarantee fewer aggregate updates or lower total memory use, and does not establish
end-to-end query speedup.

Compressed shuffle also required an ownership fix in `RapidsCachingWriter`: its input
iterator owns the incoming batch, while `addCompressedBatch` consumes a reference. The
writer now passes an incremented reference. The copy-compression OFF query reproduced
premature buffer closure before this fix; CPU/OFF/ON pass with all four codecs afterwards.
The aggregate-time metric includes the fused merge and pack together, and the shuffle
partition-time metric also measures this combined preparation; do not sum these metrics
as independent GPU service times.

## Verification and A/B

`PartialAggregatePackSuite` checks CPU Spark partition IDs and CPU aggregate values for both key
widths, negative/null keys, wrapping sums, all-null groups, empty partitions, and injected retry
OOMs with allocation leak checks. The native PARTITION_PACK test also checks sliced input.
`PartialAggregateShuffleSuite` compares CPU/OFF/ON across multiple input batches, AQE,
none/copy/lz4/zstd compression, empty input and unsupported aggregate/Kudo fallback.

On 2026-10-11, the Spark 3.5.1 `sql-plugin,tests` reactor completed `verify -DskipTests`
successfully, including Scala style checks and test compilation. GPU test execution was
separate: 83 tests passed across the partial-pack, partial-shuffle, partition-pack,
partitioning, GPU Kudo and aggregate retry suites. This includes maximum-load collision
probing, INT64 key extrema, retry allocation cleanup, and split-state merging. Both native
PARTITION_PACK tests passed, including sliced nullable buffers and wrapping sums.

`PartialAggregatePackBenchmark rows distinctKeys partitions iterations` measures resident
canonical buffer terminal merge plus shuffle preparation. Its reference already enables
partition+pack, isolating the incremental merge fusion. Five warmups and alternating variant
order are used; setup, first-pass aggregation and transport are excluded. It verifies output
row counts; the suites establish semantic correctness.

Local NVIDIA L4 results below use one non-null INT32 key, one INT64 SUM buffer, 32 partitions,
no compression, five warmups and 50 alternating samples per variant. Peak allocation delta
excludes the resident input and is RMM tracked allocation demand, rather than process memory.

| Buffer rows | Distinct keys | OFF median ms | ON median ms | Time change | OFF/ON peak delta MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 65,536 | 2.751 | 1.946 | -29.3% | 24.6 / 40.8 |
| 4,194,304 | 262,144 | 2.986 | 5.766 | +93.1% | 98.1 / 163.0 |
| 4,194,304 | 4,194,304 | 6.949 | 12.381 | +78.2% | 144.0 / 208.5 |

Six additional 15-sample shapes/runs are retained in the evidence directory. Smaller inputs
showed 11%-30% lower median time; both large-input regressions persisted in the longer runs.
These are local microbenchmarks with run-to-run variation, without confidence intervals,
kernel-count profiling or distributed TPC-DS validation. The row-based hash capacity and
full-capacity initialization/traversal are plausible contributors to the regressions;
this attribution has not been established by a profiler. Reducing state size and avoiding
unnecessary reaggregation of unique buffers are follow-up optimization targets. Keep the
switch off by default and evaluate actual query workloads before enabling it broadly.

Build logs, test logs, all benchmark samples, artifact hashes and source fingerprints are
in `.cache/partial-aggregate-pack/2026-10-11/`. Validation used uncommitted source changes;
embedded JNI revision metadata identifies the baseline commit, while the manifest fingerprints
identify the modified sources used for validation. This build validates the plugin modules;
it does not produce a new distribution archive.
