# Hash partition + packed-table fusion prototype

Enable the independent, default-off switch on a query/session before constructing its plan:

```text
spark.rapids.shuffle.partitionPack.enabled=true
```

The switch applies to hash partitioning with the RAPIDS GPU shuffle packed-table endpoint.
It supports flat fixed-width columns, including decimal128, and strings with INT32 offsets.
Nested columns, CPU shuffle, and GPU Kudo serialization retain their existing paths.
The existing packed-table compression codecs remain supported. Both the Spark plugin and
the matching modified `cudf-spark-jni` artifact are required.

## Data flow

```text
OFF: Spark hash/pmod → partition the payload table → contiguousSplit → packed buffers
 ON: Spark hash/pmod → partition row indices → gather payload into final packed buffers
```

The native prototype reuses cuDF partitioning on one INT32 row-index column. It materializes
integer control data but never a fully reordered payload table. It then computes string-length
prefixes where needed, allocates each final partition buffer, and gathers data and validity
directly into those buffers. cuDF `pack_metadata` describes the final layout; the shuffle writer,
reader, compression and spill formats are unchanged. Spark still computes partition IDs,
including its empty-key seed and Hive hash behavior. Row order within a partition is unspecified.

For payload size S, the intended saving is one payload-table write and the subsequent read
for packing, approximately 2S bytes. Row indices, string-length scans, metadata transfers,
hashing, compression, and output allocations remain. This does not fuse partial aggregation
or reduce shuffle payload size. It also does not provide scan/filter/aggregate fusion.

## Validation and reproduction

`PartitionPackSuite` compares partition IDs with CPU Murmur3/Hive hashing and validates every
output column against rows gathered from the original input. It covers NULL, empty and Unicode
strings, all-null strings, booleans, decimal128, empty keys, empty partitions, zero rows and
delayed injected retry OOMs. `GpuPartitioningSuite` compares the fused and reference paths
with none/copy/lz4/zstd compression. `GpuKudoWritePartitioningSuite` exercises Kudo split retry
with the new flag enabled. The native `PARTITION_PACK` target checks sliced nullable payloads
and partition IDs with non-zero offsets.

`PartitionPackBenchmark` accepts:

```text
rows partitions stringBytes payloadLongColumns iterations [rmmCsv] [OFF|ON|AB]
```

`stringBytes=0` selects an INT32 key; otherwise the key is a unique string of that width.
Payload columns are INT64. It uses resident inputs, CUDA_ASYNC, five warmups, alternating
variant order and the same production `columnarEvalAny` routing and packed-table endpoint.
GPU-shuffle selection is forced in this local harness because it does not install a distributed
shuffle manager. Source construction and transfers are excluded. Timing includes hashing,
partitioning, packing, synchronization and closing the outputs. It checks output row counts;
semantic correctness is established by the suites above. This is a local preparation benchmark,
not an end-to-end shuffle or TPC-DS result.

For Nsight, use OFF and ON in separate processes with one measured iteration and
`--capture-range=cudaProfilerApi`; warmups are excluded from capture. RMM logs and
`peakDelta` measure allocation demand, not physical device bandwidth.

Local build uses Spark 3.5.1 (`-Dbuildver=351`) and JNI 26.08.1-SNAPSHOT. Incremental JNI
builds use the repository Docker wrapper, then the Spark sql-plugin/tests reactor compiles
against the installed JNI artifact. No POM version changes are needed.

## Local results, 2026-10-11

NVIDIA L4, driver 595.71.05, CUDA 12.9.1, JDK 17, CUDA_ASYNC with a 1 GiB initial pool.
Five warmups and 15 measured iterations, unique keys and no NULLs, uncompressed packed tables.
The final two large cases were repeated after all compilation finished, with no concurrent
GPU work. Small-case wall times were collected during compilation and are provisional.

| Rows | Partitions | Key | INT64 payload columns | OFF ms | ON ms | Time saved |
|---:|---:|---|---:|---:|---:|---:|
| 131,072 | 32 | INT32 | 2 | 1.267 | 1.018 | 20% |
| 131,072 | 32 | 32-byte string | 2 | 2.173 | 1.528 | 30% |
| 4,194,304 | 32 | INT32 | 8 | 10.043 | 5.708 | 43% |
| 4,194,304 | 32 | 32-byte string | 2 | 10.636 | 7.268 | 32% |
| 131,072 | 1 | 32-byte string | 2 | 1.586 | 1.344 | 15% |
| 131,072 | 200 | 32-byte string | 2 | 5.712 | 3.657 | 36% |

For the large fixed-width case, additional peak live RMM allocation fell from 570.5 MB to
335.6 MB. For the large string case it fell from 436.2 MB to 302.0 MB. These deltas exclude
the resident input and are not physical CUDA pool reservation measurements. Both large inputs
exceed GPU cache capacity. These results describe shuffle preparation only; aggregate and
transport costs can substantially dilute the end-to-end benefit.

An initial string implementation regressed by 24%. Nsight attributed 8.19 ms to a per-thread
character-copy loop. The final implementation separates offset/mask generation from a warp
character copy and uses width-sized stores for fixed-width values. For the large string case,
the measured range uses 17 kernels versus 30 with the switch OFF. Total GPU kernel time is
6.35 ms versus 8.96 ms. Character copying is 2.46 ms, plus 0.11 ms for offset/mask generation.
Memcpy counters do not include memory traffic inside these kernels.

Validation completed: 47 GPU Scala tests across the three suites above, the native sliced-input
test, strict Scala compilation, and the Spark 3.5.1 sql-plugin/tests Maven verify reactor,
including scalastyle. Maven used `-DskipTests`; the GPU suites were executed separately.
The modified Spark sources are shared across Spark versions; this build validates 3.5.1.
Distributed TPC-DS has not been run.

Evidence is retained under `.cache/partition-pack/2026-10-11/`: raw measurements, Nsight
reports, build/test logs and a source fingerprint manifest. `initial/` retains the pre-optimization
measurements; `*-during-build.log` retains the large cases before the idle-host repeat.
The installed JNI classifier JAR SHA-256 is
`0d85177205b445dce540df69551bb851284acd1a0bc9bd0801985bfe05a9f935`.
Added JNI kernels were built for CUDA architectures 80 and 89, including PTX.
The required JNI source commit is `192b1c77b8f16e0968e91c16c483a9401aaf4f50`.
The local JAR above was built before the implementation commits; its source fingerprint
is recorded in the evidence manifest.
