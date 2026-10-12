# GPU Filter + Project fusion

Set `spark.rapids.sql.filterProjectFusion.enabled=true` to enable fusion. The default is
`false`, so A/B runs can use the same query and settings with only this option changed.
Check the executed plan for `GpuFilterProjectExec` to confirm that a query actually fused.

The first implementation fuses adjacent GPU `Project(Filter(input))` operators. It does
not move predicates across other operators. When unused input columns can be omitted,
the predicate produces an internal selection of row indices. Only columns needed by
computed projections and direct output columns
are gathered; downstream operators receive ordinary compact columnar batches. An
all-true predicate avoids gathering. When every input column is needed by the projection,
the operator filters these columns directly, avoiding an extra index column. Tiered
projections also use a direct projection fast path when all rows pass. This reduces
intermediate materialization on wide
inputs with narrow outputs. It does not guarantee fewer kernels for every query.

Eligibility requires deterministic expressions without side effects and a closed
whitelist: integer/boolean/decimal references and literals, comparisons, logical operations,
null checks, and safe integer addition, subtraction, and multiplication. Decimal computations
add non-ANSI addition, subtraction, multiplication and division, their precision/overflow
wrappers, and non-ANSI casts between decimals and integral types or between decimal types.
Existing scale alignment, rounding and null-on-overflow/divide-by-zero behavior is preserved.
Any data type can be projected as a direct reference, including strings and nested payloads.
Throwing ANSI arithmetic, ANSI decimal casts, other casts, non-decimal division, UDFs, and
nondeterministic expressions remain on the existing path. Projects that disable splitting
are also left unchanged.

Fusion retains the projection backend selected by `spark.rapids.sql.projectAstEnabled`.
Expression binding happens once on the driver; AST compilation remains partition-local.
Decimal expressions use the existing tiered GPU evaluator rather than AST. With
`spark.rapids.sql.projectPartialAstEnabled=true`, eligible integer outputs can still use
partial AST alongside decimal outputs. This extension reduces filter/project materialization;
it does not combine decimal arithmetic kernels or remove their required intermediate columns.
OOM retries and row splitting restart the complete fused operation from the original
spillable input. Results, row order, partitioning, and ordering follow the existing
operators; batch sizes and counts may change. The fused node reports input/output rows
and batches, operation time, and the number of unique computed-dependency and direct
output columns processed per successful output batch. These column counts are not
allocation-byte measurements. Set `spark.rapids.sql.metrics.level=DEBUG` to expose
these metrics for profiling.

## Validation and A/B profiling

`FilterProjectFusionSuite` compares CPU, GPU with fusion disabled, and GPU with fusion
enabled and checks physical plans. `FilterProjectRetrySuite` covers retry/split OOM,
spilled inputs, task abandonment, duplicate references, and empty dependencies.
`FilterProjectDecimalSuite` adds CPU/OFF/ON checks for DECIMAL32/64/128, nullable predicates,
different scales, arithmetic, rounding, overflow, zero divisors, CSE, precision-loss settings,
partial AST interaction and ANSI fallback. `FilterProjectDecimalRetrySuite` covers retry/split
OOMs, failures after decimal outputs have been allocated, and restoring spilled decimal inputs.

`com.nvidia.spark.rapids.FilterProjectFusionBenchmark` is a manual operator benchmark in
the tests module. Run it using the Spark 3.5.1 test runtime classpath. Its arguments are:

```text
enabled passPercent inputColumns arithmeticDepth rmmLogFile
false   50          32           1               /tmp/fusion-off-rmm.csv
true    50          32           1               /tmp/fusion-on-rmm.csv
```

The benchmark creates identical one-million-row integer inputs, warms up five times,
then measures twenty iterations with GPU synchronization. It prints median operation
time and the scoped peak RMM bytes above resident input. The RMM log records actual
allocation requests. Use `-` as the log argument to disable allocation logging for timing
runs, and measure timings without a profiler attached. Use pass rates of 10, 50, and
90 percent and increase arithmetic
depth for a compute-heavy projection. Alternate off/on processes and repeat runs.

For kernel counts and CUDA copies, wrap each process with Nsight Systems:

```text
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/fusion-off java <test-runtime-options> \
  com.nvidia.spark.rapids.FilterProjectFusionBenchmark \
  false 50 32 1 /tmp/fusion-off-rmm.csv
```

The profiler range covers measured iterations only. Compare allocation requests,
peak bytes, kernel counts, CUDA copies, and batch distributions separately. CUDA memcpy
statistics do not include data movement performed by gather/filter kernels. This
microbenchmark excludes scan pruning, task scheduling, and retry overhead; also run
representative end-to-end queries and confirm their executed plans. Enabling fusion
for all eligible plans currently has no cost model, so measure narrow inputs and high
pass rates before enabling it for a workload.

## Initial measurements

Measured on an NVIDIA L4 with Spark 3.5.1 dependencies, the tiered backend, and the
unpooled `CUDA_DEFAULT` RMM allocator. Inputs contain 1,048,576 rows and 32 integer
columns; outputs contain one direct column and one computed column. Timings below
are the median of three alternating A/B rounds, each with five warmup and twenty
synchronized measured iterations. Timing runs used no profiler or allocation log and
ran after the build finished. Kernel counts cover twenty measured iterations. Allocation
requests come from RMM logs, excluding input creation and averaged over the five warmup
plus twenty measured iterations. Peak memory is the scoped RMM maximum above resident
input, rather than total device memory.

| Passing rows | Arithmetic depth | Median ms off / on | Peak MiB off / on | Allocation requests MiB off / on | Kernels per batch off / on |
| --- | --- | --- | --- | --- | --- |
| 10% | 1 | 2.296 / 1.277 | 17.80 / 9.40 | 18.21 / 10.61 | 38 / 10 |
| 50% | 1 | 6.450 / 1.759 | 69.00 / 11.00 | 71.01 / 17.01 | 38 / 10 |
| 90% | 1 | 8.008 / 1.902 | 120.20 / 15.40 | 123.81 / 23.41 | 38 / 10 |
| 50% | 8 | 9.297 / 3.453 | 69.00 / 11.00 | 85.01 / 31.01 | 45 / 17 |

An initial implementation increased two-column input costs by constructing an unnecessary
index column. The all-inputs-needed fast path now uses direct filtering: the two-column,
50% passing control measured 1.380 / 1.222 ms off / on, with 9 MiB peak extra memory
in both paths. The all-true tiered path projects directly: the 32-column, 100% passing
control measured 0.777 / 0.752 ms with 4 MiB peak extra memory in both paths. These
small timing differences should be interpreted with run-to-run noise in mind.
The feature remains disabled by default and still needs
workload-specific A/B evaluation before broad enablement.

CUDA memcpy traces contain small scalar transfers and no bulk device-to-device copies
in either path. Simple fused projections increased host-to-device transfers from 13 to
18 and device-to-host transfers from 8 to 10 per batch. The materialization savings are
in filter/gather kernels and their allocation requests, so reduced CUDA memcpy volume
is not an observed benefit. This operator benchmark also excludes coalescing and uses
already materialized GPU inputs; Spark can prune unused scan columns. It establishes
operator-level savings for the stated input shape, without claiming end-to-end Spark
speedups or performance with a pooled allocator.
