# GPU Expand + Aggregate fusion

Set `spark.rapids.sql.expandAggregateFusion.enabled=true` to enable fusion; the default
is `false`. This option is independent of `spark.rapids.sql.filterProjectFusion.enabled`.
Confirm eligibility using `GpuExpandAggregateExec` in the executed plan. Enable
`spark.rapids.sql.explain=ALL` for skipped-fusion reasons.

The implementation fuses adjacent GPU Expand and Partial HashAggregate for
simple grouping sets, rollup, and cube. It aggregates each branch against the original
spillable input, omitting branch-constant grouping columns from the input-sized group-by
projection. It restores constants, substituted NULL keys, and the discriminator on the
smaller aggregate buffers, then uses the existing full hash merge, spill/repartition,
final projection, shuffle, and final aggregate machinery. No JNI/CUDA changes are required.

Fusion removes the `GpuCoalesceBatches` that `GpuExpandExec.coalesceAfter` places between
Expand and the partial aggregate, so the fused node requests the same target batch size on
its own input through `childrenCoalesceGoal`. This batches the unexpanded input, which is a
factor of `branches` smaller than what the unfused plan coalesces. `optimizeCoalesce` then
drops the inserted node whenever the child's `outputBatching` already satisfies the goal, so
a child that already emits target-size batches costs nothing, and concatenation of a single
batch is a passthrough in any case.

This affects throughput, not results: without the goal the first pass sees whatever batches
the child emits, each branch group-by reduces poorly, and the resulting partials push the
work into the merge pass, which hashes the data a second time.

Within an input batch, later branches can merge a retained earlier branch's partial buffers
instead of scanning the original rows. This requires identical substituted aggregate
functions and every live target key to be present in the same source output position.
Only a complete, unsplit branch that reduces row count by at least twofold is retained;
at most one branch is cached, and a smaller compatible result can replace it. Incompatible
arguments (including branch-specific NULLs), incompatible key layouts, and uncompressed
results continue to aggregate the original input. COUNT merges sum existing counts; they never
count partial rows. Integral SUM, MIN, and MAX use their existing merge implementations.

When a partition has exactly one nonempty input batch and every branch emits one unsplit
result, the unique branch discriminators prove that the restored keys are disjoint.
The merge path can concatenate these results without another group-by. Multiple input
batches and split results retain normal merge aggregation. This preserves fully aggregated
output even when skipping partial-output aggregation is disabled.

This remains a sequence of cuDF calls, not one native call for all grouping sets. Existing
aggregate preprocessing still allocates columns when necessary. Direct-reference Expand
columns already share input buffers through reference counting, so logical expanded bytes
are not equivalent to physical writes. Eliminated traffic depends on literal/expression
materialization and subsequent coalescing; aggregation still reads required input values.

## Eligibility and semantics

- Expand entries must be direct input references or typed literals. Non-grouping payloads
  must be identical across branches, excluding multi-distinct rewrite shapes.
- Every branch must have a structurally unique tuple of literal keys actually referenced
  by the aggregate grouping expressions. Column names are not used to identify grouping
  IDs. Duplicate grouping sets can fuse when Spark supplies a unique grouping-position
  key; ambiguous or pruned discriminators retain the original plan.
- Grouping keys support boolean, integer, date, timestamp, and string. Floating-point,
  decimal, binary, and nested grouping keys remain on the existing path.
- Aggregates support non-distinct COUNT with one supported scalar argument, integer SUM,
  and integer MIN/MAX. Arguments must be direct references or literals. Aggregate FILTER,
  computed arguments, AVG, collections, and throwing ANSI COUNT/SUM are excluded.
- Only pure Partial mode without required child distribution is eligible. Single-pass
  sort strategies (`allowSinglePassAgg` or `forceSinglePassAgg`) retain the original
  operator. The full hash merge output-skipping configuration remains effective.

A branch with only constant keys uses reduction on nonempty input. It skips zero-row
batches and empty partitions, preserving grouped aggregate semantics rather than
manufacturing an empty-input grand total. Actual NULL keys and substituted NULL keys
remain distinct through the restored discriminator. Non-ANSI integral SUM preserves
wrapping behavior. Output attributes and unknown partitioning follow the original stage;
output order and batch sizes/counts are not guaranteed for hash aggregation.

The original batch and any cached partial stay spillable until all branches finish.
Each attempt covers preprocessing, aggregation, postprocessing, and restoration. OOM
retries start from the current original rows or compatible complete partial buffers,
and splitting emits valid partial buffers. Successful split
pieces are committed once; neither later splits nor later branches replay them.
Unexpected exceptions are propagated. Early iterator/task completion releases retained
input and pending split pieces.

The fused node retains aggregate metrics and adds input rows, input batches, and
`branchBatches` (successful branch partial batches, including successful split pieces),
`reusedBranchBatches` (successful batches computed from partial buffers), and
`branchInputRows` (rows supplied to branch aggregation before retry splitting).
Set `spark.rapids.sql.metrics.level=DEBUG` for these metrics. They do not measure allocated
bytes; use RMM logs and CUDA profiling for allocation/kernel evidence.

## Validation and A/B

`ExpandAggregateFusionSuite` checks CPU/off/on equality and actual fusion plans for
nullable grouping keys and measures, duplicate sets, branch-specific NULL aggregate
arguments, integer widening/overflow, boolean/date/timestamp/string keys, empty input,
AQE, both fusion-switch combinations, and fallback eligibility.
`ExpandAggregateRetrySuite` covers native retry/split OOM before and after branch progress,
late postprocess/reduction failures on original and reused inputs, all-null reductions,
compression guards, safe merge bypass, multiple inputs, spill restoration, early close,
task abandonment with pending splits, metrics, and unexpected-error cleanup.

`com.nvidia.spark.rapids.ExpandAggregateFusionBenchmark` runs a local full partial-stage
benchmark using the Spark 3.5.1 test runtime classpath. Arguments are:

```text
enabled branches cardinality inputColumns rmmLogFile
false   4        16          8            /tmp/expand-off-rmm.csv
true    4        16          8            /tmp/expand-on-rmm.csv
```

An optional sixth argument specifies the number of repeated input batches (default 1).
Use multiple batches to exercise cross-batch merge rather than the single-input shortcut.
The printed branch row count and reused batch count expose whether partial reuse occurred.

It creates 131,072 GPU-resident rows, `branches - 1` correlated integer dimensions,
a constant long measure, and rollup branches. Each run performs COUNT and SUM. The
baseline includes Expand projections and target-size coalescing, followed by the existing
partial aggregate and merge. Fusion uses branch aggregation, compatible partial reuse,
and the same merge/final-pass machinery with the disjoint-key shortcut. Unused resident
columns are omitted from Expand in both paths.

Use 2/4/8 branches, low and high key cardinalities, and narrow/wide resident inputs.
Cardinality must divide the row count; the maximum is 131,072. Each run warms up 30 times
and measures 20 synchronized iterations. The earlier 3-warmup setting left this path far
short of steady state: repeated runs of one configuration ranged from 11.7 ms to 32.5 ms,
enough to hide or invent a regression. Use `-` to disable RMM logging for timing runs;
alternate off/on order and repeat without other compilation or profiling activity.

For CUDA kernel/copy counts, wrap the Java process with:

```text
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/expand-off java <test-runtime-options> \
  com.nvidia.spark.rapids.ExpandAggregateFusionBenchmark \
  false 4 16 8 /tmp/expand-off-rmm.csv
```

The capture range includes measured iterations only. RMM logs include setup and warmup,
so remove setup allocations and account for the warmup iterations separately.
The printed peak is the scoped maximum allocation above the resident-input/setup baseline.

This benchmark excludes scan pruning, network shuffle, scheduling, final distributed
aggregation, and failure injection. It uses the unpooled `CUDA_DEFAULT` allocator, and
correlated dimensions rather than a general TPC-DS distribution. Representative
end-to-end TPC-DS A/B on the distributed cluster remains a later validation step.

## Measurements

Profiling on an NVIDIA L4 found that fusion reduces input-sized materialization for
grouping-set aggregates. The default-off flag has no cost model to select fusion versus
the original path; the internal twofold compression guard controls partial reuse only.
Enabling fusion does not promise faster execution on every eligible plan.

The tables use 131,072 rows per input batch, eight resident input columns, the tiered
backend, and an unpooled allocator. Each cell lists the medians of three alternating
off/on rounds, each round 30 warmups plus 20 synchronized measured iterations, run with
no profiler or RMM log. These are local observations, not workload-level speedup
guarantees. Scoped peak bytes exclude setup and resident input.

Single input batch:

| Branches | Key cardinality | Median ms off / on | Peak extra MiB off / on |
| --- | --- | --- | --- |
| 2 | 16 | 4.83 5.89 6.24 / 4.35 4.44 4.01 | 10 / 2 |
| 4 | 16 | 6.45 7.15 6.97 / 5.57 5.01 | 24 / 2 |
| 8 | 16 | 11.42 11.39 11.38 / 9.10 7.78 | 68 / 3 |
| 4 | 131,072 | 8.76 8.20 8.61 / 7.81 | 32 / 26 |

Four repeated input batches (524,288 total rows, identical values so cross-batch keys
overlap), which exercise the merge boundary rather than the single-input shortcut:

| Branches | Key cardinality | Median ms off / on | Peak extra MiB off / on |
| --- | --- | --- | --- |
| 2 | 16 | 9.55 8.53 8.29 / 5.50 5.87 6.37 | 40 / 28 |
| 4 | 16 | 13.80 13.95 14.58 / 6.80 7.37 7.18 | 96 / 28 |
| 8 | 16 | 38.31 40.88 38.54 / 9.76 9.94 10.10 | 359 / 30 |
| 4 | 131,072 | 17.04 15.25 15.53 / 9.84 9.80 9.92 | 117 / 33 |

Every measured configuration improves, and the peak-memory reduction grows with the branch
count because the expanded batch is never materialized. The eight-branch multi-batch case
is the clearest: the unfused plan coalesces 1,048,576 expanded rows before aggregating,
while fusion aggregates eight times against the 524,288-row unexpanded input.

An earlier revision of this document reported the high-cardinality multi-batch case as
about 60% slower. That regression came from the fused node not declaring
`childrenCoalesceGoal`: it inherited whatever batches the child emitted instead of the
target-size batching the unfused plan gets from `GpuExpandExec.coalesceAfter`. Each branch
group-by then reduced poorly and the merge pass hashed the data a second time. With the
batching contract restored, that configuration is faster rather than slower, and the
single-input disjoint-key shortcut now applies to multi-batch partitions too, because the
batches are coalesced before the first pass sees them.

Some rows above list only two of three rounds. Those JVM launches produced no result line
during an unattended sweep that captured only each process's last line of output, so the
error was not retained. The affected configurations were re-run afterwards, including under
artificial CPU load, without a single failure in 17 consecutive launches; the cause was not
identified and should be re-checked if it recurs with full output captured.
