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
Cardinality must divide the row count; the maximum is 131,072. Each run warms up three
times and measures ten synchronized iterations. Use `-` to disable RMM logging for timing
runs; alternate off/on order and repeat without other compilation or profiling activity.

For CUDA kernel/copy counts, wrap the Java process with:

```text
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/expand-off java <test-runtime-options> \
  com.nvidia.spark.rapids.ExpandAggregateFusionBenchmark \
  false 4 16 8 /tmp/expand-off-rmm.csv
```

The capture range includes measured iterations only. RMM logs include setup and warmup,
so remove setup allocations and account for the three warmup iterations separately.
The printed peak is the scoped maximum allocation above the resident-input/setup baseline.

This benchmark excludes scan pruning, network shuffle, scheduling, final distributed
aggregation, and failure injection. It uses the unpooled `CUDA_DEFAULT` allocator, and
correlated dimensions rather than a general TPC-DS distribution. Representative
end-to-end TPC-DS A/B on the distributed cluster remains a later validation step.

## Measurements

Profiling on an NVIDIA L4 found that fusion reduces input-sized materialization for
low-cardinality rollups. Partial reuse and the disjoint-key shortcut lower kernel counts
relative to the first fused implementation: 33 to 19 at two branches, 66 to 48 at four,
and 151 to 124 at eight. They remain higher than the coalesced unfused baseline, which
can aggregate branches together. The default-off flag has no cost model to select fusion
versus the original path; the internal twofold compression guard controls partial reuse
only. Enabling fusion does not promise lower kernel counts or faster execution on every
eligible plan.

The table uses 131,072 input rows, eight resident input columns, the tiered backend,
and an unpooled allocator. Timings are the median of three alternating off/on rounds,
each with three warmups and ten synchronized measured iterations, run after compilation
and tests finished with no profiler or RMM log. JIT/warmup variation was substantial in
the first round; these are local observations, not workload-level speedup guarantees.
Nsight kernel counts cover ten measured iterations and are normalized per original input.
Allocation requests exclude verified setup/input creation, averaged over the three warmup
plus ten measured iterations. Scoped peak bytes exclude setup and resident input.

| Branches | Key cardinality | Median ms off / on | Peak extra MiB off / on | Allocation requests MiB off / on | Kernels per input off / on |
| --- | --- | --- | --- | --- | --- |
| 2 | 16 | 4.041 / 1.645 | 10.09 / 2.56 | 12.61 / 2.56 | 17 / 19 |
| 4 | 16 | 5.280 / 3.192 | 24.24 / 2.56 | 29.80 / 2.57 | 23 / 48 |
| 8 | 16 | 9.967 / 6.383 | 68.93 / 3.06 | 80.57 / 3.09 | 38 / 124 |
| 4 | 131,072 | 6.432 / 5.602 | 32.75 / 26.67 | 48.94 / 38.87 | 23 / 51 |

At four branches and cardinality 16, branch aggregation reads 131,120 rows per input
instead of the first implementation's 524,288: one original input plus three 16-row
partials. Three branches reuse buffers. High cardinality reads all 524,288 original
branch rows and reuses no buffers. Its single-input improvement comes from skipping
the unnecessary final merge, not from partial reuse.

The first implementation's high-cardinality single-input case was about 53% slower;
the optimized single-input case above removes that regression in this local benchmark.
However, that shortcut does not apply to multiple input batches. Four repeated input
batches (524,288 total rows, identical values so cross-batch keys overlap) produced:

| Branches | Key cardinality | Median ms off / on | Peak extra MiB off / on |
| --- | --- | --- | --- |
| 2 | 16 | 5.884 / 5.299 | 40.18 / 2.56 |
| 4 | 16 | 11.355 / 9.191 | 96.81 / 2.56 |
| 8 | 16 | 32.280 / 19.522 | 359.02 / 3.07 |
| 4 | 131,072 | 13.112 / 21.017 | 117.31 / 108.62 |

The multi-input low-cardinality cases still benefit, but the high-cardinality control
remains about 60% slower with little peak-memory improvement. Both versions must merge
overlapping keys; fusion still performs more branch calls. An eligible plan is therefore
not necessarily profitable, and the switch remains off by default. These repeated-batch
measurements exercise the merge boundary, not a representative distributed workload.

CUDA memcpy statistics also depend on coalescing and cardinality. At four branches and
cardinality 16, device-to-device copies fell from about 8 MiB to 1.2 KiB per input. At
cardinality 131,072 they increased to about 9 MiB. Eight-branch coalescing moved data
through kernels rather than bulk CUDA memcpy in this profile. These copy counters exclude
data movement performed inside kernels and should not be treated as total GPU traffic.
