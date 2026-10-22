# Generate/filter fusion

Apply a filter above an `explode` before the retained child columns are replicated:

```properties
spark.rapids.sql.generateFilterFusion.enabled=true
spark.rapids.sql.generateFilterFusion.minReplicatedBytes=16
```

Both default to off and 16 respectively. For an A/B comparison keep batch size, the other
fusion flags and AQE identical between runs.

## What it changes

`GpuGenerateExec` projects `[retained child columns..., generator input]` and asks cuDF to
explode that table. cuDF replicates every retained column once per generated element, so a row
whose array holds 1000 values copies the whole payload 1000 times. A filter above it then
discards whatever the predicate rejects, after the copying is already paid for.

The fused operator explodes `[row index, generator input]` instead. Replication costs four
bytes per generated row, the predicate runs against that narrow result, and the payload is
gathered once for the rows that survive. Rows the predicate rejects never have their payload
copied.

Split points are still computed on the full width projected batch, so the memory bound is the
one the unfused generator would have used even though the exploded batch is now narrow. The
generator's own retry and splitting are unchanged; a row index survives an internal split
because its values are carried through the explode untouched.

## When it applies

- `explode` and `posexplode` over an array, with or without `outer`.
- The predicate reads only the generator's output. One that also reads a replicated column
  would have to gather that column before it could be evaluated, which is left to the unfused
  path. Spark already pushes a conjunct that reads only replicated columns below the generate,
  so the mixed case is rarer than it looks.
- There is at least one retained child column, and they total at least
  `minReplicatedBytes` per row. Below that the index and the gather cost more than the
  replication they avoid.
- Maps, `GpuReplicateRows` and the fixed length lazy array path keep their existing execution.

Position comes from the explode, before the predicate, and is filtered along with the value, so
it stays the index inside the original array. With `outer`, an empty or null array still
produces one null row, which a value predicate drops and an `IS NULL` predicate keeps, exactly
as the unfused pair does.

## Measurements

Mechanism, measured directly against cuDF with no plan rule
(`GenerateFilterCostBenchmark`, L4, 8M generated rows held constant):

| replicated payload | keep 10% | keep 50% | keep 100% |
| --- | --- | --- | --- |
| one long | 0.99-1.10x | 0.94-1.03x | 0.84-0.95x |
| eight longs and a string | 3.13-3.29x | 1.99-2.13x | 1.41-1.46x |

The array length barely matters once the generated row count is held constant; what the fusion
saves is bytes per surviving row, so the payload width and the share of rows the predicate
rejects are what move it. The win at 100% survival comes from the eager pair paying two full
width passes, the replication and the filter's own copy, where the fused path pays one gather.

End to end through Spark (`GenerateFilterFusionBenchmark`, same machine, op time of just the
operators the fusion replaces):

| rows | k | keep | payload | operator op time | whole query |
| --- | --- | --- | --- | --- | --- |
| 500k | 16 | 10% | 8 longs + string | -44.0% | -2.1% |
| 500k | 16 | 50% | 8 longs + string | -52.4% | -2.9% |
| 500k | 16 | 10% | one long | -42.9% | -0.5% |
| 250k | 32 | 10% | 8 longs + string | -60.8% | -5.3% |

The operator itself is consistently 40-60% faster. The whole query moves much less because the
explode and filter are only part of it; in these runs the surrounding array construction
dominates, and a workload reading arrays from storage would give the operator a larger share.

Note the one disagreement: at cuDF level a single long payload showed no gain, while end to end
it still shows -42.9%, because the fused operator also avoids a second operator's batch
handling. The default threshold follows the more conservative cuDF measurement. It is a tuning
knob, not a correctness boundary, and it is not settled.

These establish the mechanism and path selection. They do not establish a gain on any
production workload.

## Coverage

`GenerateFilterFusionSuite` compares CPU against GPU with fusion off and on for predicates on
the exploded value, on the position, and on null elements; predicates that keep nothing and
everything; `explode_outer` and `posexplode_outer` against both a value predicate and an
`IS NULL` predicate; arrays that are long, short, empty, null and carrying null elements;
multiple output batches at a small target size; and the rejection paths for a predicate that
reads a replicated column, for payloads below the byte threshold, and for maps.
