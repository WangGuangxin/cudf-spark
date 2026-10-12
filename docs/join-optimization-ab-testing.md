# GPU hash join optimization A/B switches

All four switches default to `true`, preserving the current execution behavior. Set them
before constructing and executing a fresh query. Keep other join, AQE, batch-size and shuffle
settings fixed between variants, and avoid reusing a cached query result or an existing RDD.

| Optimization commit | Configuration | Behavior when false |
| --- | --- | --- |
| `c1fc82924` projected build key cache | `spark.rapids.sql.join.buildKeyCache.enabled` | Reproject build keys for statistics and each stream batch; release probe-local projected keys. |
| `eb16975c2` outer join bitmap | `spark.rapids.sql.join.outerBitmap.enabled` | Track unmatched build rows with a spillable boolean column, filtering gather indices and scattering matched rows. |
| `c7dfbf2be` deferred join chain payload | `spark.rapids.sql.join.chain.enabled` | Execute each join independently and materialize its output. This switch already existed. |
| `5c3dff538` reusable hash state | `spark.rapids.sql.join.hashReuse.enabled` | Build and close hash state for each probe, without retaining an evictable hash handle. Applies to ordinary hash joins and fused chains. |

The key cache and hash state switches are independent. A run can reuse projected keys while
rebuilding hash state, or reproject keys while reusing a hash state for the stable build side.
When hash reuse is disabled, ordinary hash joins also omit the key-buffer copies used to
detach retained hash state from contiguous payload allocations.

Fused chains keep their projected dimension keys as part of chain execution. To measure the
ordinary build-key cache, disable `chain.enabled` in both variants, then toggle only
`buildKeyCache.enabled`. Hash reuse can be measured with chains either enabled or disabled;
keep the chain setting fixed between those variants. Bitmap tracking affects hash outer joins
that track matches on the buffered build side; inner joins do not allocate such a tracker.
The switches preserve current correctness fixes and spill/retry handling. They do not force
a hash join strategy or change physical build-side selection.

For an aggregate A/B of all four optimizations, use this OFF configuration:

```properties
spark.rapids.sql.join.buildKeyCache.enabled=false
spark.rapids.sql.join.outerBitmap.enabled=false
spark.rapids.sql.join.chain.enabled=false
spark.rapids.sql.join.hashReuse.enabled=false
```

Use the same four settings with `true` for ON. For attribution, change one switch at a time
with the dependency above respected. The existing `spark.rapids.sql.join.chain.logDiagnostics`
and join chain SQL metrics can establish whether eligible chains actually executed in a run.

Regression coverage checks independent SQLConf/map configuration propagation, projected-key
evaluation counts, hash-build counts in iterators and chains, no retained state after disabled
probes, duplicate/null keys, both outer tracking implementations, retry, splitting, and
host/disk spill. Query suites compare GPU OFF/ON against CPU across join types and hash join
strategies, and cover mixed chains with AQE and sized joins. These tests establish correctness
and path selection; they do not establish workload performance gains.

On 2026-10-11, 174 GPU tests passed: 79 iterator/resource tests and 95 CPU/GPU comparison
and configuration tests. Logs and source/artifact fingerprints are retained under
`.cache/join-ab-switches/2026-10-11/`. No JNI source change is needed for these switches;
the build reuses the previously installed JNI artifact.

Join chains also support non-unique inner build keys under `chain.enabled`. Each stage
retains the distinct hash fast path for unique keys and uses a regular hash join for
non-unique keys, preserving every matching build payload. Semi stages still deduplicate
build keys. Payloads stay in their original sources until the final gather.

Duplicate builds require an additional group-by count to obtain the maximum multiplicity
of a complete key tuple. Null tuples count only when probe null equality allows matches.
The product of these maxima bounds intermediate index-map growth and drives input splitting;
it is conservative because it does not account for selectivity or correlations between stages.
Nested splits are processed before their pending siblings to bound buffered batch metadata.
A task falls back before probing if even one input row would exceed the map budget, or a
build is oversized. `joinChainFallbackNonUnique` now counts excessive non-unique fanout,
rather than every duplicate build. Keep `chain.enabled` fixed when testing hash reuse,
and compare chain on/off for this extension. Performance still needs workload measurement.
