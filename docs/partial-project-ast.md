# Partial AST projection

Enable AST evaluation for eligible outputs in a mixed GPU projection with:

```properties
spark.rapids.sql.projectPartialAstEnabled=true
```

The default is `false`. For an A/B comparison, keep
`spark.rapids.sql.projectAstEnabled=false` and change only the new flag.
Keep batch size, tiered projection, combined expressions and filter/project fusion
settings identical between runs.

For example, in a non-ANSI query:

```sql
SELECT (a * 2 + b) * 3, substring(payload, 2, 3), payload
FROM input
```

If `a` and `b` are AST-compatible types, the arithmetic output uses AST;
`substring` uses ordinary GPU evaluation; `payload` remains a column reference.
Eligibility uses the existing expression metadata and requires a fixed-width
result. ANSI overflow checks, unsupported casts, division and unsupported
comparisons retain their existing evaluation.

The implementation applies AST after common subexpression elimination and
binding. Shared expressions still become tier outputs evaluated once, and each
eligible tier output uses its own AST kernel. This does not combine the entire
projection list into one kernel. It does not extract AST islands from inside an
otherwise ineligible output.

`GpuProjectExec` remains the physical operator. Its eligible bound tier expressions
are wrapped in `GpuPartialProjectAst`; output attributes and direct references are
preserved. Native compiled expressions are cached per task and released on task
completion, including early downstream termination. Existing pre-splitting,
OOM retries and nondeterministic checkpoints continue to apply.

On main, the existing `projectAstEnabled` flag selects per-expression
`GpuProjectAstExpression` evaluation inside `GpuProjectExec` and retains precedence.
The new flag controls partial AST evaluation when `projectAstEnabled=false`, without
wrapping the existing AST backend a second time.

Correctness coverage is in `PartialProjectAstSuite` and
`PartialProjectAstRetrySuite`, including CPU/off/on comparisons, mixed string and
nested outputs, shared expressions, ANSI errors and conditional division,
floating special values, seeded nondeterminism, injected retry/split OOMs,
count-only batches and native-state serialization across task completion.

Performance gains have not been established by these correctness tests. Measure
project time, kernel launches, allocation traffic and end-to-end query time on
representative workloads before enabling the flag by default.
