# Ordered Tool History Projection

Model Context Reconstruction owns source evidence separately from the serving
provider's replay policy. A preparation-scoped, copy-on-write cache shares the
Responses source selection and ordered contributions between tool-less folding
and sealed-turn replay. Complete output takes precedence over streaming sidecars;
streaming positions order sidecars when text mapping is unambiguous.

The projection does not select the serving provider or model. Primary and fallback
preparations each select native replay versus portable folding using the actual
serving model and invocation options. The cache is not persisted or shared between
runs. Plain messages without Responses evidence allocate no projection objects.

Fallback preparation starts from the pruned history before any primary wire
shaping. Each fallback applies its own input limits, thinking normalization,
folding, and measured request projection. A model's call-specific API-mode answer
is authoritative, including `false`; a Responses default cannot override an
explicit Chat selection. Graph regressions exercise both directions and a failed
intermediate fallback, including source provenance and native media.

Primary and fallback paths reuse the serving-policy, artifact projection, tail
cache, and synthetic-context compaction helpers. Fallbacks also restore legacy
formatting and sanitize orphan tool pairs before adding cache markers. Artifact
expansion is retried without the artifact when necessary; oversized synthetic
context is compacted against the final, serving-specific measured payload before
rejecting a fallback. No content is appended after the final budget check.

Regression coverage checks source identity through fallback folding before origin
tracking, mixed model/tool provenance, completed media, mixed native and parsed
calls, bounded nested traversal, per-value argument limits, and source immutability.
This is the first shared consumer slice; compaction and budgeting retain their
existing interfaces rather than changing checkpoint formats in this PR.

## Benchmark

Run `npx tsx src/scripts/bench-tool-history-projection.ts` from the repository root.
The benchmark compares a shared preparation with separate preparation caches for
folding and native replay. Results are medians of five alternating samples, each
containing 100 preparations. It is a cache comparison, not a baseline-release
speedup claim or a general allocation profiler.

Local sample (milliseconds per 100 preparations):

| History | Separate caches | Shared cache | Projection objects per preparation |
| --- | ---: | ---: | ---: |
| 500 plain-text messages | 2.56 | 2.40 | 0 |
| 100 tool-heavy Responses messages | 1150.27 | 1105.02 | 100 |

The plain-text fold also asserts input-array identity. Timings are diagnostic,
not CI thresholds; machine load and runtime warmup affect these small differences.
