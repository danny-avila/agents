# Provider-tool derivation benchmark

The graph used to make two universal passes before provider-specific request
shaping: first to drop incomplete streamed text blocks, then to bound every
tool-call input representation. The new provider-tool derivation performs both
operations in one pass and makes one clone when an assistant message needs both
repairs.

Run it with:

```sh
npm run bench:provider-derivation
```

Result on 2026-08-24, median of seven alternating samples with 250 derivations
per sample:

| Scenario          | Sequential passes | One-pass derivation | Relative speed |
| ----------------- | ----------------: | ------------------: | -------------: |
| 500 text messages |           0.35 ms |             0.20 ms |          1.69x |
| 100 tool turns    |        1452.26 ms |          1367.12 ms |          1.06x |

The tool case is the decision metric. Its 5.9% CPU reduction is modest because
safe serialization and truncation of tool arguments still dominate the work;
the benchmark verifies the provider-visible serialized messages are identical
before timing either path.
