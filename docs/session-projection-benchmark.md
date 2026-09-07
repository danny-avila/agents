# Session projection cache

`AgentSession` now reuses an in-memory index and the active message-entry projection across continuations. The indexed entry count is the append revision; the active leaf identifies the projected branch. Reads index only newly appended log entries, extend the active projection along the new parent chain, or rebuild when the leaf switches branches. A summary discards the preceding projection.

Every read still materializes fresh LangChain messages. Full message materialization remains O(active messages); this change makes **log indexing and path traversal** incremental, not the entire agent turn. `Run` callers that do not use `AgentSession`, including LibreChat's direct `Run` integrations, do not automatically benefit.

## Compatibility

- No persisted format or public signature changes.
- Existing mutable entry access through `AgentSession.getSessionStore()` permanently disables caching for that store and its shared-entry fork family. Subsequent reads use the original full derivation, including mutations made through previously retained references.
- Reopened stores have independent caches. Forks share an ownership flag because their entry objects are shared.
- Each read returns fresh message wrappers and summary objects. Existing nested content/metadata reference semantics are preserved.
- Duplicate IDs in legacy logs fall back to the existing derivation.
- No trace events, provider payloads, token accounting, or summary content are changed.

## Reproduce

```sh
npx tsx src/scripts/bench-session-projection.ts
```

The benchmark compares `deriveMessages(store.getPath())` with `deriveSessionMessages(store)` on the same real JSONL store, asserts output equivalence, and alternates measurement order. Warm timings are medians of seven batches after warm-up. Delta timings are medians of 30 single reads, each after a real appended message. Both include fresh message construction; disk writes, model calls, and network latency are excluded. The generated tool-rich log has three records per original message. Compacted scenarios retain 40 messages after a summary while keeping the older log records.

## Local results, 2026-09-07

Node 24, macOS. Values are milliseconds per projection; they are microbenchmark results, not end-to-end request speedups.

| Original messages | Compacted | Full warm read | Cached warm read | Full delta read | Cached delta read |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 | No | 0.017 | 0.005 | 0.024 | 0.011 |
| 1,000 | No | 0.181 | 0.060 | 0.179 | 0.064 |
| 10,000 | No | 3.369 | 0.756 | 3.466 | 0.946 |
| 100 | Yes | 0.018 | 0.002 | 0.035 | 0.008 |
| 1,000 | Yes | 0.167 | 0.002 | 0.310 | 0.009 |
| 10,000 | Yes | 3.395 | 0.002 | 3.598 | 0.013 |

First cache reads still build an O(log size) index, and the index adds one map entry per log record. Cold-start timings are printed separately but are not claimed as a gain. The cache holds one active projection per store, not one per historical branch. Public mutable-store users deliberately retain baseline performance.
