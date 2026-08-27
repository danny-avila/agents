# ADR 0007: Guide Compaction with a Bounded Semantic Index

## Status

Accepted

## Context

Compaction currently receives the complete raw Compaction Range and a detailed
checkpoint instruction. LibreChat already pays to produce lean, user-visible
semantic signals around that history: tool intent and outcome, activity-phase
labels, and reasoning labels. Those signals are stripped from normal provider
messages or live outside the message projection, so the summarizer cannot use
them to find the most important evidence in a tool-heavy history.

Copying every label into every model request would add latency and tokens to the
normal hot path. Replacing raw tool evidence with generated labels would make an
advisory model output authoritative and weaken compaction fidelity. Appending
unbounded host text to compaction would also erode the cache-aligned request
prefix and create a prompt-injection and trace-disclosure surface.

## Decision

`AgentInputs` accepts an optional Compaction Semantic Index. Each entry names a
persisted source message and content part. Tool entries also name their tool
call, and reasoning labels name their reasoning step. Entries carry a monotonic
revision, committed/pending lifecycle, and a redaction bit.

The summarization module validates each source identity and exact persisted
content-part index against only the raw messages in the selected Compaction
Range. For each logical identity it selects the highest revision, rejects
conflicting ties, and excludes pending, redacted, empty, malformed, or
out-of-range entries. Oversized newer revisions become bounded, textless
tombstones so they cannot resurrect older guidance. It then orders accepted
entries by exact provenance-contribution order and local identity, escapes them
as data, and enforces fixed per-entry, total-character, and entry-count budgets.

The rendered appendix is placed inside the unique final HumanMessage, before
the compaction instruction. Raw history and its provider cache breakpoint are
unchanged. Raw messages remain authoritative, and no raw evidence is removed.
When the host supplies no index, the ordinary request path and compaction prompt
remain unchanged.

Compaction traces record included-entry, character, and omitted-entry counts,
including entries rejected while the caller-owned input is snapshotted.
The export-time trace shaper redacts the appendix body from observation input.
Self-spawned subagents do not inherit their parent's run-scoped index; explicit
child configurations may provide an index belonging to the child.

This first interface snapshots labels committed before AgentContext
construction. Same-run label ingestion would require a separate late-bound
resolver or append interface with explicit timing and failure semantics.

## Consequences

- Caller-owned index data is bounded and snapshotted once at AgentContext
  construction. Normal model requests perform no index serialization or
  source-message traversal; those costs occur only when compaction fires.
- Hosts can improve checkpoint navigation using semantic work already paid for,
  while the SDK keeps generated labels advisory and bounded.
- A later LibreChat adapter owns extraction, lifecycle, ownership, and rollout
  policy. This decision neither generates labels nor changes application data.
- Semantic replacement, range selection, durable compaction pins, and hidden
  reasoning remain outside this interface.
