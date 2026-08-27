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

`formatAgentMessages` can derive an optional Compaction Semantic Index while
its existing Model Context Reconstruction analysis walks persisted assistant
content. It returns the index beside provider messages, summary, and token
metadata so the host forwards one projection into `AgentInputs` without a
separate payload pass. Each entry names a persisted source message and content
part. Tool entries also name their tool call, and reasoning labels name their
reasoning step. Entries carry a monotonic revision, committed/pending lifecycle,
and a redaction bit.

Derivation recognizes settled tool outcomes, visible reasoning labels, and
activity-phase labels from their explicit persisted fields. A tool `intent`
field is admitted only when the host supplies that tool's name in the enabled
semantic-label set; the SDK never guesses whether an arbitrary business
`intent` argument is a label. Missing stable message identity fails closed.
The option is disabled by default and allocates no index array when absent.

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

This first interface derives and snapshots labels present during Model Context
Reconstruction, before AgentContext construction. Same-run label ingestion
would require a separate late-bound resolver or append interface with explicit
timing and failure semantics.

## Consequences

- Derived index data is bounded during Model Context Reconstruction and
  snapshotted once at AgentContext construction. Enabling derivation adds no
  second persisted-content traversal and serializes nothing into ordinary
  model requests; appendix rendering occurs only when compaction fires.
- Hosts can improve checkpoint navigation using semantic work already paid for,
  while the SDK keeps generated labels advisory and bounded.
- LibreChat owns lifecycle, enablement, tool-intent classification, forwarding,
  and rollout policy. The formatter owns extraction because it already assigns
  the exact source coordinates the index requires. This decision neither
  generates labels nor changes application data.
- Semantic replacement, range selection, durable compaction pins, and hidden
  reasoning remain outside this interface.
