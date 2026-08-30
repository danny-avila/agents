# ADR 0008: Evolve Compaction Guidance from Warm-Turn Deltas

## Status

Accepted

## Context

ADR 0007 established a bounded Compaction Semantic Index derived while Model
Context Reconstruction formats persisted history. A host can persist and
replay that exact snapshot across Event Actor and human-in-the-loop
continuations. Labels created by later warm turns are not present in the
construction-time snapshot, however. Rebuilding the index from the complete
conversation after every continuation would add an unbounded history scan and
duplicate the formatter's semantic extraction policy in the host.

A public mutable collector would avoid the scan but expose retention rings,
identity hashes, and cursor state as a serialized compatibility surface. A
separate public entry merge would still require the host to construct trusted
source coordinates or perform another message pass.

## Decision

The existing `formatAgentMessages` compaction option accepts an optional
`baseSnapshot`. This serializable envelope contains the retained entries and
their cumulative producer-side entry count. Before formatting the current
payload, the SDK snapshots and validates this caller-owned state. Valid prior
entries enter the private coverage-balanced collector in chronological order;
semantic entries derived from the current payload follow during the
formatter's existing content pass. The returned
`compactionSemanticIndexSnapshot` is the evolved bounded envelope; the
existing `compactionSemanticIndex` result remains its entries-only projection.

The public interface exposes no mutable accumulator or retention internals.
The prior snapshot remains subject to ADR 0007's 256-entry input bound. An
oversized or invalid prior snapshot fails closed, while valid guidance from the
current payload is still derived. Newer revisions and tombstones in the delta
renew their logical identity's recency and prevent stale labels from
resurfacing once balanced admission is active.

Evolution costs `O(B + delta)`, where `B` is the bounded prior snapshot and
`delta` is the current payload. It is therefore independent of unbounded
conversation length, but is not described as pure `O(delta)`. When semantic
derivation is disabled, the formatter does not snapshot, allocate, or scan an
index.

The host remains responsible for deciding when a continuation is settled and
durable, supplying the last committed snapshot, and persisting the evolved
result through its existing compare-and-set continuation seam. Provider
messages, compaction range selection, cache checkpoints, graph state, and
Langfuse trace shaping are unchanged.

## Consequences

- Warm continuations can add tool intents, outcomes, activity phases, and
  reasoning labels without rescanning historical messages.
- Formatter ownership keeps source-coordinate extraction and retention policy
  local to the module that already has the necessary context.
- Persisted snapshots remain plain entries plus one cumulative count and can
  cross process or actor boundaries without serializing SDK implementation
  details. The count preserves omission telemetry across JSON persistence.
- LibreChat needs a focused adoption change after the SDK version containing
  this interface is published.
