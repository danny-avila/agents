# ADR 0004: Observe Provider Message Projection Provenance Before Enforcement

- Status: Accepted
- Date: 2026-08-24

## Context

Agent sessions now reconstruct model context from one append-only session log,
but provider-bound messages still pass through several graph, runnable, and
provider projections. Migrating those paths to a strict log-first invariant in
one change would make existing unstamped synthetic and runtime-generated
messages fail without first locating every provenance gap.

The final `handleChatModelStart` callback observes the exact message batch after
runnable-level instruction injection and before provider serialization or I/O.
It is shared by primary, fallback, summarization, and nested agent invocations
that use `attemptInvoke`.

## Decision

Add one opt-in Provider Message Projection Invariant at that callback boundary.

- `off` is the default and adds no callback handler or message scan.
- `observe` emits one privacy-safe warning per model attempt when gaps exist.
- `assert` raises before provider I/O and is intended for tests and verified
  environments.

Source-backed user, model, and tool provenance requires a non-empty source
message ID. Source-less provenance is accepted only for explicitly synthetic
contributions. Mixed source-backed and synthetic contributions remain valid.

Reports contain only counts, bounded message positions, roles, and issue codes.
They never contain message content, source IDs, tool arguments, tool results, or
message objects. Phase one does not resolve source IDs against a store because
session entries, checkpoints, graph messages, and nested agents use distinct ID
namespaces.

## Consequences

Observation can inventory real provenance gaps before enforcement or deeper
projection consolidation. Assertion provides a host-testable contract without
changing production defaults. Enabled modes add one linear scan immediately
before model I/O; the benchmark records that cost.

Auxiliary model calls that bypass `attemptInvoke`, including title and activity
labels, remain outside this phase. A later decision can enable assertion by
default only after observation shows complete provenance across supported
paths.

## Alternatives Considered

- Enforce immediately: rejected because known synthetic and runtime-generated
  messages are not yet uniformly stamped.
- Inspect graph state earlier: rejected because runnable-level system messages
  and final provider transformations would be invisible.
- Validate source IDs against one store: rejected because no single store owns
  every valid message namespace.
