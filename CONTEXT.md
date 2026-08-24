# Domain Context

## Durable Subagent Execution

A **Durable Subagent Execution** is one child invocation identified by its
durable parent thread, parent checkpoint fork, parent agent, parent tool call,
tool batch, and resume attempt. Concurrent parent forks and resume attempts are
different executions even when they reuse the same parent tool-call ID.

An **Execution Record** is the in-process source of truth for one Durable
Subagent Execution. It owns the canonical execution address, resolved child
run and checkpoint-thread identity, approval scope, effective subagent type and
configuration revision, pending resolution and invocation work, active graph
state, completion state, and invalidation.

Child identity resolution is prepared before it is committed. Checkpoint
forks, approval replay restoration, and cleanup tracking become authoritative
only while the same Execution Record is still current; invalidated
preparations roll back only resources owned by their preparation lease. A
checkpoint writer keeps exclusive ownership until its preparation settles and
any owned cleanup finishes, so a retry cannot adopt a branch that the old
writer can still modify. Pre-existing branches are never deleted as
preparation-owned state. Active and cleaning resources stay fenced; retries
fail closed and can reacquire them only after they become free.

An **Invocation Binding** is the immutable subagent type, configuration
revision, and description accepted when an Execution Record starts. Exact
duplicate dispatches share its pending result. A same-address dispatch with a
different invocation fails closed instead of observing the first dispatch's
result.

After a resume attempt successfully forks its source checkpoints, the source
Execution Record is invalidated and removed. Its active graph, completed
messages, resolved configuration, and approval session are retired together.

An **Effective Definition Binding** is the subagent type and optional
configuration revision that actually ran after lifecycle hooks. Durable replay
must use this binding rather than reconstructing it from the original tool
call.

A **Resume Projection** is the durable manifest view produced from an
Execution Record. A projection is scoped to one exact fork and resume attempt;
an unscoped projection fails closed when a parent tool-call ID is ambiguous.

## Tool Caller Capabilities

A **Caller Capability Projection** is the effective classification of tool
definitions by the contexts permitted by `allowed_callers`: direct model calls,
programmatic code execution, both, or neither. Prompt guidance, model binding,
and runtime dispatch derive from this projection rather than recomputing caller
rules independently.

A **Programmatic Tool Manifest** is the exact list of registered tool names a
programmatic invocation declares that its code depends on. Mixed direct and
code-execution configurations require the manifest so caller policy can reject
direct-only dependencies before starting an execution runtime without parsing
the submitted programming language.

## Context Pressure Measurement

A **Context Pressure Meter** is the request-scoped projection of retained
conversation messages into the provider's available context budget. It keeps
provider-grounded attribution and exact token counts together so repeated
artifact, compaction, prepared-request, and fallback probes tokenize only new
message objects. Its Agent Context may retain exact counts across provider
requests only for stable string-content messages whose token-relevant surface
still matches.
Mutable or ambiguous message shapes are recounted. Provider projections must
clone changed messages; mutating a measured message invalidates its retained
count on the next comparison. Custom token counters remain uncached unless a
host explicitly declares that they implement the same deterministic surface
contract.
See [ADR 0001](docs/adr/0001-reuse-exact-context-token-counts.md) for the
cache boundary and host-lifecycle trade-offs.
Host-supplied index token counts are the baseline's accounting weight; a
retained message is tokenized only when no such count exists or when a
provider projection replaced its object, since the exact count then serves
only as the subtrahend of that projection's delta. Native brand checks in
content validation use `node:util/types` slot reads rather than throwing
probes, so plain content objects never pay an exception per value.

Provider SDK modules load on first use, never at import time: built-ins register
loaders with the provider registry, `instanceof` guards resolve their classes
through the same lazy seam, and every such load goes through `src/lazyRequire.ts`
— the one module allowed to touch `import.meta`, stubbed in jest so tests resolve
source modules. Adding an eager import of a provider SDK anywhere on the root
entry's graph regresses every host's boot.

The meter's estimates decide overflow and recovery only. Provider-reported
usage remains the source of truth for billing and Langfuse observations.

## Runtime Provider Registration

A **Provider Registration** is the process-local binding of one provider name
to its chat-model constructor and the message/streaming traits the agents
runtime must apply. Built-in and host providers resolve through the same
registry. Hosts register before constructing agent configuration; duplicate
names fail closed, and a disposer removes only the exact registration that
created it.

Provider family, manual tool-stream handling, and strict message alternation
belong to the registration. Family governs shared protocol behavior such as
thinking detection, output-token option names, and atomic tool-call handling.
Exact provider-specific behavior such as Bedrock cache shaping or OpenRouter
replay handling remains keyed to the built-in provider identity. Host option
types are declaration-merged through the stable
`@librechat/agents/provider-registration` subpath without widening the built-in
option maps.

See [ADR 0003](docs/adr/0003-runtime-provider-registration.md) for the shared
registry seam and process-lifecycle trade-offs.

## Coding-Tool Execution

An **Execution World** is the stable pairing of a filesystem namespace and a
subprocess launcher used by coding tools. The Node host has one default world;
remote backends retain one world per execution configuration so rebuilding an
agent's tool binding does not make a warm backend look new to capability
probes. Sandbox identity belongs to the same world, while timeout,
cancellation, output, and command results remain invocation-scoped.

See [ADR 0002](docs/adr/0002-stable-execution-worlds.md) for the adapter
identity and compatibility boundary.
