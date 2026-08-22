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
