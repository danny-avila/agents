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
