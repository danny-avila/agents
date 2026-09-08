# Tool approval replay contract

An approval belongs to an execution, not to a JavaScript object or the current
hook registry. Reconstructing a `Run` must not turn a rejected action into an
allowed action or repeat a sibling whose completed result was checkpointed.

## Ownership and lifecycle

1. The ToolNode resolves execution arguments and evaluates current policy.
2. An `ask` records the exact tool-call identity, arguments and decision
   allowlist. The originating execution owns that review. Parent nodes carry
   child evidence through the trusted subagent adapter but do not consume it.
3. Before publishing a pause, the direct batch waits for its started siblings
   to settle. Completed results, hook context, reference content and completion
   delivery state accompany the interrupt in the checkpoint. Only the active
   batch is included; another thread's in-flight results are not copied.
4. `Run.resume` obtains private evidence from the checkpoint, not host-supplied
   configurable fields. Public interrupts do not expose the private records.
5. A fresh ToolNode restores completed siblings and consumes the reviewed
   decision for pending calls. Current denial still wins. Changed proposals or
   disallowed decisions fail closed. Removing a hook cannot remove a pending
   review; a valid approval uses the arguments actually reviewed.
6. A child checkpoint fork receives a new execution scope. Only the trusted
   subagent adapter rebinds evidence from the manifest's proven source scope
   into that destination, including durable pending writes that have not yet
   been consumed. Each child restores its own pending evidence. Unrelated child
   and parent scopes are unchanged. A foreign owner is accepted only as transit
   through a trusted child tool, a matching parent batch record, and a validated
   descendant manifest; changing tool-call IDs cannot bypass ownership checks.
7. Completed batches release their process-local acceleration cache. That cache
   is not the durable source of truth: selecting a checkpoint replaces any
   newer local result for that batch, including when the checkpoint is empty.
8. The checkpoint also retains per-tool usage counters and active-call turns.
   Restored outputs and pending calls keep their original execution and
   completion indices, with or without a hook registry. Historical per-call
   turn entries are not copied into the active batch's checkpoint.
9. Batch identity binds the assistant message and tool proposals, not transcript
   length: additional resume messages cannot repeat completed work. The
   pre-batch output-reference snapshot is a frozen input view, distinct from
   the graph's shared live registry. Replay retains its original batch turn
   without lowering the shared counter or discarding concurrent outputs.
   Cached messages are rebound to the active reference scope; completed siblings
   cannot change what pending siblings originally resolved. Current reference
   size limits also apply to the frozen view.
10. Direct calls without IDs use their original proposal position in the private
    completion record. Filtering completed named calls cannot shift that identity;
    successful ID-less outputs survive pauses without repeating their side effects.

The private checkpoint record is versioned. Its codec preserves LangChain
messages; Command outputs are reconstructed before graph execution. Malformed
records fail closed. The replay module has no HTTP, database or host UI dependency.

## Recovery guarantees and host obligations

Hosts should set `TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY` in `configurable`
to a generation identifier. Keep it stable through every approval resume,
including rebuilt `Run` instances, and change it for each new generation. A
conversation id alone is insufficient; edits that reuse a response id also need
a generation epoch. Explicit scoping removes LangGraph's changing task namespace
from the composite owner while retaining the executing agent id.

Carried evidence is active only for its resumed interrupt and consuming task.
The interrupt id is checked against LangGraph's resume map, and the task's
scratchpad distinguishes a resumed node from later steps of the same invocation.
A validated parent replaying a child approval can forward that child's evidence
without a local resume value. Stale review and settled-batch evidence are ignored
together. An active approval with a different execution owner still fails closed;
starting a fresh generation may select a different agent without inheriting the
prior approval.

This protocol guarantees replay of **checkpointed completed work** when resuming
an approval pause. Rebuilding a Run/ToolNode with the same checkpointer is a
supported operation; retaining the old hooks or instances is not required.

Standalone ToolNode callers must supply a stable `configurable.thread_id` (or
the SDK-managed approval execution scope) to identify a replay. Anonymous
invocations are independent operations and never share completed-result state.

This is not a distributed exactly-once transaction for arbitrary external tools.
A process can die after an external side effect but before a checkpoint commits.
That outcome is ambiguous: absence of a saved result does not prove the tool did
not run. Hosts must serialize ownership of a running thread, and tools requiring
retry safety must use a durable operation/idempotency key or reconcile with their
external system. A deliberate checkpoint fork is a new branch, not an ordinary
duplicate submission. The SDK must not advertise arbitrary shell commands or
third-party tools as exactly-once across that crash window.

## Compatibility and rollout

- New consumers accept legacy approval payloads without the private record.
  Reviewed rejection and allowlist restrictions still apply. Unreviewed direct
  siblings fail closed when their prior policy cannot be reconstructed.
- Existing public approval payload and resume-decision shapes are unchanged.
- Old SDK consumers cannot restore the new completion record. Do not route a
  paused run between SDK versions indiscriminately: drain or version-pin paused
  runs during rollout and before rollback. A durable checkpointer alone is not
  an execution-owner lease.

## Focused acceptance coverage

- Fresh Run and ToolNode: direct and event approval with rewritten arguments,
  with the former hook registry absent.
- Fresh runtime with a different Run ID: mixed direct/event pause, preserving
  one side effect, its returned result, and one completion event.
- Repeated terminal resume: no additional execution.
- Nested checkpoint forks: rejection remains effective without the former
  hooks, and parent siblings remain independent of child approval evidence.
- Codec and validation: messages, Commands, nested ownership, absent legacy
  state and malformed records. Restored results require a proposal binding;
  custom payloads that resemble private wrappers preserve their public shape.
- Older-checkpoint selection overrides a newer local cache; empty execution
  scopes remain isolated anonymous invocations.

External-side-effect crash reconciliation and concurrent independent host
ownership require the host/tool contract above; these tests do not prove those
external guarantees.
