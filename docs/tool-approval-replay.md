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
   into that destination. Unrelated child and parent scopes are unchanged.
7. Completed batches release their process-local acceleration cache. That cache
   is not the durable source of truth.

The private checkpoint record is versioned. Its codec preserves LangChain
messages; Command outputs are reconstructed before graph execution. Malformed
records fail closed. The replay module has no HTTP, database or host UI dependency.

## Recovery guarantees and host obligations

This protocol guarantees replay of **checkpointed completed work** when resuming
an approval pause. Rebuilding a Run/ToolNode with the same checkpointer is a
supported operation; retaining the old hooks or instances is not required.

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
  state and malformed records.

External-side-effect crash reconciliation and concurrent independent host
ownership require the host/tool contract above; these tests do not prove those
external guarantees.
