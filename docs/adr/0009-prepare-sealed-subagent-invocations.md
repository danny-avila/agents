# ADR 0009: Prepare Sealed Foreground Subagent Invocations in ToolNode

## Status

Accepted

## Context

Event-driven tools already start from provider stream completion signals, but
all direct graph tools are excluded. A parent therefore finishes generating
every subagent prompt before any child starts. The graph-tool exclusion also
preserves a real ordering guarantee: direct tools can interrupt or redirect
the batch before event-driven tools reach the host.

Starting SubagentExecutor directly from the stream would duplicate ToolNode's
runtime setup and bypass its hooks, replay, output limits and reference
registration. Its Execution Record only deduplicates pending child work; it
is not a replacement for the complete tool lifecycle. Durable Subagent
Execution additionally requires the finalized parent checkpoint/batch identity.

## Decision

A **Prepared Subagent Invocation** is the raw invocation of a built-in
foreground subagent started by ToolNode during one explicitly open model
attempt. It is bound to its owning agent, tool-call ID and canonical arguments.
A graph-owned module reserves bounded work, adopts the raw result once, and
cancels abandoned work. The stream uses the existing provider sealing and
canonical-argument parsing implementation; it never infers completion from a
closing JSON brace alone.

The first adapter is the built-in subagent wrapper. It invokes the existing
SubagentExecutor through the same runtime construction as normal tools. Normal
ToolNode execution adopts its raw output, then applies existing output limits,
references and completion handling. Ordinary event tools retain the existing
direct-before-event ordering. There is no generic eager-direct-tool interface
or second subagent scheduler.

Early starts require event-driven execution with eager execution enabled, a
single built-in subagent graph tool, no checkpointer, no human-in-the-loop
mode, no interrupting tool names, and no parent tool lifecycle hooks. Background
calls, continued child threads, output-reference arguments and excluded tools
retain normal execution. Child lifecycle hooks still execute in the child.
An observation-only PostToolBatch registry remains compatible.

The provider-attempt scope opens admission and closes it before retry or
fallback. Completion reconciles reservations with the final tool calls. Late
buffered stream events cannot reopen a closed attempt. If an attempt fails,
discards or changes a call after delegation begins, the run fails closed rather
than automatically retrying work whose effects cannot be undone. Reset and
terminal cleanup cancel unadopted and still-running adopted invocations.

`eagerEventToolExecution.maxPendingSubagents` bounds outstanding early work
and retained raw results per graph; the default is four and zero disables this
path. Excess calls execute normally with the completed batch. Cancellation-
ignoring work keeps its admission slot until it settles. This is an early-work
limit, not a new process-wide limit for all foreground subagents.

## Consequences

- Long sibling prompts overlap with the first child's execution.
- ToolNode remains the module owning tool runtime and result processing;
  streaming does not acquire graph lifecycle responsibilities.
- The reservation interface concentrates identity, cancellation, admission,
  failure containment and result adoption in one testable module.
- Checkpointed/HITL runs keep their established replay semantics and do not
  receive the latency improvement in this first pass.
- Earlier child side effects are possible before the parent response commits.
  Cancellation is best effort, never rollback. Automatic provider fallback is
  deliberately unavailable after an early child invocation starts.
- Normal result order, references and completion events remain batch-owned;
  child activity can appear before the parent finishes generating sibling calls.
- Tests exercise actual streamed model and child graphs, plus cancellation,
  argument changes, duplicate admission, capacity and ToolNode adoption.
