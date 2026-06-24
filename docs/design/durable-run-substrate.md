# Design: Durable Run Substrate — checkpointer + RunControl drain

Status: **Draft for review** · Scope: design only (no code) · Repos: `@librechat/agents` (SDK) + LibreChat (app)

> Goal: a single durable substrate that unifies **abort**, **stream/run resume**, and **HITL** on top of LangGraph 1.4's checkpointer + `RunControl`, replacing three mechanisms that today solve overlapping problems in disconnected ways.

---

## 1. The core insight: two orthogonal durability layers

LibreChat conflates (or rather, only implements *one of*) two distinct kinds of durability:

| Layer | "What" it persists | Purpose | Today |
|---|---|---|---|
| **Presentation durability** | The *output* the user sees — SSE chunks, content parts, run steps, token/context usage | Client reconnect / multi-tab / network blip → replay what was rendered | **Solid:** `GenerationJobManager` (Redis Streams chunk-log + content reconstruction) |
| **Computation durability** | The *graph execution state* — channel values, pending writes, the interrupted node | Resume the actual computation: after HITL approval, after a graceful drain (deploy/SIGTERM), or on another worker | **Missing/faked:** in-memory `MemorySaver` fallback only |

These are **not** the same thing, and the prevailing instinct ("wrap a LangGraph checkpointer as an `IJobStore`") conflates them. `GenerationJobManager` answers *"what did the user already see?"*; a checkpointer answers *"what is the agent in the middle of doing, and how do I continue it?"* The substrate adds the **missing computation layer**; presentation stays where it is (and can later be *derived* from it, §7).

## 2. Current state — three disconnected mechanisms

1. **HITL checkpoint = in-memory only.** `Run.create → applyHITLCheckpointerFallback()` installs a `MemorySaver` when `humanInTheLoop.enabled` and no checkpointer is supplied (`agents/src/run.ts:256-323`). Process-local: an interrupt cannot survive a deploy/restart or move across workers. Resume discovery walks `getStateHistory` (`run.ts:1165-1241`).
2. **Stream resume = bespoke app layer.** `GenerationJobManager` (`packages/api/src/stream/GenerationJobManager.ts`) persists the *output*: Redis Streams `stream:{id}:chunks` (XADD), job metadata (HSET), run steps (JSON), with `getResumeState()` → a `sync` SSE event that replays `runSteps` + `aggregatedContent`. Pluggable `IJobStore`/`IEventTransport` (in-memory ↔ Redis). **`streamId === conversationId` always.**
3. **Abort = hard cancel.** `abortJob()` calls `abortController.abort()` (or Redis pub/sub `emitAbort` cross-replica), harvests partial content, persists an unfinished message (`abortMiddleware.js:102-209`). The SDK *stores* an `AbortSignal` on the graph but **never uses it** for graceful stop (`agents/src/graphs/Graph.ts:575`). Aborted computation is lost, not resumable.

Net: presentation is durable; **computation is not**; and abort throws away in-flight work.

## 3. The substrate

### 3.1 Durable checkpointer (`BaseCheckpointSaver`)

Implement a durable saver against the 1.4.5 contract (`@langchain/langgraph-checkpoint`):

- `getTuple(config)`, `put(config, checkpoint, metadata, newVersions)`, `putWrites(config, writes, taskId)`, `list(config, opts)`, `deleteThread(threadId)` (new in 1.4.0), and `getDeltaChannelHistory({config, channels})` (new in 1.4.0 — a default walk-the-parent-chain impl exists; override for store-native efficiency).
- `SerializerProtocol` (`dumpsTyped → [type, bytes]` / `loadsTyped`) for checkpoint/write/blob bytes.

**Where it lives → decision (D1).** The SDK has no DB connection; LibreChat already has Mongo + an `ioredis` client + the pluggable persistence pattern. Recommendation: **implement the saver in LibreChat (`packages/api`), reusing its Redis/Mongo infra, and pass it to the SDK via `compileOptions.checkpointer`.** The SDK stays storage-agnostic (it already accepts `compileOptions.checkpointer`, `graphs/Graph.ts:837,2575`).

**Storage shape (Redis-first, mirrors GenerationJobManager's existing keying):**
```
checkpoint:{thread}:{ns}:{id}        → serialized checkpoint tuple (config, checkpoint, metadata)
checkpoint:{thread}:{ns}:writes:{task} → pending writes (per task)
checkpoint:{thread}:{ns}:blobs:{ch}  → DeltaChannel snapshots (large channel values)
checkpoint:{thread}:index            → ordered checkpoint ids (for list/getStateHistory)
```
Mongo equivalent: `checkpoints` / `checkpoint_writes` / `checkpoint_blobs` collections keyed by `(thread_id, checkpoint_ns, checkpoint_id)`. **Decision (D2): Redis, Mongo, or both?** Redis matches the existing stream infra + TTL ergonomics and is the natural fit for *transient resumable runs*; Mongo fits *long-lived, replayable* threads. Likely **Redis for the live run + optional Mongo for durable archive**, but start Redis-only.

### 3.2 DeltaChannel for `messages` — non-negotiable for efficiency

A full checkpoint serializes `channel_values.messages`, which for LibreChat includes **base64 images/PDFs and full history → 100s KB–MB per checkpoint**, written *every superstep*. That's untenable at HITL/abort frequency. 1.4.0's `DeltaChannel` + `messagesDeltaReducer` stores only the delta writes between periodic snapshots and reconstructs via `getDeltaChannelHistory`. **Adopt `messagesDeltaReducer` for the message channel** so the durable saver writes ~1 KB/step instead of MBs. (This is also exactly the "checkpoint efficiency" item already flagged.) **Decision (D3): snapshotFrequency** (e.g. every N messages / supersteps) tuned to the media-heavy reality.

### 3.3 RunControl drain — graceful, resumable abort

Replace hard-cancel with cooperative draining (1.4.0):
- Add `control?: RunControl` to the SDK `Run` and thread it into `graph.streamEvents(input, {...config, control}, …)` (`run.ts:797`).
- On external signal (SIGTERM for deploys; user-abort; "all subscribers left" — `agents/request.js:271-329`), call `control.requestDrain(reason)`. The graph stops at the next superstep boundary, **persists its checkpoint**, and throws `GraphDrained` (`isGraphDrained(e)`); the SDK surfaces it as a resumable halt rather than a lost run.
- Resume = `Run` re-created with the same `thread_id` + durable checkpointer; existing `resume()` / `resolveInterruptResumeConfig` machinery applies.

This makes **abort and HITL the same mechanism**: both are "pause at a boundary, checkpoint, resume later," differing only in *who* triggers the pause (system vs human) and *what* the resume carries.

### 3.4 Identity — already aligned

`conversationId === streamId` today; it becomes the LangGraph **`thread_id`** verbatim. `checkpoint_id`/`checkpoint_ns` (config.configurable) identify resume points; the SDK already plumbs these (`run.ts:685,762-763`). No new identity model needed — just route `conversationId → configurable.thread_id`.

### 3.5 SDK changes required (small, additive)

1. Accept/prefer a host-supplied durable checkpointer over the `MemorySaver` fallback (already supported via `compileOptions.checkpointer`; just stop forcing MemorySaver when one is present — already idempotent).
2. Expose `control: RunControl` on `Run` + catch/surface `GraphDrained`.
3. (Optional, with DeltaChannel) allow the message channel to be compiled as a `DeltaChannel`.

Everything else (interrupt/resume, thread config, getStateHistory discovery) **already exists** — the SDK is largely ready; the gap is a *durable* saver + *drain* wiring.

## 4. How it unifies abort / resume / HITL

| Use case | Trigger | Mechanism on the substrate |
|---|---|---|
| **HITL** | `interrupt()` in a tool/node (`ToolNode.ts:1363`) | checkpoint at interrupt → durable store → resume with `Command({resume, update, goto})` (1.4.5) |
| **Graceful abort / deploy** | `control.requestDrain()` (SIGTERM, user abort, subscribers-left) | checkpoint at boundary → `GraphDrained` → resume later, same thread |
| **Stream resume** | client reconnect | **presentation** replay (GenerationJobManager) for instant catch-up; **computation** continues from checkpoint if the run had drained/interrupted |

One substrate, one identity (`thread_id`), one durable store — three behaviors.

## 5. HITL e2e (first consumer)

What it needs from the substrate + the deltas on each side:
- **SDK:** durable checkpointer (this design) + adopt 1.4.5 `Command({resume, update, goto})` so an approval can *also* commit a state edit + reroute in one superstep (today `resume()` only does `Command({resume})`, `run.ts:1142-1163`).
- **LibreChat backend:** new `on_interrupt` SSE event carrying the interrupt payload; a resume endpoint (mirror the abort endpoint) that calls `run.resume(decision, …)`; persist `thread_id`/`checkpoint_id` on the conversation.
- **Frontend:** `useSSE` handles `on_interrupt` → renders an approval/ask UI → POSTs the decision → server resumes.

HITL is just "interrupt + durable checkpoint + resume" — which is why the substrate is the prerequisite, not a parallel effort.

## 6. Concurrency (1.4.4)

If LibreChat ever shares a *compiled* graph across concurrent requests or runs agents in BullMQ workers with `concurrency > 1`, 1.4.4's "isolate concurrent singleton-agent invocations by thread" fix is load-bearing (ambient `AsyncLocalStorage` config no longer leaks across top-level `invoke()`s). **Action:** confirm LibreChat's run-per-request model (compile-per-run vs shared compiled graph) — the durable substrate makes cross-worker execution attractive, so this must be settled before fan-out.

## 7. Relationship to GenerationJobManager

They are **complementary**, and should stay so initially:
- **Now:** checkpointer = computation durability (new); GenerationJobManager = presentation durability (unchanged). The checkpointer makes HITL/abort durable; GJM keeps fast client replay.
- **Later (optional consolidation):** once the checkpoint is the source of truth for computation, GJM's bespoke chunk-log could be *derived* from checkpoint state (reconstruct content from messages channel) rather than maintained as a separate XADD log — shrinking the bespoke layer. **Do not** couple them in v1; prove the checkpointer first.

## 8. Phasing

1. **Durable checkpointer (Redis)** in `packages/api`, `conversationId → thread_id`, wired via `compileOptions.checkpointer`; swap the HITL `MemorySaver` fallback for it. *(Unblocks durable HITL.)*
2. **DeltaChannel for messages** + `getDeltaChannelHistory` in the saver. *(Makes #1 affordable.)*
3. **RunControl drain** in the SDK `Run` + LibreChat triggers (SIGTERM, abort, subscribers-left). *(Graceful, resumable abort.)*
4. **HITL e2e** (`on_interrupt` SSE + resume endpoint + `Command(resume,update,goto)` + frontend UI).
5. **(Optional)** derive presentation replay from checkpoint; trim GJM.

## 9. Open decisions
- **D1:** checkpointer in LibreChat `packages/api` (reuse infra) vs a generic saver in the SDK. *(Rec: LibreChat.)*
- **D2:** Redis vs Mongo vs both for checkpoint storage. *(Rec: Redis-first.)*
- **D3:** DeltaChannel `snapshotFrequency` + whether to also delta the non-message channels.
- **D4:** retention/TTL for checkpoints (align with GJM's 1200s/300s, or longer for resumable HITL threads that may wait hours/days for a human).
- **D5:** subsume vs complement GenerationJobManager (rec: complement in v1).
- **D6:** serialization — reuse GJM's JSON approach vs a compact codec for blobs; how base64 media is stored (inline blob vs reference to existing file storage).

## 10. Risks
- **Checkpoint size / write amplification** without DeltaChannel → #2 must land with #1.
- **Serializer fidelity** — non-plain objects (Command/Send, custom content parts) must round-trip; 1.3.3 fixed some of this upstream, verify against our content types.
- **DeltaChannel is beta** — `getDeltaChannelHistory` is new; validate reconstruction equals full-snapshot state for our message shapes (images, reasoning blocks, tool calls).
- **Long-lived HITL threads** (human waits days) imply checkpoint retention >> stream TTL; GC policy must not reap a paused run.
- **Concurrency model** (§6) must be confirmed before any cross-worker resume.

---

### Appendix — key references
- SDK: `run.ts:256-323` (MemorySaver fallback), `run.ts:797` (streamEvents), `run.ts:1142-1241` (resume + checkpoint discovery), `graphs/Graph.ts:575` (unused AbortSignal), `graphs/Graph.ts:837,2575` (compileOptions.checkpointer), `tools/ToolNode.ts:1363` (interrupt).
- LangGraph 1.4.5: `@langchain/langgraph-checkpoint` `BaseCheckpointSaver` (getTuple/put/putWrites/list/deleteThread/getDeltaChannelHistory), `pregel/runtime` `RunControl`, `pregel/types` `PregelOptions.control`, `errors` `GraphDrained`, `channels/delta` `DeltaChannel`, `graph/messages_reducer` `messagesDeltaReducer`.
- LibreChat: `packages/api/src/stream/GenerationJobManager.ts` (job lifecycle, emitChunk, getResumeState, subscribeWithResume, abortJob), `implementations/{InMemory,Redis}{JobStore,EventTransport}.ts`, `abortMiddleware.js:102-209`, `routes/agents/index.js:63-162` (stream/abort endpoints), `agents/request.js` (identity, subscribers-left).
</content>
