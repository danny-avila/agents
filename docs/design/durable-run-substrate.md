# Design: Durable Run Substrate — checkpointer + RunControl drain

Status: **Draft for review** · Scope: design only (no code) · Repos: `@librechat/agents` (SDK) + LibreChat (app)

> Goal: a single durable substrate that unifies **abort**, **stream/run resume**, and **HITL** on top of LangGraph 1.4's checkpointer + `RunControl`, replacing three mechanisms that today solve overlapping problems in disconnected ways.

**Related SDK PRs (status):** `#272` (resume forwards `update` + `goto`, langgraph 1.4.5) is **OPEN**. `#273` (`ToolRuntime.state` migration, replaces `getCurrentTaskInput`) is **OPEN**. Neither is merged yet.

---

## North-star: relocatable execution

HITL is the first consumer, not the point. The substrate exists so an agent run's **state lives outside the compute that executes it**, which is the prerequisite for running agents on ephemeral or isolated compute (AWS Lambda, Temporal activities, autoscaled workers) instead of one long-lived process. Three mechanisms make a run relocatable:

- **State outside compute** via the Mongo checkpointer: every superstep boundary is durably persisted, fully keyed by `thread_id` (= `conversationId`), with **zero run state retained in-process**. A different worker or invocation holding only the `thread_id` plus the shared checkpointer can pick the run up.
- **RunControl drain** (1.4.0): on SIGTERM, scale-in, or handoff, `requestDrain()` stops at the next superstep boundary, checkpoints, and throws `GraphDrained`; another worker resumes from that checkpoint. Drain plus resume *is* relocation.
- **1.4.4 thread-isolation** lets concurrent runs share one compiled graph without ambient-config leakage, which is required before fanning runs across a worker pool.

**Three deployment shapes, and they are NOT all RemoteGraph:**

1. **langgraph-server tier (RemoteGraph):** run the graph behind langgraph's server and call it via `RemoteGraph`. Heaviest; only if we want the full server runtime.
2. **Temporal wraps langgraph:** Temporal owns orchestration and durability; langgraph runs inside activities. Pick **one** durability model (Temporal history *or* the checkpointer); running both is redundant and conflicting.
3. **Lambda plus shared checkpointer (no RemoteGraph):** just the SDK plus the Mongo checkpointer. Each invocation rebuilds the `Run` from `thread_id`, runs to the next boundary or completion, and persists. Lightest path and the natural fit for LibreChat's existing infra.

**Design implication:** build the checkpointer foundation *relocation-ready* from day one (zero in-process run state, fully `thread_id`-keyed, drain-compatible). One concrete requirement this surfaces: on a **rebuilt** `Run` (shapes 2 and 3), the SDK message baseline (`Graph.startIndex`, set in the message reducer) is seeded from the first write *after* checkpoint restore, so a `Command.update.messages` injected on resume (e.g. a human edit) is excluded from `getRunMessages()` / `returnContent` even though it is committed to the checkpoint (Codex P2 on `#272`). The relocatable-execution work must seed the wrapper baseline from the *restored* checkpoint length so injected updates and continued output are not dropped. Inert today; only reachable once a durable checkpointer plus a rebuilt Run exist.

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

**Adopt the official saver, do not build one.** `@langchain/langgraph-checkpoint-mongodb@1.4.0` already implements the full 1.4 contract (`getTuple` / `put` / `putWrites` / `list` / `deleteThread`; `getDeltaChannelHistory` is inherited as a concrete default from `BaseCheckpointSaver`) and bundles `@langchain/langgraph-checkpoint@1.1.2`, the exact base langgraph 1.4.5 uses. Its constructor takes an **injected** `MongoClient` (`new MongoDBSaver({ client, dbName })`), so it rides LibreChat's existing Mongoose connection (`mongoose.connection.getClient()`) with no new pool or lifecycle. The official Redis saver is stale (`1.0.10`, pre-1.4, lacks the `deleteThread`/delta contract), so it is not a candidate.

**Where it lives → decision (D1, resolved).** LibreChat `packages/api`: a thin, config-driven `createCheckpointer(config)` factory that constructs `MongoDBSaver` from the existing client and is passed to the SDK via `graphConfig.compileOptions.checkpointer` (`graphs/Graph.ts:837,2575`). The SDK stays storage-agnostic and needs **zero changes**: `applyHITLCheckpointerFallback` already prefers a caller-supplied checkpointer over `MemorySaver`.

**Backend → decision (D2, resolved): Mongo.** It is LibreChat's durable store (the right home for HITL threads that may wait hours or days, where Redis TTLs fight retention), the official saver is current, and it reuses the existing connection. Collections: `checkpoints` / `checkpoint_writes`, keyed by `(thread_id, checkpoint_ns, checkpoint_id)`. No lock-in: the wiring depends on the `BaseCheckpointSaver` interface, so the factory can swap to Redis/Postgres later with no change to the run/HITL path, and checkpoint data is in-flight run state, not canonical history.

### 3.2 DeltaChannel for `messages` — near-term, driven by Mongo's 16MB doc limit

A full checkpoint serializes `channel_values.messages`, which for LibreChat includes **base64 images/PDFs and full history → 100s KB to MB per checkpoint**, written *every superstep*. Two pressures make this untenable: CPU/write-amplification at HITL/abort frequency, and, because the backend is **Mongo**, the **BSON 16 MB hard document limit**. A long HITL thread with inlined media can bloat a full-blob checkpoint doc and, in the limit, hit the cap, which hard-fails the `put`. So DeltaChannel is **not** a generic "later optimization"; it is a near-term follow-up specifically gated by the Mongo doc-size ceiling.

- **Phase 1 (works today):** full-blob checkpoints via the official Mongo saver. Fine for bounded threads and the simplest path to durable HITL.
- **Phase 2 (DeltaChannel):** `DeltaChannel` + `messagesDeltaReducer` (1.4.0) store only delta writes between periodic snapshots (sentinel + writes-replay, snapshot cadence) and reconstruct via `getDeltaChannelHistory`, dropping per-step writes from MBs to ~1 KB and keeping any single doc well under 16 MB. **Decision (D3): snapshot cadence** tuned to the media-heavy reality. Note: the official Mongo saver does not store-natively optimize deltas yet (it inherits the base `getDeltaChannelHistory`), so Phase 2 may need a thin saver extension or an upstream contribution.

### 3.3 RunControl drain — graceful, resumable abort

Replace hard-cancel with cooperative draining (1.4.0):
- Add `control?: RunControl` to the SDK `Run` and thread it into `graph.streamEvents(input, {...config, control}, …)` (`run.ts:797`).
- On external signal (SIGTERM for deploys; user-abort; "all subscribers left" — `agents/request.js:271-329`), call `control.requestDrain(reason)`. The graph stops at the next superstep boundary, **persists its checkpoint**, and throws `GraphDrained` (`isGraphDrained(e)`); the SDK surfaces it as a resumable halt rather than a lost run.
- Resume = `Run` re-created with the same `thread_id` + durable checkpointer; existing `resume()` / `resolveInterruptResumeConfig` machinery applies.

This makes **abort and HITL the same mechanism**: both are "pause at a boundary, checkpoint, resume later," differing only in *who* triggers the pause (system vs human) and *what* the resume carries.

### 3.4 Identity — already aligned

`conversationId === streamId` today; it becomes the LangGraph **`thread_id`** verbatim. `checkpoint_id`/`checkpoint_ns` (config.configurable) identify resume points; the SDK already plumbs these (`run.ts:685,762-763`). No new identity model needed — just route `conversationId → configurable.thread_id`.

### 3.5 SDK changes required — none for the checkpointer

1. **Checkpointer: zero changes.** A host-supplied durable checkpointer is already accepted and preferred over the `MemorySaver` fallback (`compileOptions.checkpointer`; `applyHITLCheckpointerFallback` is already idempotent), and `conversationId → configurable.thread_id` is already plumbed both sides.
2. Expose `control: RunControl` on `Run` + catch/surface `GraphDrained` (for §3.3 drain).
3. (Optional, with DeltaChannel) allow the message channel to be compiled as a `DeltaChannel` (for §3.2 Phase 2).

Interrupt/resume, thread config, and getStateHistory discovery **already exist** — the SDK is largely ready; the gaps are *drain* wiring and the optional *DeltaChannel* compile, not the checkpointer itself.

## 4. How it unifies abort / resume / HITL

| Use case | Trigger | Mechanism on the substrate |
|---|---|---|
| **HITL** | `interrupt()` in a tool/node (`ToolNode.ts:1363`) | checkpoint at interrupt → durable store → resume with `Command({resume, update, goto})` (1.4.5) |
| **Graceful abort / deploy** | `control.requestDrain()` (SIGTERM, user abort, subscribers-left) | checkpoint at boundary → `GraphDrained` → resume later, same thread |
| **Stream resume** | client reconnect | **presentation** replay (GenerationJobManager) for instant catch-up; **computation** continues from checkpoint if the run had drained/interrupted |

One substrate, one identity (`thread_id`), one durable store — three behaviors.

## 5. HITL e2e (first consumer)

What it needs from the substrate + the deltas on each side:
- **SDK:** durable checkpointer (this design) + 1.4.5 `Command({resume, update, goto})` so an approval can *also* commit a state edit + reroute in one superstep (`#272`, OPEN). Plus the rebuilt-run baseline-seeding fix called out in the north-star, so injected `update.messages` survive in `getRunMessages()` / `returnContent`.
- **LibreChat backend:** new `on_interrupt` SSE event carrying the interrupt payload; a resume endpoint (mirror the abort endpoint) that calls `run.resume(decision, …)`; persist `thread_id`/`checkpoint_id` on the conversation.
- **Frontend:** `useSSE` handles `on_interrupt` → renders an approval/ask UI → POSTs the decision → server resumes.

HITL is just "interrupt + durable checkpoint + resume" — which is why the substrate is the prerequisite, not a parallel effort.

## 6. Concurrency (1.4.4)

If LibreChat ever shares a *compiled* graph across concurrent requests or runs agents in BullMQ workers with `concurrency > 1`, 1.4.4's "isolate concurrent singleton-agent invocations by thread" fix is load-bearing (ambient `AsyncLocalStorage` config no longer leaks across top-level `invoke()`s). **Action:** confirm LibreChat's run-per-request model (compile-per-run vs shared compiled graph) — the durable substrate makes cross-worker execution attractive (see north-star), so this must be settled before fan-out.

## 7. Relationship to GenerationJobManager

They are **complementary**, and should stay so initially:
- **Now:** checkpointer = computation durability (new); GenerationJobManager = presentation durability (unchanged). The checkpointer makes HITL/abort durable; GJM keeps fast client replay.
- **Later (optional consolidation):** once the checkpoint is the source of truth for computation, GJM's bespoke chunk-log could be *derived* from checkpoint state (reconstruct content from messages channel) rather than maintained as a separate XADD log — shrinking the bespoke layer. **Do not** couple them in v1; prove the checkpointer first.

## 8. Phasing

1. **Durable checkpointer (Mongo, official saver)** in `packages/api`, `conversationId → thread_id`, wired via `graphConfig.compileOptions.checkpointer`, gated on HITL-capable runs; full-blob Phase 1. *(Unblocks durable HITL. Zero SDK changes.)*
2. **DeltaChannel for messages** + store-native `getDeltaChannelHistory`. *(Makes #1 affordable AND keeps docs under Mongo's 16MB limit.)*
3. **RunControl drain** in the SDK `Run` + LibreChat triggers (SIGTERM, abort, subscribers-left). *(Graceful, resumable abort = relocation.)*
4. **HITL e2e** (`on_interrupt` SSE + resume endpoint + `Command(resume,update,goto)` + frontend UI), including the rebuilt-run baseline-seeding fix (north-star requirement).
5. **(Optional)** derive presentation replay from checkpoint; trim GJM.

## 9. Open decisions
- **D1 (resolved):** checkpointer factory in LibreChat `packages/api` (adopt the official saver, inject the existing Mongoose client); the SDK stays storage-agnostic and unchanged.
- **D2 (resolved): Mongo** (official `@langchain/langgraph-checkpoint-mongodb@1.4.0`). Redis saver is stale; Mongo is the durable store and reuses the connection. Swappable later via the `BaseCheckpointSaver` interface (no lock-in).
- **D3:** DeltaChannel snapshot cadence + whether to also delta the non-message channels (Phase 2, gated by the 16MB limit, not just CPU).
- **D4:** retention/TTL for checkpoints (HITL threads may wait hours or days, far longer than GJM's 1200s/300s); GC must not reap a paused run.
- **D5:** subsume vs complement GenerationJobManager (rec: complement in v1).
- **D6:** serialization / blob strategy — how base64 media is stored (inline blob vs reference to existing file storage). Interacts directly with the 16MB limit, which favors references for media.

## 10. Risks
- **Mongo 16MB BSON doc limit** — full-blob checkpoints of long, media-heavy HITL threads can bloat toward (and in the limit hit) the cap, hard-failing `put`. Phase 1 is bounded by this, so **DeltaChannel (#2) is the mitigation and is therefore near-term, not optional.**
- **Serializer fidelity** — non-plain objects (Command/Send, custom content parts) must round-trip; 1.3.3 fixed some of this upstream, verify against our content types.
- **DeltaChannel is beta** — `getDeltaChannelHistory` is new; validate reconstruction equals full-snapshot state for our message shapes (images, reasoning blocks, tool calls).
- **Long-lived HITL threads** (human waits days) imply checkpoint retention >> stream TTL; GC policy must not reap a paused run.
- **Concurrency model** (§6) must be confirmed before any cross-worker resume.

---

### Appendix — key references
- SDK: `run.ts:256-323` (MemorySaver fallback), `run.ts:797` (streamEvents), `run.ts:1142-1241` (resume + checkpoint discovery), `graphs/Graph.ts:575` (unused AbortSignal), `graphs/Graph.ts:837,2575` (compileOptions.checkpointer), `graphs/Graph.ts:2554` (message reducer `startIndex` baseline), `tools/ToolNode.ts:1363` (interrupt).
- LangGraph 1.4.5: `@langchain/langgraph-checkpoint` `BaseCheckpointSaver` (getTuple/put/putWrites/list/deleteThread/getDeltaChannelHistory), `@langchain/langgraph-checkpoint-mongodb@1.4.0` `MongoDBSaver`, `pregel/runtime` `RunControl`, `pregel/types` `PregelOptions.control`, `errors` `GraphDrained`, `channels/delta` `DeltaChannel`, `graph/messages_reducer` `messagesDeltaReducer`.
- LibreChat: `packages/api/src/stream/GenerationJobManager.ts` (job lifecycle, emitChunk, getResumeState, subscribeWithResume, abortJob), `implementations/{InMemory,Redis}{JobStore,EventTransport}.ts`, `abortMiddleware.js:102-209`, `routes/agents/index.js:63-162` (stream/abort endpoints), `agents/request.js` (identity, subscribers-left), `packages/api/src/agents/run.ts:1154-1173` (Run.create injection point), `packages/api/src/agents/config.ts:11-30` (resolve-config cascade pattern), `api/db/connect.js` (Mongoose connection → `getClient()`).
