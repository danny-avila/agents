# Summarization & Context Management Behavior

## Overview

LibreChat's agent context management uses a staged pipeline inspired by Claude Code's compaction approach. The behavior differs based on whether summarization is enabled or disabled for the agent.

Both paths share **observation masking** as the first line of defense. The key difference is what happens when masking alone isn't enough: summarization-enabled agents compact the full conversation via an LLM call, while summarization-disabled agents apply progressively aggressive mechanical truncation.

---

## Shared Behavior (Both Paths)

### Observation Masking (80%+ context pressure)

When the total message tokens exceed 80% of the pruning budget, consumed ToolMessages are replaced with tight head+tail truncations (~300 chars) that serve as informative placeholders.

**Consumed** means: a subsequent AI message exists with substantive text content (not purely tool calls). The model has already read and acted on the result.

- AI messages are **never masked** — they contain the model's own reasoning and conclusions, which prevents the model from repeating work after tool results are masked.
- **Unconsumed** tool results (the latest outputs the model hasn't responded to yet) are left intact.
- This runs every agent node turn when pressure is at or above 80%.

### Token Budget Anatomy

```
maxContextTokens (e.g. 8000)
  - reserveTokens (5% default)
  = pruningBudget
  - instructionTokens (system message + tool schemas)
  = effectiveMaxTokens (available for conversation messages)
```

`contextPressure = totalMessageTokens / pruningBudget`

---

## When Summarization is Enabled

### Pipeline (every agent node turn)

1. **< 80% pressure**: No modifications. Messages pass through untouched.

2. **80%+ pressure — Observation masking**: Consumed ToolMessages masked to ~300 char placeholders. Pre-masking snapshot saved so the summarizer can access un-masked originals later.

3. **Fit-to-budget truncation**: Any individual message still exceeding `effectiveMaxTokens` is truncated via `preFlightTruncateToolResults` / `preFlightTruncateToolCallInputs`. Uses 30% of effective budget as per-result cap with recency weighting.

4. **Pruning split**: `getMessagesWithinTokenLimit` determines which messages fit (`context`) and which overflow (`messagesToRefine`). Messages are kept newest-first.

5. **Summarization trigger**: If `messagesToRefine` is non-empty:
   - `shouldSkipSummarization` only blocks when the message count hasn't changed since the last summary (prevents re-summarizing identical content).
   - No per-run cap on summarization count.
   - If triggered: **full compaction** fires.

### Full Compaction

When summarization fires:

- The **entire conversation** (un-masked originals from the snapshot) is sent to the summarizer — not just the dropped messages.
- The summarizer produces a structured checkpoint covering the full conversation history.
- Graph state is wiped completely (`createRemoveAllMessage()`) — no surviving messages.
- The summary is stored on `AgentContext` but **not** injected into the system prompt (doesn't inflate `instructionTokens`).

### Post-Compaction Clean Slate

After compaction, the message array is empty. On the next agent node turn:

- The system runnable detects `messages.length === 0` with a mid-run summary present.
- It injects `[SystemMessage(instructions), HumanMessage(summary)]`.
- The model reads the checkpoint as a user message and continues naturally — making tool calls or responding.
- The summary competes for message budget rather than permanently reducing the instruction ceiling.

### Summarization Tiers

1. **Tier 1 (cache-hit)**: Raw conversation messages sent with tools bound + summarization instruction as final HumanMessage. Benefits from cache hits on the system prompt + tool definitions prefix. Falls back to Tier 2 on failure.

2. **Tier 2 (single-pass)**: Messages formatted as text in a `<conversation>` block, sent with a SystemMessage prompt + HumanMessage body. Half the character budget of Tier 1.

3. **Tier 3 (metadata stub)**: No LLM call. Generates a mechanical summary listing tool names and message counts. Last resort.

### Summarization Prompt

The prompt is written in the tone of a user directing the assistant — assertive, first-person, active voice:

> "Hold on, before you continue I need you to write me a checkpoint of everything so far..."

This prevents the model from continuing to roleplay or respond to the conversation instead of producing a structured checkpoint.

### Fallback Fading

If observation masking + fit-to-budget still produce an **empty context** (no messages fit at all), context pressure fading is applied as a fallback before emergency truncation. This uses the same pressure-band graduated truncation from the disabled path.

### Cross-Run Behavior

- `initialSummary` from the prior run is included in the **system prompt** via `buildInstructionsString`.
- `formatAgentMessages` drops messages before the summary boundary in the message chain.
- The model sees the system prompt (with summary) + the user's new message.
- Mid-run summaries do NOT go into the system prompt — they use the HumanMessage injection on clean slate.

---

## When Summarization is Disabled

### Pipeline (every agent node turn)

1. **< 80% pressure**: No modifications.

2. **80%+ pressure — Observation masking**: Same consumed-only masking as the summarization-enabled path. Consumed ToolMessages masked, unconsumed left intact, AI messages untouched.

3. **80%+ pressure — Context pressure fading**: Additional progressive truncation of remaining oversized tool results based on graduated pressure bands:

   | Pressure | Budget factor | Effect                                        |
   | -------- | ------------- | --------------------------------------------- |
   | 80%      | 1.0           | Gentle — oldest results get light truncation  |
   | 85%      | 0.5           | Moderate — older results shrink significantly |
   | 90%      | 0.2           | Aggressive — most results heavily truncated   |
   | 99%      | 0.05          | Emergency — effectively one-line placeholders |

   Recency weighting: oldest tool results get 20% of the budget factor, newest get 100%.

4. **Position-based context pruning** (if `contextPruningConfig.enabled`): Additional position-based degradation of old tool results.

5. **Pruning**: `getMessagesWithinTokenLimit` drops oldest messages to fit budget. Orphan repair strips unpaired tool_use/tool_result blocks.

6. **Emergency truncation** (if pruning produces empty context): Proportional budget divided across all messages, aggressive head+tail truncation, retry pruning.

### Key Difference from Enabled Path

Messages that get pruned are **gone** — no summary captures them. The model loses context of what it did in earlier turns. This is acceptable for simpler conversations but problematic for long agentic runs with many tool calls.

---

## Summary Injection Locations

| Scenario                     | Where                                       | Why                                                        |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| Mid-run post-compaction      | `HumanMessage` when `messages.length === 0` | Clean slate; doesn't inflate `instructionTokens`           |
| Mid-run subsequent turns     | Nowhere — already consumed                  | Model read the checkpoint and is working from it           |
| Cross-run (`initialSummary`) | System prompt via `buildInstructionsString` | One-time cost; model needs it alongside user's new message |
| No summary                   | N/A                                         | Normal `[SystemMessage, ...messages]`                      |

---

## Observation Masking Details

A ToolMessage is **consumed** when a subsequent AI message exists with substantive text content — meaning the model has read and acted on the result. Detection walks backwards from the end of the messages array:

1. Find the first AI message with non-empty text content (not just tool calls).
2. All ToolMessages before that point are consumed.
3. ToolMessages after that point are unconsumed.

Masking uses `truncateToolResultContent` with a ~300 char limit, producing head+tail truncations that preserve the beginning and end of the result. This is more informative than a synthetic placeholder — the model can still see what the tool returned at a glance.

---

## Key Design Decisions

1. **Summarization IS the pruning** — when enabled, no messages are hard-pruned without being captured in a summary first. The summary replaces dropped messages.

2. **Full compaction over rolling summary** — each compaction sees the entire conversation, avoiding compound information loss from summarizing summaries-of-summaries.

3. **Summary as user message, not system prompt** — mid-run summaries are injected as a HumanMessage to avoid inflating `instructionTokens` and shrinking the available budget for messages.

4. **Observation masking for both paths** — consumed tool results are masked regardless of whether summarization is enabled. The model's own AI message text preserves what it concluded from those results.

5. **No events XML** — with full compaction the LLM sees the entire conversation each time, making structured event extraction redundant with the checkpoint's markdown content.

---

## Configuration Reference

### `librechat.yaml` — `summarization` block

| Field              | Type           | Default                    | Description                                                                                                                     |
| ------------------ | -------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`          | `boolean`      | `true`                     | Top-level kill switch. Set `false` to disable summarization globally.                                                           |
| `provider`         | `string`       | Agent's own provider       | LLM provider for the summarizer (e.g. `anthropic`, `bedrock`).                                                                  |
| `model`            | `string`       | Agent's own model          | Model for summarization calls.                                                                                                  |
| `parameters`       | `object`       | `{}`                       | Extra LLM constructor params (temperature, etc.). Also accepts `parts` and `minMessagesForSplit` for multi-stage summarization. |
| `prompt`           | `string`       | Built-in checkpoint prompt | Custom prompt for initial summarization.                                                                                        |
| `updatePrompt`     | `string`       | Built-in update prompt     | Custom prompt for re-compaction when a prior summary exists. Falls back to `prompt`.                                            |
| `trigger`          | `object`       | Always on overflow         | When to fire summarization. See trigger types below.                                                                            |
| `stream`           | `boolean`      | `true`                     | Whether to stream the summarizer's output. Set `false` to use invoke.                                                           |
| `reserveRatio`     | `number (0-1)` | `0.05`                     | Fraction of token budget reserved as headroom. Pruning triggers at `budget * (1 - reserveRatio)`.                               |
| `maxSummaryTokens` | `number`       | `2048`                     | Max output tokens for the summarization model.                                                                                  |
| `maxInputTokens`   | `number`       | `baseContextTokens`        | Max input tokens per summarization pass. Controls how much conversation context is formatted for the summarizer.                |
| `contextPruning`   | `object`       | disabled                   | Position-based context pruning (only applies when summarization is disabled).                                                   |
| `overflowRecovery` | `object`       | disabled                   | Overflow recovery retry configuration.                                                                                          |

### Trigger types (`trigger` field)

| Type                 | Value     | Behavior                                                    |
| -------------------- | --------- | ----------------------------------------------------------- |
| `token_ratio`        | `0.0-1.0` | Fire when `prePruneTotalTokens / maxContextTokens >= value` |
| `remaining_tokens`   | `number`  | Fire when `remainingContextTokens <= value`                 |
| `messages_to_refine` | `number`  | Fire when `messagesToRefine.length >= value`                |
| _(not set)_          | —         | Fire whenever pruning drops any messages (default)          |

### `contextPruning` sub-config

| Field                  | Type            | Default | Description                                                 |
| ---------------------- | --------------- | ------- | ----------------------------------------------------------- |
| `enabled`              | `boolean`       | `false` | Enable position-based tool result degradation.              |
| `keepLastAssistants`   | `number (0-10)` | —       | Number of recent assistant turns to protect from pruning.   |
| `softTrimRatio`        | `number (0-1)`  | —       | Position threshold for head+tail soft-trim.                 |
| `hardClearRatio`       | `number (0-1)`  | —       | Position threshold for full content replacement.            |
| `minPrunableToolChars` | `number`        | —       | Minimum chars before a tool result is eligible for pruning. |

### `overflowRecovery` sub-config

| Field         | Type            | Default | Description                                                 |
| ------------- | --------------- | ------- | ----------------------------------------------------------- |
| `enabled`     | `boolean`       | `false` | Enable overflow recovery retry loop.                        |
| `maxAttempts` | `number (1-10)` | —       | Max retry attempts when provider rejects oversized context. |

### `parameters` sub-fields (extracted before passing to LLM)

| Field                 | Type     | Default | Description                                                        |
| --------------------- | -------- | ------- | ------------------------------------------------------------------ |
| `parts`               | `number` | `1`     | Number of stages for multi-stage summarization. `1` = single-pass. |
| `minMessagesForSplit` | `number` | `4`     | Minimum messages before multi-stage split is applied.              |
| `maxSummaryTokens`    | `number` | `2048`  | Can also be set here (same as top-level field).                    |
