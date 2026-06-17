# Prompt-cache strategy benchmark

This documents _why_ the default prompt-cache strategy is a **single tail
breakpoint** anchored on the conversation tail (the Claude Code approach),
rather than the legacy **"last two user messages"** markers — and how to
reproduce the comparison live against a real provider.

- Tail strategy: [`addTailCacheControl`](../src/messages/cache.ts) /
  [`addBedrockTailCacheControl`](../src/messages/cache.ts)
- Legacy strategy: `addCacheControl` / `addBedrockCacheControl` (still exported
  for back-compat)
- Benchmark: [`src/scripts/bench-prompt-cache.ts`](../src/scripts/bench-prompt-cache.ts)

## Why the tail strategy wins

A prompt-cache breakpoint caches everything _before_ it; the provider then
reads back the longest matching cached prefix on the next call, automatically,
regardless of where that call's own breakpoints sit.

- **Legacy** pins markers on the **last two user messages**. In an agent tool
  loop there is often only **one** user message for many turns, so the only
  breakpoint sits at the top of the conversation. Every assistant/tool turn
  appended afterwards falls _outside_ the cached prefix and is re-sent
  **uncached (full price) on every subsequent call**. Cache write/fresh ≫ read.
- **Tail** rides the true tail, so the transcript is written to cache once and
  read back as history grows append-only. Freshly appended turns enter the
  cached prefix on the next call instead of being reprocessed.

This is the dominant agent shape (one request → many tool calls), which is
exactly where the legacy approach degrades hardest.

## Metric

For each model call the provider reports a token breakdown. Summed per scenario:

- `read` — tokens served from cache (**higher is better**)
- `write` — tokens written to cache (`cache_creation`)
- `fresh` — uncached input processed at full price; this balloons when caching
  fails to cover the transcript (**lower is better**)
- `effective` — a cost proxy in input-token-equivalents using the published
  multipliers: `read ×0.1 + write ×1.25 + fresh ×1.0` (**lower is better**)

`fresh` is computed provider-agnostically as
`(total_tokens − output_tokens) − cache_read − cache_creation`. This matters
because the two providers report `input_tokens` differently: Anthropic folds the
cached tokens _into_ `input_tokens`, while Bedrock reports `input_tokens` as the
fresh delta only with the cache buckets separate. Deriving `fresh` from
`total_tokens` is correct on both.

## Representative results

Live, `claude-sonnet-4-5`, `rounds=6` (exact counts vary run to run; the
direction is stable). `effective` is the headline — lower is cheaper.

### Anthropic

| Scenario                                     | strategy |    read |  write |      fresh |          effective |
| -------------------------------------------- | -------- | ------: | -----: | ---------: | -----------------: |
| Agent tool loop (1 user turn, N tool rounds) | legacy   |  98,352 | 24,588 | **46,221** |             86,791 |
|                                              | tail     | 137,363 | 31,801 |     **33** |  **53,521** (−38%) |
| Multi-turn chat (frequent user messages)     | legacy   |  90,478 | 23,662 | **21,595** |             60,220 |
|                                              | tail     | 104,555 | 22,162 |     **18** |  **38,176** (−37%) |
| Realistic agent (user turns + tool rounds)   | legacy   | 498,440 | 39,525 | **50,654** |            149,904 |
|                                              | tail     | 519,827 | 41,702 |     **90** | **104,200** (−30%) |

### Bedrock (Converse)

| Scenario                                     | strategy |    read |  write |      fresh |         effective |
| -------------------------------------------- | -------- | ------: | -----: | ---------: | ----------------: |
| Agent tool loop (1 user turn, N tool rounds) | legacy   | 100,430 | 20,086 | **21,618** |            56,768 |
|                                              | tail     | 122,308 | 28,778 |     **33** | **48,236** (−15%) |
| Multi-turn chat (frequent user messages)     | legacy   | 119,565 | 25,164 |         18 |            43,430 |
|                                              | tail     | 104,555 | 22,162 |         18 | **38,176** (−12%) |
| Realistic agent (user turns + tool rounds)   | legacy   | 521,423 | 39,514 | **27,556** |           129,091 |
|                                              | tail     | 596,553 | 46,228 |     **90** | **117,530** (−9%) |

The tail strategy is cheaper (lower `effective`) in **every** scenario on both
providers. The clearest signal is `fresh`: the legacy approach reprocesses tens
of thousands of full-price tokens in any tool-bearing conversation, which the
tail strategy reduces to near zero. Even the legacy strong case (frequent user
messages, no tools) is a tie-or-win for the tail strategy.

## Reproduce

Requires real credentials in `.env` (or point `BENCH_ENV_FILE` at one):
`ANTHROPIC_API_KEY` for Anthropic, `BEDROCK_AWS_ACCESS_KEY_ID` /
`BEDROCK_AWS_SECRET_ACCESS_KEY` (and a region) for Bedrock. It makes real, paid
API calls and is **not** a unit test (CI never runs it).

```bash
npm run bench:cache                          # Anthropic (default)
npm run bench:cache -- --provider bedrock     # Bedrock Converse
npm run bench:cache -- --rounds 10 --model claude-sonnet-4-5
```

Each scenario runs the _same_ conversation under both strategies in separate
cache namespaces (unique per run), then prints the per-strategy totals and the
delta.
