# ADR 0005: Select Pairing-Balanced Compaction Ranges

## Status

Accepted

## Context

Compaction previously cut only at user-message boundaries and always retained
the complete most recent user-led turn. That protected a lone pasted payload,
but it also made a normal first-turn agent loop indivisible: dozens of closed
assistant tool calls and results exposed no compactable prefix. Overflow
recovery then had no summarization path even though most of the turn was
completed history.

A benchmark with 20, 50, and 100 tool steps measured zero compactable tokens
under the turn-only policy. Pairing-balanced cuts exposed 79.8% to 83.0% of the
same histories while preserving every call/result pair.

## Decision

Compaction range selection continues to prefer whole user-led turns. Only when
that policy exposes no head, a token counter and context window are available,
and at least one tool call has a matching result, selection may fall back to a
boundary inside the earliest retained turn.

The fallback selects a contiguous prefix and may end only while no tool call is
pending. Parallel calls remain pending until every matching result arrives, and
an open call remains in the retained tail. The tail retains the configured
`retainRecent.tokens` budget, or 16% of the context window when no explicit
budget exists. A lone user message has no closed tool unit and remains intact.

Range pricing reuses the `AgentContext` exact-token cache when its counter is
compatible. Summarization, state replacement, source coverage, hooks, and
Langfuse observations stay behind the existing summarize node.

## Consequences

- Long single-turn tool loops gain a usable overflow and pressure-compaction
  path instead of deterministically exposing an empty head.
- Recent raw evidence remains verbatim, and tool call/result pairing is valid
  on both sides of the boundary.
- The initial user request can enter the summarized prefix after completed tool
  work exists; the checkpoint prompt must continue preserving the goal and
  exact constraints. User-only payloads retain the previous protection.
- Selection adds one bounded linear scan only after compaction has already
  fired and the turn-level policy found no head.
- Exact system + tools + messages replay for provider cache reuse remains a
  separate request-projection concern.
