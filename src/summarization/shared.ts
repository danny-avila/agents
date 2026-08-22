/**
 * Summarization primitives shared by the in-run summarize node and by callers
 * that compact a conversation outside a run. Kept apart from `node.ts` so the
 * package can export them without exporting the graph node itself.
 */

/**
 * Token overhead of the XML wrapper + instruction text added around the
 * summary at injection time in AgentContext.buildSystemRunnable:
 * `<summary>\n${text}\n</summary>\n\nYour context window was compacted...`
 * ~33 tokens on Anthropic, ~24-27 on OpenAI.  Using 33 as a safe ceiling.
 */
export const SUMMARY_WRAPPER_OVERHEAD_TOKENS = 33;

/** Structured checkpoint prompt for fresh summarization (no prior summary). */
export const DEFAULT_SUMMARIZATION_PROMPT = `Hold on, before you continue I need you to write me a checkpoint of everything so far. Your context window is filling up and this checkpoint replaces the messages above, so capture everything you need to pick right back up.

Don't second-guess or fact-check anything you did, your tool results reflect exactly what happened. If a tool result appears truncated, that's just a display artifact from context management: the tool executed fully. Just record what you did and what you observed. Only the checkpoint, don't respond to me or continue the conversation.

## Checkpoint

## Goal
What I asked you to do and any sub-goals you identified.

## Constraints & Preferences
Any rules, preferences, or configuration I established.

## Progress
### Done
- What you completed and the outcomes

### In Progress
- What you're currently working on

## Key Decisions
Decisions you made and why.

## Next Steps
Concrete task actions remaining, in priority order.

## Critical Context
Exact identifiers, names, error messages, URLs, and details you need to preserve verbatim.

Rules:
- Record what you did and observed, don't judge or re-evaluate it
- For each tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Short declarative sentences
- Skip empty sections`;

/** Prompt for re-compaction when a prior summary exists. */
export const DEFAULT_UPDATE_SUMMARIZATION_PROMPT = `Hold on again, update your checkpoint. Merge the new messages into your existing checkpoint and give me a single consolidated replacement.

Keep it roughly the same length as your last checkpoint. Compress older details to make room for what's new, don't just append. Give recent actions more detail, compress older items to one-liners.

Don't fact-check or second-guess anything, your tool results are ground truth. If a tool result appears truncated, that's just a display artifact: the tool executed fully. Only the checkpoint, don't respond to me or continue the conversation.

Rules:
- Merge new progress into existing sections, don't duplicate headers
- Compress older completed items into one-line entries
- Move items from "In Progress" to "Done" when you completed them
- Update "Next Steps" to reflect current task priorities.
- For each new tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Skip empty sections`;

const SUMMARIZATION_PARAM_KEYS = new Set(['maxSummaryTokens']);

export function separateSummarizationParameters(
  parameters: Record<string, unknown>
): {
  llmParams: Record<string, unknown>;
  maxSummaryTokens?: number;
} {
  const llmParams: Record<string, unknown> = {};
  let maxSummaryTokens: number | undefined;

  for (const [key, value] of Object.entries(parameters)) {
    if (SUMMARIZATION_PARAM_KEYS.has(key)) {
      if (
        key === 'maxSummaryTokens' &&
        typeof value === 'number' &&
        value > 0
      ) {
        maxSummaryTokens = value;
      }
    } else {
      llmParams[key] = value;
    }
  }

  return { llmParams, maxSummaryTokens };
}

export function buildSummarizationInstruction(
  promptText: string,
  updatePromptText: string | undefined,
  priorSummaryText?: string
): string {
  const prior = priorSummaryText?.trim() ?? '';
  const effectivePrompt = prior ? (updatePromptText ?? promptText) : promptText;
  const parts = [effectivePrompt];
  if (prior) {
    parts.push(`\n\n<previous-summary>\n${prior}\n</previous-summary>`);
  }
  return parts.join('');
}
