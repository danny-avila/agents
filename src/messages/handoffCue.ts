// src/messages/handoffCue.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';

/**
 * Bracketed-meta convention, like the handoff path's
 * `[Processed tool result and transferring to …]` bridge. The wording makes
 * two things unambiguous to the model: the assistant turn above is FINISHED,
 * and it belongs to a previous stage — so the successor answers as itself
 * instead of continuing someone else's sentence.
 */
export const PREDECESSOR_HANDOFF_CUE =
  '[The assistant message above is the completed output of a previous ' +
  'agent. Respond now according to your own role and instructions.]';

/**
 * Appends a user-turn handoff cue when a payload ends with an assistant turn
 * that THIS RUN produced — which only happens when a different agent in a
 * multi-agent workflow produced it (an agent's own self-loops always re-enter
 * on a tool result or an injected user turn).
 *
 * Why: providers with prefill semantics (Anthropic, Bedrock-Claude) treat a
 * trailing assistant message as a prefill and CONTINUE it. A bare direct-edge
 * successor therefore speaks in its predecessor's voice — or, when the
 * trailing turn reads complete (a preemption steer's short resume, say),
 * returns empty content (danny-avila/agents#345, reproduced live 3/3).
 * Handoff edges with instructions and prompt-instruction edges already break
 * the prefill with a user turn; this closes the same gap for bare edges.
 *
 * Fail-safe OFF by identity: the trailing payload message must carry the same
 * id as the run's last recorded message (`runTail`, from the graph's
 * non-allocating `getLastRunMessage()`). Host-supplied trailing assistant
 * turns (deliberate prefill flows) never match — the run has not produced
 * them — so single-agent prefill behavior is untouched. Wire-only: the cue is
 * appended to the provider projection, never to graph state or host history.
 */
export function appendPredecessorHandoffCue(
  messages: BaseMessage[],
  runTail: BaseMessage | undefined
): BaseMessage[] {
  const last = messages.at(-1);
  if (last == null || last.getType() !== 'ai') {
    return messages;
  }
  const lastRun = runTail;
  if (lastRun == null || lastRun.getType() !== 'ai') {
    return messages;
  }
  if (last.id == null || lastRun.id == null || last.id !== lastRun.id) {
    return messages;
  }
  return [
    ...messages,
    new HumanMessage({
      content: PREDECESSOR_HANDOFF_CUE,
      additional_kwargs: { role: 'user', isMeta: true, source: 'handoff' },
    }),
  ];
}
