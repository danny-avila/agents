import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  appendPredecessorHandoffCue,
  PREDECESSOR_HANDOFF_CUE,
} from './handoffCue';

describe('appendPredecessorHandoffCue', () => {
  const runAi = new AIMessage({ content: 'predecessor output', id: 'run-ai-1' });

  it('appends the cue when the payload ends with the run-produced assistant turn', () => {
    const payload = [new HumanMessage('ask'), runAi];
    const result = appendPredecessorHandoffCue(payload, [runAi]);
    expect(result).toHaveLength(3);
    expect(result[2].content).toBe(PREDECESSOR_HANDOFF_CUE);
    expect(result[2].additional_kwargs).toEqual({
      role: 'user',
      isMeta: true,
      source: 'handoff',
    });
    /** Non-mutating: the input array is untouched. */
    expect(payload).toHaveLength(2);
  });

  it('matches a transformed clone of the run tail by id', () => {
    const clone = new AIMessage({ content: 'projected copy', id: 'run-ai-1' });
    const result = appendPredecessorHandoffCue([clone], [runAi]);
    expect(result).toHaveLength(2);
  });

  /**
   * Host-supplied trailing assistant turns are deliberate prefill (continue
   * generation flows). The run has not produced them, so the id never
   * matches and the cue must stay off.
   */
  it('leaves host-history prefill payloads alone', () => {
    const hostAi = new AIMessage({ content: 'host prefill', id: 'host-ai-9' });
    expect(appendPredecessorHandoffCue([hostAi], [runAi])).toHaveLength(1);
    expect(appendPredecessorHandoffCue([hostAi], [])).toHaveLength(1);
    expect(appendPredecessorHandoffCue([hostAi], undefined)).toHaveLength(1);
  });

  it('is off when the payload does not end on an assistant turn', () => {
    const tool = new ToolMessage({ content: 'r', tool_call_id: 't1' });
    expect(
      appendPredecessorHandoffCue([runAi, tool], [runAi, tool])
    ).toHaveLength(2);
    expect(
      appendPredecessorHandoffCue([new HumanMessage('hi')], [runAi])
    ).toHaveLength(1);
    expect(appendPredecessorHandoffCue([], [runAi])).toHaveLength(0);
  });

  it('fails safe when ids are missing on either side', () => {
    const noId = new AIMessage({ content: 'no id' });
    expect(appendPredecessorHandoffCue([noId], [noId])).toHaveLength(1);
  });

  it('is off when the run tail is not an assistant turn', () => {
    const runTail = new HumanMessage({ content: 'steer', id: 'h1' });
    const ai = new AIMessage({ content: 'x', id: 'h1' });
    expect(appendPredecessorHandoffCue([ai], [runTail])).toHaveLength(1);
  });
});
