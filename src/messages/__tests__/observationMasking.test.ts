import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { TokenCounter } from '@/types/run';
import { maskConsumedToolResults } from '@/messages/prune';

const charCounter: TokenCounter = (msg) => {
  const raw = msg.content;
  if (typeof raw === 'string') return raw.length;
  return 0;
};

function toolMsg(
  content: string,
  name = 'tool',
  toolCallId = `tc_${Math.random().toString(36).slice(2, 8)}`
): ToolMessage {
  return new ToolMessage({ content, tool_call_id: toolCallId, name });
}

function aiWithText(text: string): AIMessage {
  return new AIMessage(text);
}

function aiToolCall(toolCallId: string, name = 'tool'): AIMessage {
  return new AIMessage({
    content: [{ type: 'tool_use', id: toolCallId, name, input: {} }],
    tool_calls: [{ id: toolCallId, name, args: {}, type: 'tool_call' }],
  });
}

describe('maskConsumedToolResults', () => {
  it('masks consumed tool results (followed by AI with text)', () => {
    const tcId = 'tc_1';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId, 'search'),
      toolMsg('A'.repeat(1000), 'search', tcId),
      aiWithText('Based on the search results, here is the answer.'),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 1000,
      3: 50,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    expect(count).toBe(1);
    const maskedContent = messages[2].content as string;
    expect(maskedContent.length).toBeLessThan(1000);
    expect(maskedContent.length).toBeLessThanOrEqual(300);
  });

  it('does NOT mask unconsumed tool results (no subsequent AI text)', () => {
    const tcId = 'tc_1';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId, 'search'),
      toolMsg('A'.repeat(1000), 'search', tcId),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 1000,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    expect(count).toBe(0);
    expect((messages[2].content as string).length).toBe(1000);
  });

  it('does NOT mask tool results followed by AI with only tool calls (no text)', () => {
    const tcId1 = 'tc_1';
    const tcId2 = 'tc_2';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId1, 'search'),
      toolMsg('A'.repeat(1000), 'search', tcId1),
      aiToolCall(tcId2, 'fetch'),
      toolMsg('B'.repeat(500), 'fetch', tcId2),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 1000,
      3: 20,
      4: 500,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    // No AI message with substantive text exists, so nothing is consumed
    expect(count).toBe(0);
  });

  it('masks multiple consumed results before a text AI response', () => {
    const tcId1 = 'tc_1';
    const tcId2 = 'tc_2';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId1, 'search'),
      toolMsg('A'.repeat(1000), 'search', tcId1),
      aiToolCall(tcId2, 'fetch'),
      toolMsg('B'.repeat(800), 'fetch', tcId2),
      aiWithText('Here are the combined results.'),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 1000,
      3: 20,
      4: 800,
      5: 30,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    expect(count).toBe(2);
    expect((messages[2].content as string).length).toBeLessThanOrEqual(300);
    expect((messages[4].content as string).length).toBeLessThanOrEqual(300);
  });

  it('never masks AI messages', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiWithText('A'.repeat(2000)),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 2000,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    expect(count).toBe(0);
    expect((messages[1].content as string).length).toBe(2000);
  });

  it('skips short tool results below maxChars threshold', () => {
    const tcId = 'tc_1';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId, 'search'),
      toolMsg('short result', 'search', tcId),
      aiWithText('Got it.'),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 12,
      3: 7,
    };

    const count = maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    expect(count).toBe(0);
    expect(messages[2].content).toBe('short result');
  });

  it('updates indexTokenCountMap for masked messages', () => {
    const tcId = 'tc_1';
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      aiToolCall(tcId, 'search'),
      toolMsg('A'.repeat(2000), 'search', tcId),
      aiWithText('Summary of results.'),
    ];
    const map: Record<string, number | undefined> = {
      0: 5,
      1: 20,
      2: 2000,
      3: 20,
    };

    maskConsumedToolResults({
      messages,
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });

    // Token count should be updated to match the masked content length
    expect(map[2]).toBeLessThan(2000);
    expect(map[2]).toBe((messages[2].content as string).length);
  });

  it('handles empty messages array', () => {
    const map: Record<string, number | undefined> = {};
    const count = maskConsumedToolResults({
      messages: [],
      indexTokenCountMap: map,
      tokenCounter: charCounter,
    });
    expect(count).toBe(0);
  });
});
