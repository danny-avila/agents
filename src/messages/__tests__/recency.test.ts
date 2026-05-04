import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  SystemMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { splitAtRecencyBoundary } from '@/messages/recency';

describe('splitAtRecencyBoundary', () => {
  describe('default behavior (turns: 2)', () => {
    it('returns empty head and full tail for an empty array', () => {
      const result = splitAtRecencyBoundary([], { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
      expect(result.tailStartIndex).toBe(0);
    });

    it('always preserves the most recent turn even with one large message', () => {
      const messages = [new HumanMessage('huge first message'.repeat(1000))];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('keeps a complete user-assistant exchange in the tail', () => {
      const messages = [new HumanMessage('hi'), new AIMessage('hello')];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('places older turns in the head when there are more turns than the cap', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage('reply 1'),
        new HumanMessage('turn 2'),
        new AIMessage('reply 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual(messages.slice(0, 2));
      expect(result.tail).toEqual(messages.slice(2));
      expect(result.tailTurnCount).toBe(2);
      expect(result.tailStartIndex).toBe(2);
    });

    it('preserves tool_use / tool_result pairs across the boundary', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_a', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result A',
          tool_call_id: 'call_a',
          name: 'search',
        }),
        new AIMessage('done with turn 1'),
        new HumanMessage('turn 2'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'call_b', name: 'search', args: {} }],
        }),
        new ToolMessage({
          content: 'result B',
          tool_call_id: 'call_b',
          name: 'search',
        }),
        new AIMessage('done with turn 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      // Head must contain turn 1's complete tool_use → tool_result pair.
      expect(result.head).toHaveLength(4);
      expect(result.head[0]).toBe(messages[0]);
      expect(result.head[3]).toBe(messages[3]);
      // Tail starts cleanly at turn 2's HumanMessage — never mid-pair.
      expect(result.tail[0]).toBe(messages[4]);
      expect(result.tail).toHaveLength(6);
    });
  });

  describe('disabled (turns: 0)', () => {
    it('puts everything in head when turns is 0', () => {
      const messages = [
        new HumanMessage('one'),
        new AIMessage('two'),
        new HumanMessage('three'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 0 });
      expect(result.head).toEqual(messages);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
    });

    it('treats negative turns as 0', () => {
      const messages = [new HumanMessage('a'), new AIMessage('b')];
      const result = splitAtRecencyBoundary(messages, { turns: -5 });
      expect(result.tail).toEqual([]);
      expect(result.head).toEqual(messages);
    });
  });

  describe('token cap', () => {
    it('honors the token cap when adding older turns', () => {
      const messages = [
        new HumanMessage('turn 1'),
        new AIMessage('reply 1'),
        new HumanMessage('turn 2'),
        new AIMessage('reply 2'),
        new HumanMessage('turn 3'),
        new AIMessage('reply 3'),
      ];
      const tokenCounter = (): number => 100;
      const result = splitAtRecencyBoundary(messages, {
        turns: 5,
        tokens: 250,
        tokenCounter,
      });
      // Last turn is always preserved (200 tokens for 2 messages).
      // Adding turn 2 would push to 400, exceeding cap of 250 → stop.
      expect(result.tailTurnCount).toBe(1);
      expect(result.tail).toEqual(messages.slice(4));
    });

    it('always preserves the most recent turn even when it exceeds the cap', () => {
      const messages = [new HumanMessage('huge'), new AIMessage('also huge')];
      const tokenCounter = (): number => 1_000_000;
      const result = splitAtRecencyBoundary(messages, {
        turns: 2,
        tokens: 10,
        tokenCounter,
      });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
      expect(result.tailTurnCount).toBe(1);
    });

    it('ignores the token cap when no tokenCounter is provided', () => {
      const messages = [
        new HumanMessage('a'),
        new AIMessage('b'),
        new HumanMessage('c'),
        new AIMessage('d'),
      ];
      const result = splitAtRecencyBoundary(messages, {
        turns: 3,
        tokens: 1, // would force tail to most-recent-only if applied
      });
      // No tokenCounter → fall back to turn-based selection only.
      expect(result.tailTurnCount).toBe(2);
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
    });
  });

  describe('linearity', () => {
    it('calls tokenCounter once per message in visited turns (no quadratic recount)', () => {
      // Build a long history: 200 turns × 10 messages = 2,000 messages.
      // If the boundary search were quadratic in the number of turns,
      // the call count would explode (e.g., 200 × 2,000 = 400,000).
      // The disjoint-slice invariant guarantees one call per visited
      // message, bounded by messages.length even with a generous turn
      // budget that visits every turn.
      const messages: BaseMessage[] = [];
      const turnCount = 200;
      const messagesPerTurn = 10;
      for (let t = 0; t < turnCount; t++) {
        messages.push(new HumanMessage(`turn ${t} query`));
        for (let m = 1; m < messagesPerTurn; m++) {
          messages.push(new AIMessage(`turn ${t} reply ${m}`));
        }
      }

      let calls = 0;
      const tokenCounter = (): number => {
        calls += 1;
        return 1;
      };

      // Generous tokens cap so the loop visits every turn.
      // turnsCap also generous so the limit isn't hit early.
      splitAtRecencyBoundary(messages, {
        turns: 1_000_000,
        tokens: 1_000_000,
        tokenCounter,
      });

      // Strictly bounded by messages.length.  No message is counted
      // twice, regardless of how many turns the splitter walks.
      expect(calls).toBeLessThanOrEqual(messages.length);
      expect(calls).toBe(messages.length);
    });

    it('stops counting once the tokens cap is exceeded (no scan past the boundary)', () => {
      const messages: BaseMessage[] = [];
      for (let t = 0; t < 50; t++) {
        messages.push(new HumanMessage(`turn ${t}`));
        messages.push(new AIMessage(`reply ${t}`));
      }

      let calls = 0;
      const tokenCounter = (): number => {
        calls += 1;
        return 1; // 1 token per message → 100 tokens total
      };

      // Cap of 10 tokens lets us include the last 5 turns (10 messages)
      // before the next turn's 2 tokens would overflow.
      const result = splitAtRecencyBoundary(messages, {
        turns: 1_000,
        tokens: 10,
        tokenCounter,
      });

      // Visited at most: 5 included turns × 2 messages + one over-budget
      // turn × 2 messages (counted then rejected) = 12 messages.  Far
      // less than the full 100.
      expect(calls).toBeLessThanOrEqual(12);
      expect(result.tailTurnCount).toBe(5);
    });
  });

  describe('degenerate inputs', () => {
    it('puts everything in the head when there is no HumanMessage', () => {
      const messages = [
        new SystemMessage('preamble'),
        new AIMessage('starter'),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 2 });
      expect(result.head).toEqual(messages);
      expect(result.tail).toEqual([]);
      expect(result.tailTurnCount).toBe(0);
    });

    it('handles a HumanMessage at index 0 with prior non-human messages absent', () => {
      const messages = [new HumanMessage('only')];
      const result = splitAtRecencyBoundary(messages, { turns: 3 });
      expect(result.head).toEqual([]);
      expect(result.tail).toEqual(messages);
    });

    it('handles tool messages as the very last messages', () => {
      const messages = [
        new HumanMessage('q1'),
        new AIMessage('a1'),
        new HumanMessage('q2'),
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'c1', name: 't', args: {} }],
        }),
        new ToolMessage({ content: 'r', tool_call_id: 'c1', name: 't' }),
      ];
      const result = splitAtRecencyBoundary(messages, { turns: 1 });
      // Most recent turn includes the trailing tool result.
      expect(result.tail).toEqual(messages.slice(2));
      expect(result.head).toEqual(messages.slice(0, 2));
    });
  });
});
