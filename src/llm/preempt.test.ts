import { AIMessageChunk } from '@langchain/core/messages';
import { canSealPreempt, resolveMaxSeals } from './preempt';

/**
 * The budget is read by two consumers that interpret it differently — a
 * numeric comparison in the seal gate, and an addition into the graph's
 * recursion limit. Normalizing once keeps them in agreement.
 */
describe('resolveMaxSeals', () => {
  it('defaults when unset', () => {
    expect(resolveMaxSeals(undefined)).toBe(8);
  });

  it('passes a sane whole number through', () => {
    expect(resolveMaxSeals(3)).toBe(3);
  });

  it('floors a fractional budget so both readers agree', () => {
    expect(resolveMaxSeals(1.5)).toBe(1);
  });

  it('honors zero as a deliberate never-seal', () => {
    expect(resolveMaxSeals(0)).toBe(0);
  });

  it('clamps a negative budget to never-seal', () => {
    expect(resolveMaxSeals(-4)).toBe(0);
  });

  it('falls back rather than poisoning the recursion limit', () => {
    expect(resolveMaxSeals(NaN)).toBe(8);
    expect(resolveMaxSeals(Infinity)).toBe(8);
    expect(resolveMaxSeals(-Infinity)).toBe(8);
  });
});

function chunk(fields: Partial<AIMessageChunk>): AIMessageChunk {
  return new AIMessageChunk({
    content: '',
    ...fields,
  } as ConstructorParameters<typeof AIMessageChunk>[0]);
}

describe('canSealPreempt', () => {
  describe('refuses to seal', () => {
    it('on a missing chunk', () => {
      expect(canSealPreempt(undefined)).toBe(false);
    });

    it('on empty string content', () => {
      expect(canSealPreempt(chunk({ content: '' }))).toBe(false);
    });

    it('on whitespace-only string content', () => {
      expect(canSealPreempt(chunk({ content: '  \n\t ' }))).toBe(false);
    });

    it('on an empty content array', () => {
      expect(canSealPreempt(chunk({ content: [] }))).toBe(false);
    });

    it('on whitespace-only text blocks', () => {
      expect(
        canSealPreempt(chunk({ content: [{ type: 'text', text: '   ' }] }))
      ).toBe(false);
    });

    it('when the only content is thinking', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'thinking', thinking: 'Let me work through this.' },
            ],
          })
        )
      ).toBe(false);
    });

    it('when the only content is reasoning', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [{ type: 'reasoning', reasoning: 'Considering.' }],
          })
        )
      ).toBe(false);
    });

    it('when the only content is redacted_thinking', () => {
      expect(
        canSealPreempt(
          chunk({ content: [{ type: 'redacted_thinking', data: 'xxx' }] })
        )
      ).toBe(false);
    });

    it('with a resolved tool call present', () => {
      expect(
        canSealPreempt(
          chunk({
            content: 'Looking that up.',
            tool_calls: [{ id: 'c1', name: 'search', args: {} }],
          })
        )
      ).toBe(false);
    });

    it('with a partial tool call still streaming', () => {
      expect(
        canSealPreempt(
          chunk({
            content: 'Looking that up.',
            tool_call_chunks: [
              { id: 'c1', name: 'search', args: '{"q"', index: 0 },
            ],
          })
        )
      ).toBe(false);
    });

    /**
     * Assigned after construction on purpose: `AIMessageChunk` derives
     * `invalid_tool_calls` from `tool_call_chunks` and drops a directly
     * supplied array, so this is the only way to exercise the guard on its
     * own rather than through the `tool_call_chunks` check above it.
     */
    it('with an invalid tool call present', () => {
      const c = chunk({ content: 'Looking that up.' });
      c.invalid_tool_calls = [
        { id: 'c1', name: 'search', args: '{oops', error: 'bad json' },
      ];
      expect(canSealPreempt(c)).toBe(false);
    });

    /**
     * Anthropic emits a `tool_call_chunk` alongside every `server_tool_use`
     * content block, and `concat` keeps it on the accumulated message for the
     * rest of the turn. So a web-search turn is refused by the tool-call gate,
     * both while the search is in flight and after its result has landed —
     * this pins that invariant rather than a content-block check.
     */
    it('while an Anthropic server tool is in flight', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Searching the web for that.' },
              {
                type: 'server_tool_use',
                id: 'srvtoolu_1',
                name: 'web_search',
                input: { query: 'x' },
              },
            ],
            tool_call_chunks: [
              { id: 'srvtoolu_1', name: 'web_search', args: '', index: 0 },
            ],
          })
        )
      ).toBe(false);
    });

    /**
     * Gemini server-side tools land as `toolCall` / `toolResponse` content
     * blocks and never populate `tool_calls` or `tool_call_chunks`, so the
     * tool-call gates above cannot see them.
     */
    it('while a Google server tool call is unanswered', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Let me look that up.' },
              { type: 'toolCall', id: 'gcall_1', name: 'google_search' },
            ],
          })
        )
      ).toBe(false);
    });

    it('with more Google server tool calls than responses', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Checking two things.' },
              { type: 'toolCall', id: 'gcall_1', name: 'google_search' },
              { type: 'toolResponse', id: 'gcall_1', response: {} },
              { type: 'toolCall', id: 'gcall_2', name: 'google_search' },
            ],
          })
        )
      ).toBe(false);
    });

    it('while one of several server tool calls is still unanswered', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Checking two sources.' },
              {
                type: 'web_search_tool_result',
                tool_use_id: 'srvtoolu_1',
                content: [],
              },
            ],
            tool_call_chunks: [
              { id: 'srvtoolu_1', name: 'web_search', args: '', index: 0 },
              { id: 'srvtoolu_2', name: 'web_search', args: '', index: 1 },
            ],
          })
        )
      ).toBe(false);
    });
  });

  describe('seals', () => {
    it('on non-empty string content', () => {
      expect(canSealPreempt(chunk({ content: 'The migration has' }))).toBe(
        true
      );
    });

    it('on a non-empty text block', () => {
      expect(
        canSealPreempt(
          chunk({ content: [{ type: 'text', text: 'The migration has' }] })
        )
      ).toBe(true);
    });

    it('on text that follows signed thinking', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              {
                type: 'thinking',
                thinking: 'Working it out.',
                signature: 'sig',
              },
              { type: 'text', text: 'The migration has' },
            ],
          })
        )
      ).toBe(true);
    });

    it('with empty tool-call arrays present', () => {
      expect(
        canSealPreempt(
          chunk({
            content: 'The migration has',
            tool_calls: [],
            tool_call_chunks: [],
            invalid_tool_calls: [],
          })
        )
      ).toBe(true);
    });

    it('once every Google server tool call has been answered', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Here is what I found.' },
              { type: 'toolCall', id: 'gcall_1', name: 'google_search' },
              { type: 'toolResponse', id: 'gcall_1', response: {} },
            ],
          })
        )
      ).toBe(true);
    });

    /**
     * Provider-side tools never reach `ToolNode`, so no `PostToolBatch`
     * boundary exists to drain into. Refusing forever after a search would
     * defer a queued message to the end of the turn rather than to the next
     * tool step, so a call whose result has landed counts as settled.
     */
    it('once an Anthropic server tool result has landed', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Here is what I found.' },
              {
                type: 'server_tool_use',
                id: 'srvtoolu_1',
                name: 'web_search',
                input: { query: 'x' },
              },
              {
                type: 'web_search_tool_result',
                tool_use_id: 'srvtoolu_1',
                content: [],
              },
            ],
            tool_call_chunks: [
              { id: 'srvtoolu_1', name: 'web_search', args: '', index: 0 },
            ],
          })
        )
      ).toBe(true);
    });

    it('on a non-text block that carries no tool call', () => {
      expect(
        canSealPreempt(
          chunk({
            content: [
              { type: 'text', text: 'Here is the diagram.' },
              { type: 'image_url', image_url: { url: 'https://x/y.png' } },
            ],
          })
        )
      ).toBe(true);
    });
  });
});
