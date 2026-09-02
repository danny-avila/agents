import { AIMessageChunk } from '@langchain/core/messages';
import {
  canRestartPreempt,
  canSealPreempt,
  forgetPreemptRestartedRun,
  notePreemptRestartedRun,
  resolveMaxSeals,
  readPreemptRestartedRun,
  resolvePreemptAction,
  resolveRestartGraceMs,
} from './preempt';

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

describe('resolveRestartGraceMs', () => {
  it('defaults when unset', () => {
    expect(resolveRestartGraceMs(undefined)).toBe(2_000);
  });

  it('honors zero as a deliberate convert-immediately', () => {
    expect(resolveRestartGraceMs(0)).toBe(0);
  });

  it('clamps a negative window rather than inverting the comparison', () => {
    expect(resolveRestartGraceMs(-500)).toBe(0);
  });

  it('falls back rather than never converting', () => {
    expect(resolveRestartGraceMs(NaN)).toBe(2_000);
    expect(resolveRestartGraceMs(Infinity)).toBe(2_000);
  });
});

/**
 * The membership rule is a whitelist, so every case that is not recognizably
 * reasoning must refuse. Refusing costs a slower interrupt; wrongly discarding
 * destroys work the model already did.
 */
describe('canRestartPreempt', () => {
  describe('discards', () => {
    it('a stream that has produced nothing', () => {
      expect(canRestartPreempt(undefined)).toBe(true);
    });

    it('an empty accumulation', () => {
      expect(canRestartPreempt(chunk({ content: '' }))).toBe(true);
      expect(canRestartPreempt(chunk({ content: [] }))).toBe(true);
    });

    it('whitespace a provider emitted before its first word', () => {
      expect(canRestartPreempt(chunk({ content: '  \n' }))).toBe(true);
    });

    /** The block form of the same thing, which several providers send. */
    it('a blank text block opening the message', () => {
      expect(
        canRestartPreempt(
          chunk({
            content: [
              { type: 'text', text: '' },
              { type: 'thinking', thinking: 'working' },
            ],
          })
        )
      ).toBe(true);
    });

    /**
     * A reasoning-only Responses turn populates `output` too at natural
     * completion, and that is exactly the turn a restart should discard.
     */
    it('a Responses turn whose authoritative output is reasoning only', () => {
      const reasoningOnly = chunk({ content: [] });
      reasoningOnly.response_metadata = {
        output: [{ type: 'reasoning', id: 'rs_1' }],
      };
      expect(canRestartPreempt(reasoningOnly)).toBe(true);
    });

    it('every provider spelling of a reasoning block', () => {
      expect(
        canRestartPreempt(
          chunk({
            content: [
              { type: 'thinking', thinking: 'anthropic' },
              { type: 'redacted_thinking', data: 'ciphertext' },
              { type: 'reasoning', reasoning: 'google' },
              { type: 'reasoning_content', reasoningText: { text: 'bedrock' } },
            ],
          })
        )
      ).toBe(true);
    });
  });

  describe('refuses to discard', () => {
    it('an answer the user can already read', () => {
      expect(
        canRestartPreempt(chunk({ content: [{ type: 'text', text: 'Hi' }] }))
      ).toBe(false);
    });

    it('reasoning that has begun turning into text', () => {
      expect(
        canRestartPreempt(
          chunk({
            content: [
              { type: 'thinking', thinking: 'almost' },
              { type: 'text', text: 'The answer is' },
            ],
          })
        )
      ).toBe(false);
    });

    it('a tool call, whether complete or still streaming', () => {
      expect(
        canRestartPreempt(
          chunk({
            tool_calls: [{ id: 't1', name: 'search', args: {} }],
          })
        )
      ).toBe(false);
      expect(
        canRestartPreempt(
          chunk({
            tool_call_chunks: [{ id: 't1', name: 'search', args: '{"q"' }],
          })
        )
      ).toBe(false);
    });

    /**
     * Assigned after construction because `AIMessageChunk` derives
     * `invalid_tool_calls` from `tool_call_chunks` and drops a directly
     * supplied array — the same reason the seal gate's own case does it.
     */
    it('a tool call the provider malformed', () => {
      const malformed = chunk({ content: '' });
      malformed.invalid_tool_calls = [
        { id: 't1', name: 'search', args: '{oops', error: 'bad json' },
      ];
      expect(canRestartPreempt(malformed)).toBe(false);
    });

    /**
     * The one place this is STRICTER than the seal gate. A seal may treat a
     * settled server tool call as harmless, because the answer stays on the
     * message. Re-issuing the request would run — and bill — that search again.
     */
    it('a server-side search the provider already ran and answered', () => {
      expect(
        canRestartPreempt(
          chunk({
            tool_calls: [{ id: 'srvtoolu_1', name: 'web_search', args: {} }],
            content: [
              {
                type: 'web_search_tool_result',
                tool_use_id: 'srvtoolu_1',
                content: [],
              },
            ],
          })
        )
      ).toBe(false);
    });

    /**
     * Responses reports its built-in tools nowhere the ordinary gates look:
     * `content` stays empty and `tool_calls` is never populated, so without
     * this the turn reads as disposable and reissuing would run a completed
     * search — or a shell/apply-patch call — a second time.
     */
    it('a completed OpenAI Responses tool output', () => {
      const withSidecar = chunk({ content: [] });
      withSidecar.additional_kwargs.tool_outputs = [
        { type: 'web_search_call', id: 'ws_1', status: 'completed' },
      ];
      expect(canRestartPreempt(withSidecar)).toBe(false);

      const withAuthoritativeOutput = chunk({ content: [] });
      withAuthoritativeOutput.response_metadata = {
        output: [{ type: 'code_interpreter_call', id: 'ci_1' }],
      };
      expect(canRestartPreempt(withAuthoritativeOutput)).toBe(false);
    });

    it('an unrecognized block, rather than guessing it is disposable', () => {
      expect(
        canRestartPreempt(
          chunk({
            content: [{ type: 'image_url', image_url: { url: 'https://x/y' } }],
          })
        )
      ).toBe(false);
    });
  });
});

describe('resolvePreemptAction', () => {
  const thinking = chunk({
    content: [{ type: 'thinking', thinking: 'working on it' }],
  });
  const answering = chunk({ content: [{ type: 'text', text: 'The answer' }] });

  it('prefers a seal over a restart whenever one is available', () => {
    expect(
      resolvePreemptAction({
        chunk: answering,
        requestAgeMs: 60_000,
        graceMs: 0,
      })
    ).toBe('seal');
  });

  it('holds a thinking turn inside the window, in case text is moments away', () => {
    expect(
      resolvePreemptAction({ chunk: thinking, requestAgeMs: 500, graceMs: 2000 })
    ).toBe('none');
  });

  it('converts a thinking turn that outlives the window', () => {
    expect(
      resolvePreemptAction({
        chunk: thinking,
        requestAgeMs: 2000,
        graceMs: 2000,
      })
    ).toBe('restart');
  });

  /**
   * An empty accumulation does not prove the provider produced nothing — the
   * stream buffers a chunk ahead of the consumer. Only silence that outlives
   * the window does.
   */
  it('holds a silent turn inside the window, in case a chunk is in flight', () => {
    expect(
      resolvePreemptAction({
        chunk: undefined,
        requestAgeMs: 0,
        graceMs: 2000,
      })
    ).toBe('none');
  });

  it('converts a silent turn that outlives the window', () => {
    expect(
      resolvePreemptAction({
        chunk: undefined,
        requestAgeMs: 2001,
        graceMs: 2000,
      })
    ).toBe('restart');
  });

  it('never converts a turn holding work a restart would destroy', () => {
    expect(
      resolvePreemptAction({
        chunk: chunk({ tool_calls: [{ id: 't1', name: 'search', args: {} }] }),
        requestAgeMs: 60_000,
        graceMs: 0,
      })
    ).toBe('none');
  });
});

/**
 * The tracing layer keys on the run id rather than the thrown error, because
 * every provider adapter raises its own cancellation shape. That makes the
 * record's lifecycle the whole contract: exactly once, and bounded for hosts
 * that never consume it.
 */
describe('preempt-restarted run records', () => {
  /**
   * A run can be closed through several tracing handlers at once, so every
   * read must see the record — a consuming read would let the first handler
   * recognize the restart and leave the rest exporting an error.
   */
  it('reports a recorded run to every reader, with its turn', () => {
    const message = chunk({ content: 'thinking' });
    notePreemptRestartedRun('run-a', message);

    expect(readPreemptRestartedRun('run-a')?.message).toBe(message);
    expect(readPreemptRestartedRun('run-a')?.message).toBe(message);
  });

  it('does not claim a run it never recorded', () => {
    expect(readPreemptRestartedRun('run-never-restarted')).toBeUndefined();
  });

  it('forgets a record whose run closed another way', () => {
    notePreemptRestartedRun('run-manual', chunk({ content: '' }));
    forgetPreemptRestartedRun('run-manual');

    expect(readPreemptRestartedRun('run-manual')).toBeUndefined();
  });

  /**
   * The consumer is a non-awaited callback, so a burst of concurrent restarts
   * must NOT push each other out: an evicted marker closes its generation as
   * an error, which is the failure the marker exists to prevent.
   */
  it('keeps every record while its callback could still fire', () => {
    for (let i = 0; i < 500; i++) {
      notePreemptRestartedRun(`bulk-${i}`, chunk({ content: '' }));
    }

    expect(readPreemptRestartedRun('bulk-0')).toBeDefined();
    expect(readPreemptRestartedRun('bulk-499')).toBeDefined();
  });

  /** Age is what separates a queued callback from a leak. */
  it('expires a record nothing ever consumed', () => {
    const realNow = Date.now;
    try {
      const base = realNow();
      Date.now = () => base;
      notePreemptRestartedRun('stale', chunk({ content: '' }));
      Date.now = () => base + 300_001;
      notePreemptRestartedRun('fresh', chunk({ content: '' }));

      expect(readPreemptRestartedRun('stale')).toBeUndefined();
      expect(readPreemptRestartedRun('fresh')).toBeDefined();
    } finally {
      Date.now = realNow;
    }
  });
});
