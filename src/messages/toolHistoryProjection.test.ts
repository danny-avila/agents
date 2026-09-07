import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { ToolHistoryCallMirrors } from './toolHistoryProjection';
import {
  getResponsesHistorySource,
  projectResponsesHistory,
  createToolHistoryPreparation,
  recordToolHistoryCallMirror,
  isToolHistoryCallMirror,
} from './toolHistoryProjection';

describe('Ordered Tool History Projection', () => {
  test('projects computer actions as model-authored calls', () => {
    const action = { type: 'click', x: 10, y: 20 };
    const message = new AIMessage({
      content: '',
      response_metadata: {
        output: [
          { type: 'computer_call', id: 'item', call_id: 'call', action },
        ],
      },
    });
    expect(createToolHistoryPreparation().get(message)?.contributions).toEqual([
      {
        kind: 'call',
        actor: 'model',
        name: 'computer',
        callId: 'call',
        itemId: 'item',
        outputIndex: 0,
        arguments: action,
      },
    ]);
  });
  test('keeps source precedence and does not mutate provider evidence', () => {
    const output = [
      { type: 'message', content: [{ type: 'output_text', text: 'answer' }] },
    ];
    const message = new AIMessage({
      content: 'answer',
      response_metadata: { output },
      additional_kwargs: {
        tool_outputs: [{ type: 'mcp_call', output: 'stale' }],
      },
    });
    const source = getResponsesHistorySource(message)!;
    expect(source.coverage).toBe('complete-output');
    expect(source.items).toBe(output);
    expect([...projectResponsesHistory(source, () => true)]).toEqual([
      {
        outputIndex: 0,
        contentIndex: 0,
        kind: 'text',
        actor: 'model',
        text: 'answer',
      },
    ]);
    expect(message.content).toBe('answer');
  });

  test('retains actor, call identity, and event order', () => {
    const source = getResponsesHistorySource(
      new AIMessage({
        content: 'beforeafter',
        response_metadata: {
          output: [
            {
              type: 'message',
              content: [{ type: 'output_text', text: 'before' }],
            },
            {
              type: 'custom_tool_call',
              id: 'item',
              call_id: 'call',
              name: 'execute',
              input: 'code',
            },
            { type: 'mcp_call', output: 'result' },
            {
              type: 'message',
              content: [{ type: 'refusal', refusal: 'after' }],
            },
          ],
        },
      })
    )!;
    const contributions = [...projectResponsesHistory(source, () => true)];
    expect(
      contributions.map((entry) => [entry.kind, entry.actor, entry.outputIndex])
    ).toEqual([
      ['text', 'model', 0],
      ['call', 'model', 1],
      ['provider-item', 'tool', 2],
      ['text', 'model', 3],
    ]);
    expect(contributions[1]).toMatchObject({
      itemId: 'item',
      callId: 'call',
      name: 'execute',
      arguments: 'code',
    });
  });

  test('preserves completed JPEG media and excludes incomplete image fragments', () => {
    const source = getResponsesHistorySource(
      new AIMessage({
        content: [],
        additional_kwargs: {
          tool_outputs: [
            {
              type: 'image_generation_call',
              status: 'in_progress',
              result: 'partial',
            },
            {
              type: 'image_generation_call',
              status: 'completed',
              result: '/9j/AA==',
            },
          ],
        },
      })
    )!;
    expect(source.coverage).toBe('tool-sidecar');
    const contributions = [...projectResponsesHistory(source, () => true)];
    expect(
      contributions.filter((entry) => entry.kind === 'image')
    ).toHaveLength(1);
    expect(contributions[1]).toMatchObject({
      kind: 'image',
      actor: 'tool',
      image: { mimeType: 'image/jpeg', data: '/9j/AA==' },
    });
    expect(JSON.stringify(contributions)).not.toContain('partial');
  });

  test('stops nested traversal at the shared work limit', () => {
    const source = getResponsesHistorySource(
      new AIMessage({
        content: [],
        response_metadata: {
          output: [
            {
              type: 'message',
              content: [
                { type: 'output_text', text: 'first' },
                { type: 'output_text', text: 'second' },
              ],
            },
          ],
        },
      })
    )!;
    let remaining = 2;
    expect([...projectResponsesHistory(source, () => remaining-- > 0)]).toEqual(
      [
        {
          outputIndex: 0,
          contentIndex: 0,
          kind: 'text',
          actor: 'model',
          text: 'first',
        },
      ]
    );
  });

  test('reuses one projection per preparation while plain text has no projection', () => {
    const preparation = createToolHistoryPreparation();
    expect(preparation.get(new HumanMessage('plain text'))).toBeUndefined();
    const message = new AIMessage({
      content: [],
      response_metadata: {
        output: [{ type: 'mcp_call', output: 'evidence' }],
      },
    });
    const first = preparation.get(message);
    expect(preparation.get(message)).toBe(first);
    expect(createToolHistoryPreparation().get(message)).not.toBe(first);
  });

  test('restores text and sidecar ordering from recorded streaming positions', () => {
    const message = new AIMessage({
      content: [
        { type: 'text', text: 'before' },
        { type: 'text', text: 'after' },
      ],
      additional_kwargs: {
        tool_outputs: [{ type: 'mcp_call', id: 'tool', output: 'result' }],
        __openai_responses_replay_positions__: [
          { kind: 'text', itemId: 'm1', outputIndex: 0, contentIndex: 0 },
          { kind: 'output', itemId: 'tool', outputIndex: 1 },
          { kind: 'text', itemId: 'm2', outputIndex: 2, contentIndex: 0 },
        ],
      },
    });
    const projection = createToolHistoryPreparation().get(message)!;
    expect(projection.contributions.map((entry) => entry.kind)).toEqual([
      'text',
      'provider-item',
      'text',
    ]);
    expect(projection.contributions.map((entry) => entry.outputIndex)).toEqual([
      0, 1, 2,
    ]);
  });

  test('does not suppress calls whose IDs match but arguments differ', () => {
    const mirrors: ToolHistoryCallMirrors = new Map();
    recordToolHistoryCallMirror(mirrors, {
      type: 'tool_use',
      id: 'call',
      name: 'lookup',
      input: { query: 'a' },
    });
    expect(
      isToolHistoryCallMirror(mirrors, 'call', 'lookup', { query: 'a' })
    ).toBe(true);
    expect(
      isToolHistoryCallMirror(mirrors, 'call', 'lookup', { query: 'b' })
    ).toBe(false);
    recordToolHistoryCallMirror(mirrors, {
      type: 'tool_use',
      id: 'call',
      name: 'lookup',
      input: { query: 'a' },
    });
    expect(
      isToolHistoryCallMirror(mirrors, 'call', 'lookup', { query: 'a' })
    ).toBe(false);
  });
});
