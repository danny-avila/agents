import type { OpenAIClient } from '@langchain/openai';

import { ensureResponsesOutputAnnotations } from './index';

/**
 * Regression coverage for a crash against OpenAI-compatible Responses API
 * gateways (e.g. the Codex backend used by the SDT deployment) whose terminal
 * `response.completed` events omit the `annotations` field on `output_text`
 * content parts. LangChain's responses converter maps over the field
 * unconditionally, so a missing field threw "Cannot read properties of
 * undefined (reading 'map')" on every streamed completion from such a server.
 */
describe('missing annotations on terminal responses events', () => {
  it('defaults annotations to [] on output_text parts without the field', () => {
    const event = {
      type: 'response.completed',
      response: {
        id: 'resp_test',
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'Hello' }],
          },
        ],
      },
    } as unknown as OpenAIClient.Responses.ResponseStreamEvent;

    ensureResponsesOutputAnnotations(event);

    const part = (event as unknown as {
      response: { output: Array<{ content: Array<{ annotations?: unknown }> }> };
    }).response.output[0].content[0];
    expect(part.annotations).toEqual([]);
  });

  it('leaves existing annotations untouched', () => {
    const event = {
      type: 'response.completed',
      response: {
        id: 'resp_test',
        output: [
          {
            type: 'message',
            role: 'assistant',
            content: [
              {
                type: 'output_text',
                text: 'Hello',
                annotations: [{ type: 'url_citation', url: 'https://example.com' }],
              },
            ],
          },
        ],
      },
    } as unknown as OpenAIClient.Responses.ResponseStreamEvent;

    ensureResponsesOutputAnnotations(event);

    const part = (event as unknown as {
      response: { output: Array<{ content: Array<{ annotations: unknown }> }> };
    }).response.output[0].content[0];
    expect(part.annotations).toEqual([
      { type: 'url_citation', url: 'https://example.com' },
    ]);
  });

  it('does not throw on non-terminal or non-message events', () => {
    const textDelta = {
      type: 'response.output_text.delta',
      delta: 'Hi',
    } as unknown as OpenAIClient.Responses.ResponseStreamEvent;
    const toolCall = {
      type: 'response.completed',
      response: {
        id: 'resp_test',
        output: [{ type: 'function_call', call_id: 'call_1', name: 'f', arguments: '{}' }],
      },
    } as unknown as OpenAIClient.Responses.ResponseStreamEvent;

    expect(() => ensureResponsesOutputAnnotations(textDelta)).not.toThrow();
    expect(() => ensureResponsesOutputAnnotations(toolCall)).not.toThrow();
  });
});
