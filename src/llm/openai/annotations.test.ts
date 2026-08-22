import { ensureResponsesOutputAnnotations } from './index';

type ResponsesAnnotationsBoundaryEvent = Parameters<
  typeof ensureResponsesOutputAnnotations
>[0];

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
    const event: ResponsesAnnotationsBoundaryEvent = {
      type: 'response.completed',
      response: {
        output: [
          {
            type: 'message',
            content: [{ type: 'output_text' }],
          },
        ],
      },
    };

    ensureResponsesOutputAnnotations(event);

    const part = event.response?.output?.[0]?.content?.[0];
    expect(part?.annotations).toEqual([]);
  });

  it('leaves existing annotations untouched', () => {
    const event: ResponsesAnnotationsBoundaryEvent = {
      type: 'response.completed',
      response: {
        output: [
          {
            type: 'message',
            content: [
              {
                type: 'output_text',
                annotations: [{ type: 'url_citation', url: 'https://example.com' }],
              },
            ],
          },
        ],
      },
    };

    ensureResponsesOutputAnnotations(event);

    const part = event.response?.output?.[0]?.content?.[0];
    expect(part?.annotations).toEqual([
      { type: 'url_citation', url: 'https://example.com' },
    ]);
  });

  it('does not throw on non-terminal or non-message events', () => {
    const textDelta: ResponsesAnnotationsBoundaryEvent = {
      type: 'response.output_text.delta',
    };
    const toolCall: ResponsesAnnotationsBoundaryEvent = {
      type: 'response.completed',
      response: {
        output: [{ type: 'function_call' }],
      },
    };

    expect(() => ensureResponsesOutputAnnotations(textDelta)).not.toThrow();
    expect(() => ensureResponsesOutputAnnotations(toolCall)).not.toThrow();
  });
});
