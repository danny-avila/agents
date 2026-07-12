import OpenAI from 'openai';

import {
  addProgrammaticCallerLinkage,
  completeResponsesWithNativeContinuation,
  getNativeResponsesTools,
} from './index';
import { Constants } from '@/common';

type ResponsesRequest =
  | OpenAI.Responses.ResponseCreateParamsStreaming
  | OpenAI.Responses.ResponseCreateParamsNonStreaming;
type ResponsesResult =
  | OpenAI.Responses.Response
  | AsyncIterable<OpenAI.Responses.ResponseStreamEvent>;

function createResponse({
  id,
  output,
  inputTokens,
  outputTokens,
}: {
  id: string;
  output: unknown[];
  inputTokens: number;
  outputTokens: number;
}): OpenAI.Responses.Response {
  return {
    id,
    object: 'response',
    created_at: 1,
    status: 'completed',
    error: null,
    incomplete_details: null,
    instructions: null,
    max_output_tokens: null,
    model: 'gpt-5.6',
    output,
    parallel_tool_calls: true,
    previous_response_id: null,
    reasoning: null,
    store: false,
    temperature: null,
    text: { format: { type: 'text' } },
    tool_choice: 'auto',
    tools: [],
    top_p: null,
    truncation: 'disabled',
    usage: {
      input_tokens: inputTokens,
      input_tokens_details: {
        cached_tokens: 1,
      },
      output_tokens: outputTokens,
      output_tokens_details: {
        reasoning_tokens: 2,
      },
      total_tokens: inputTokens + outputTokens,
    },
    user: null,
    metadata: {},
    output_text: '',
  } as unknown as OpenAI.Responses.Response;
}

function createTerminalEvent(
  response: OpenAI.Responses.Response
): OpenAI.Responses.ResponseCompletedEvent {
  return {
    type: 'response.completed',
    sequence_number: 1,
    response,
  };
}

describe('native program continuation', () => {
  const programOutput = [
    {
      type: 'program',
      id: 'prog_1',
      call_id: 'program_call_1',
      code: 'await tools.lookup({})',
      fingerprint: 'fingerprint',
    },
    {
      type: 'program_output',
      id: 'program_output_1',
      call_id: 'program_call_1',
      result: '{"ok":true}',
      status: 'completed',
    },
  ];
  const messageOutput = [
    {
      type: 'message',
      id: 'msg_1',
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text: 'Done.', annotations: [] }],
    },
  ];

  it('maps LibreChat caller eligibility to native OpenAI tools', () => {
    const tools = [
      {
        name: 'programmatic_only',
        metadata: { allowed_callers: ['code_execution'] },
      },
      {
        name: 'hybrid',
        metadata: { allowed_callers: ['direct', 'code_execution'] },
      },
      {
        name: 'direct_only',
        metadata: { allowed_callers: ['direct'] },
      },
    ];
    const reduced = tools.map(({ name }) => ({
      type: 'function' as const,
      name,
      description: name,
      parameters: { type: 'object' as const, properties: {} },
      strict: false,
    }));

    expect(
      getNativeResponsesTools(
        tools as Parameters<typeof getNativeResponsesTools>[0],
        reduced,
        true
      )
    ).toEqual([
      { ...reduced[0], allowed_callers: ['programmatic'] },
      { ...reduced[1], allowed_callers: ['direct', 'programmatic'] },
      reduced[2],
      { type: 'programmatic_tool_calling' },
    ]);
  });

  it('omits LibreChat programmatic wrappers from native tool declarations', () => {
    const tools = [
      { name: Constants.PROGRAMMATIC_TOOL_CALLING },
      { name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING },
      { name: 'direct_tool' },
    ];
    const reduced = tools.map(({ name }) => ({
      type: 'function' as const,
      name,
      description: name,
      parameters: { type: 'object' as const, properties: {} },
      strict: false,
    }));

    expect(
      getNativeResponsesTools(
        tools as Parameters<typeof getNativeResponsesTools>[0],
        reduced,
        true
      )
    ).toEqual([
      reduced[2],
      { type: 'programmatic_tool_calling' },
    ]);
  });

  it('restores caller linkage on reconstructed function outputs', () => {
    const caller = { type: 'program' as const, caller_id: 'program_call_1' };
    expect(
      addProgrammaticCallerLinkage([
        {
          type: 'function_call',
          call_id: 'tool_call_1',
          name: 'lookup',
          arguments: '{}',
          caller,
        },
        {
          type: 'function_call_output',
          call_id: 'tool_call_1',
          output: '{"ok":true}',
        },
      ])
    ).toEqual([
      {
        type: 'function_call',
        call_id: 'tool_call_1',
        name: 'lookup',
        arguments: '{}',
        caller,
      },
      {
        type: 'function_call_output',
        call_id: 'tool_call_1',
        output: '{"ok":true}',
        caller,
      },
    ]);
  });

  it('continues stateless responses with replayed program output and merged usage', async () => {
    const first = createResponse({
      id: 'resp_1',
      output: programOutput,
      inputTokens: 10,
      outputTokens: 5,
    });
    const second = createResponse({
      id: 'resp_2',
      output: messageOutput,
      inputTokens: 20,
      outputTokens: 7,
    });
    const requests: ResponsesRequest[] = [];
    const responses = [first, second];
    const result = await completeResponsesWithNativeContinuation(
      {
        model: 'gpt-5.6',
        input: [{ role: 'user', content: 'Look it up.' }],
        store: false,
        previous_response_id: 'resp_prior',
        stream: false,
      },
      undefined,
      true,
      async (request): Promise<ResponsesResult> => {
        requests.push(request);
        const response = responses.shift();
        if (response == null) {
          throw new Error('Unexpected continuation request.');
        }
        return response;
      }
    );

    expect(Symbol.asyncIterator in result).toBe(false);
    const response = result as OpenAI.Responses.Response;
    expect(requests).toHaveLength(2);
    expect(requests[1].previous_response_id).toBe('resp_prior');
    expect(requests[1].input).toEqual([
      { role: 'user', content: 'Look it up.' },
      ...programOutput,
    ]);
    expect(response.output).toEqual([...programOutput, ...messageOutput]);
    expect(response.usage).toMatchObject({
      input_tokens: 30,
      output_tokens: 12,
      total_tokens: 42,
      input_tokens_details: { cached_tokens: 2 },
      output_tokens_details: { reasoning_tokens: 4 },
    });
  });

  it('replays string input for stateless continuations', async () => {
    const first = createResponse({
      id: 'resp_1',
      output: programOutput,
      inputTokens: 10,
      outputTokens: 5,
    });
    const second = createResponse({
      id: 'resp_2',
      output: messageOutput,
      inputTokens: 20,
      outputTokens: 7,
    });
    const requests: ResponsesRequest[] = [];
    const responses = [first, second];

    await completeResponsesWithNativeContinuation(
      {
        model: 'gpt-5.6',
        input: 'Look it up.',
        store: false,
        stream: false,
      },
      undefined,
      true,
      async (request): Promise<ResponsesResult> => {
        requests.push(request);
        const response = responses.shift();
        if (response == null) {
          throw new Error('Unexpected continuation request.');
        }
        return response;
      }
    );

    expect(requests[1].input).toEqual([
      { role: 'user', content: 'Look it up.' },
      ...programOutput,
    ]);
  });

  it('continues stored responses by response id', async () => {
    const first = createResponse({
      id: 'resp_1',
      output: programOutput,
      inputTokens: 10,
      outputTokens: 5,
    });
    const second = createResponse({
      id: 'resp_2',
      output: messageOutput,
      inputTokens: 20,
      outputTokens: 7,
    });
    const requests: ResponsesRequest[] = [];
    const responses = [first, second];
    await completeResponsesWithNativeContinuation(
      {
        model: 'gpt-5.6',
        input: [{ role: 'user', content: 'Look it up.' }],
        stream: false,
      },
      undefined,
      true,
      async (request): Promise<ResponsesResult> => {
        requests.push(request);
        const response = responses.shift();
        if (response == null) {
          throw new Error('Unexpected continuation request.');
        }
        return response;
      }
    );

    expect(requests[1]).toMatchObject({
      input: [],
      previous_response_id: 'resp_1',
    });
  });

  it('reports incomplete native program responses with the provider reason', async () => {
    const incomplete = {
      ...createResponse({
        id: 'resp_incomplete',
        output: programOutput,
        inputTokens: 10,
        outputTokens: 5,
      }),
      status: 'incomplete' as const,
      incomplete_details: { reason: 'max_output_tokens' as const },
    };

    await expect(
      completeResponsesWithNativeContinuation(
        {
          model: 'gpt-5.6',
          input: [{ role: 'user', content: 'Look it up.' }],
          stream: false,
        },
        undefined,
        true,
        async (): Promise<ResponsesResult> => incomplete
      )
    ).rejects.toThrow(
      'Native Programmatic Tool Calling response was incomplete: max_output_tokens.'
    );
  });

  it('returns incomplete responses that include a terminal message', async () => {
    const incomplete = {
      ...createResponse({
        id: 'resp_incomplete',
        output: messageOutput,
        inputTokens: 10,
        outputTokens: 5,
      }),
      status: 'incomplete' as const,
      incomplete_details: { reason: 'max_output_tokens' as const },
    };

    const result = await completeResponsesWithNativeContinuation(
      {
        model: 'gpt-5.6',
        input: [{ role: 'user', content: 'Look it up.' }],
        stream: false,
      },
      undefined,
      true,
      async (): Promise<ResponsesResult> => incomplete
    );

    expect(result).toMatchObject({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: messageOutput,
    });
  });

  it('emits one terminal streaming event with merged output and usage', async () => {
    const first = createResponse({
      id: 'resp_1',
      output: programOutput,
      inputTokens: 10,
      outputTokens: 5,
    });
    const second = createResponse({
      id: 'resp_2',
      output: messageOutput,
      inputTokens: 20,
      outputTokens: 7,
    });
    const responses = [first, second];
    const result = await completeResponsesWithNativeContinuation(
      {
        model: 'gpt-5.6',
        input: [{ role: 'user', content: 'Look it up.' }],
        store: false,
        stream: true,
      },
      undefined,
      true,
      async (): Promise<ResponsesResult> => {
        const response = responses.shift();
        if (response == null) {
          throw new Error('Unexpected continuation request.');
        }
        return (async function* () {
          yield createTerminalEvent(response);
        })();
      }
    );

    expect(Symbol.asyncIterator in result).toBe(true);
    const events: OpenAI.Responses.ResponseStreamEvent[] = [];
    for await (const event of result as AsyncIterable<OpenAI.Responses.ResponseStreamEvent>) {
      events.push(event);
    }
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('response.completed');
    if (events[0].type !== 'response.completed') {
      throw new Error('Expected a completed response.');
    }
    expect(events[0].response.output).toEqual([
      ...programOutput,
      ...messageOutput,
    ]);
    expect(events[0].response.usage?.total_tokens).toBe(42);
  });
});
