// Inherited from @langchain/anthropic@1.5.1 tests/chat_models.test.ts (strict tool calling), adapted to LibreChat's fork.
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, test, expect, jest } from '@jest/globals';
import { CustomAnthropic as ChatAnthropic } from '@/llm/anthropic';

test('extras with cache_control are merged into tool definitions', () => {
  const searchFiles = tool(
    async (input: { query: string }) => {
      return `Results for ${input.query}`;
    },
    {
      name: 'search_files',
      description: 'Search files.',
      schema: z.object({
        query: z.string(),
      }),
      extras: { cache_control: { type: 'ephemeral' } },
    }
  );

  const model = new ChatAnthropic({
    modelName: 'claude-haiku-4-5-20251001',
    anthropicApiKey: 'testing',
  });

  const formattedTools = model.formatStructuredToolToAnthropic([searchFiles]);

  expect(formattedTools).toBeDefined();
  const searchTool = formattedTools?.find(
    (t) => 'name' in t && t.name === 'search_files'
  );
  expect(searchTool).toBeDefined();
  expect(searchTool).toHaveProperty('cache_control', { type: 'ephemeral' });
});

describe('strict tool calling', () => {
  const weatherTool = tool(
    async (input: { location: string }) => `Weather in ${input.location}`,
    {
      name: 'get_current_weather',
      description: 'Get the current weather in a location',
      schema: z.object({
        location: z.string().describe('The location to get the weather for'),
      }),
    }
  );

  const strictWeatherTool = tool(
    async (input: { location: string }) => `Weather in ${input.location}`,
    {
      name: 'get_current_weather',
      description: 'Get the current weather in a location',
      schema: z.object({ location: z.string() }),
      extras: { strict: true },
    }
  );

  const openAIShapedWeatherTool = {
    type: 'function' as const,
    function: {
      name: 'get_current_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
      },
      strict: true,
    },
  };

  const anthropicShapedWeatherTool = {
    name: 'get_current_weather',
    description: 'Get the current weather in a location',
    input_schema: {
      type: 'object' as const,
      properties: { location: { type: 'string' } },
      required: ['location'],
    },
    strict: true,
  };

  type MockFetch = jest.Mock<
    (url: string | URL | Request, options?: RequestInit) => Promise<Response>
  >;

  function makeMockFetch(): MockFetch {
    const mockFetch =
      jest.fn<
        (
          url: string | URL | Request,
          options?: RequestInit
        ) => Promise<Response>
      >();
    mockFetch.mockImplementation((_url, _options) =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            id: 'msg_test',
            type: 'message',
            role: 'assistant',
            model: 'claude-haiku-4-5-20251001',
            // `tool_use` shape (not text) is required so `withStructuredOutput`
            // can parse a tool call out of the response.
            content: [
              {
                type: 'tool_use',
                id: 'toolu_test',
                name: 'get_current_weather',
                input: { location: 'test' },
              },
            ],
            stop_reason: 'tool_use',
            stop_sequence: null,
            usage: { input_tokens: 1, output_tokens: 1 },
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          }
        )
      )
    );
    return mockFetch;
  }

  function makeMockedModel(): { model: ChatAnthropic; mockFetch: MockFetch } {
    const mockFetch = makeMockFetch();
    const model = new ChatAnthropic({
      model: 'claude-haiku-4-5-20251001',
      anthropicApiKey: 'testing',
      clientOptions: { fetch: mockFetch },
      maxRetries: 0,
    });
    return { model, mockFetch };
  }

  function getRequestTools(
    mockFetch: MockFetch
  ): Array<Record<string, unknown>> {
    expect(mockFetch).toHaveBeenCalled();
    const [, init] = mockFetch.mock.calls[0];
    if (!init || !init.body) {
      throw new Error('Body not found in request.');
    }
    const body = JSON.parse(init.body as string) as {
      tools: Array<Record<string, unknown>>;
    };
    return body.tools;
  }

  test('applies strict from .bindTools call args', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([weatherTool], { strict: true });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test('applies strict from .withConfig call options', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.withConfig({
      tools: [weatherTool],
      strict: true,
    });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test('applies strict from .bindTools(...).withConfig({ strict })', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model
      .bindTools([weatherTool])
      .withConfig({ strict: true });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test('applies strict from per-call invoke options', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([weatherTool]);
    await modelWithTools.invoke("What's the weather like?", { strict: true });
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test('per-call invoke strict overrides .bindTools strict', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([weatherTool], { strict: true });
    await modelWithTools.invoke("What's the weather like?", { strict: false });
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', false);
  });

  test('applies strict from .withStructuredOutput config', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.withStructuredOutput(
      z.object({
        location: z.string().describe('The location to get the weather for'),
      }),
      { strict: true, method: 'functionCalling' }
    );
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test('omits strict when not provided anywhere', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([weatherTool]);
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).not.toHaveProperty('strict');
  });

  test('omits strict when not passed in .withStructuredOutput', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.withStructuredOutput(
      z.object({
        location: z.string().describe('The location to get the weather for'),
      }),
      { method: 'functionCalling' }
    );
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).not.toHaveProperty('strict');
  });

  test('per-tool extras.strict applies to that tool only', async () => {
    const { model, mockFetch } = makeMockedModel();
    const looseSearchTool = tool(
      async (input: { query: string }) => `Results for ${input.query}`,
      {
        name: 'search',
        description: 'Search the web',
        schema: z.object({ query: z.string() }),
      }
    );
    const modelWithTools = model.bindTools([
      strictWeatherTool,
      looseSearchTool,
    ]);
    await modelWithTools.invoke("What's the weather like?");
    const tools = getRequestTools(mockFetch);
    const strictTool = tools.find((t) => t.name === 'get_current_weather');
    const looseTool = tools.find((t) => t.name === 'search');
    expect(strictTool).toHaveProperty('strict', true);
    expect(looseTool).not.toHaveProperty('strict');
  });

  test('per-tool function.strict applies on OpenAI-shaped tools', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([openAIShapedWeatherTool]);
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test("per-call strict overrides OpenAI-shaped tool's function.strict", async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([openAIShapedWeatherTool], {
      strict: false,
    });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', false);
  });

  test('per-tool native strict applies on Anthropic-shaped tools', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([anthropicShapedWeatherTool]);
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', true);
  });

  test("per-call strict overrides Anthropic-shaped tool's own strict", async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([anthropicShapedWeatherTool], {
      strict: false,
    });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', false);
  });

  test('per-call strict overrides per-tool extras.strict', async () => {
    const { model, mockFetch } = makeMockedModel();
    const modelWithTools = model.bindTools([strictWeatherTool], {
      strict: false,
    });
    await modelWithTools.invoke("What's the weather like?");
    expect(getRequestTools(mockFetch)[0]).toHaveProperty('strict', false);
  });

  test.each([['jsonSchema'], ['jsonMode']])(
    'withStructuredOutput throws when strict is set with method = %s',
    (method) => {
      const model = new ChatAnthropic({
        model: 'claude-haiku-4-5-20251001',
        anthropicApiKey: 'testing',
      });
      expect(() =>
        model.withStructuredOutput(z.object({ location: z.string() }), {
          strict: true,
          method: method as 'jsonSchema' | 'jsonMode',
        })
      ).toThrow(/strict.*functionCalling/);
    }
  );
});
