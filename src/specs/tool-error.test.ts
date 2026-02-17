import { config } from 'dotenv';
config();
import { tool } from '@langchain/core/tools';
import { ToolCall } from '@langchain/core/messages/tool';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { StandardGraph } from '@/graphs';
import { Run } from '@/run';

const errorTool = tool(
  async () => {
    throw new Error('this is a test error I threw on purpose');
  },
  {
    name: 'errorTool',
    description: 'A tool that always throws an error',
    schema: {
      type: 'object',
      properties: {
        input: { type: 'string' },
      },
      required: [],
    },
  }
);

describe('Tool Error Handling Tests', () => {
  jest.setTimeout(30000);
  let run: Run<t.IState>;
  let contentParts: t.MessageContentComplex[];
  let conversationHistory: BaseMessage[];
  let aggregateContent: t.ContentAggregator;
  let handleToolCallErrorSpy: jest.SpyInstance;

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    run_id?: string;
    streamMode: string;
  } = {
    configurable: {
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  beforeEach(async () => {
    conversationHistory = [];
    const { contentParts: parts, aggregateContent: ac } =
      createContentAggregator();
    aggregateContent = ac;
    contentParts = parts as t.MessageContentComplex[];
    // Spy on the static method instead of the instance method
    handleToolCallErrorSpy = jest.spyOn(
      StandardGraph,
      'handleToolCallErrorStatic'
    );
  });

  afterEach(() => {
    handleToolCallErrorSpy.mockRestore();
  });

  const onMessageDeltaSpy = jest.fn();
  const onRunStepSpy = jest.fn();
  const onRunStepCompletedSpy = jest.fn();

  afterAll(() => {
    onMessageDeltaSpy.mockReset();
    onRunStepSpy.mockReset();
    onRunStepCompletedSpy.mockReset();
  });

  const setupCustomHandlers = (): Record<
    string | GraphEvents,
    t.EventHandler
  > => ({
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        if ((data.result as t.MessageContentComplex)['type'] === 'tool_call') {
          run.Graph?.overrideTestModel(
            ['Looks like there was an error calling the tool.'],
            5
          );
        }
        onRunStepCompletedSpy(event, data);
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        const runStepData = data as t.RunStep;
        onRunStepSpy(event, runStepData, metadata, graph);
        aggregateContent({ event, data: runStepData });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onMessageDeltaSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  });

  test('should handle tool call errors correctly', async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(Providers.OPENAI);
    const customHandlers = setupCustomHandlers();

    // Create the run instance
    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [errorTool],
        instructions: 'You are a helpful AI assistant.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    const toolCalls: ToolCall[] = [
      {
        name: 'errorTool',
        args: {
          input: 'test input',
        },
        id: 'call_test123',
        type: 'tool_call',
      },
    ];

    const firstResponse = 'Let me try calling the tool';
    run.Graph?.overrideTestModel([firstResponse], 5, toolCalls);

    const userMessage = 'Use the error tool';
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);

    // Verify handleToolCallError was called
    expect(handleToolCallErrorSpy).toHaveBeenCalled();

    // Find the tool call content part
    const toolCallPart = contentParts.find(
      (part) => part.type === 'tool_call'
    ) as t.ToolCallContent | undefined;

    // Verify the error message in contentParts
    expect(toolCallPart).toBeDefined();
    expect(toolCallPart?.tool_call?.args).toEqual(
      JSON.stringify(toolCalls[0].args)
    );
    expect(toolCallPart?.tool_call?.output).toContain('Error processing tool');
    expect(toolCallPart?.tool_call?.output).toContain(
      'this is a test error I threw on purpose'
    );
  });
});
