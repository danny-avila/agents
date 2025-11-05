import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import {
  HumanMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { ContentTypes, GraphEvents, Providers, TitleMethod } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { capitalizeFirstLetter } from './spec.utils';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

// Auto-skip if OpenRouter env is missing
const hasOpenRouter = (process.env.OPENROUTER_API_KEY ?? '').trim() !== '';
const describeIf = hasOpenRouter ? describe : describe.skip;

const provider = Providers.OPENROUTER;
describeIf(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(60000);
  let run: Run<t.IState>;
  let collectedUsage: UsageMetadata[];
  let conversationHistory: BaseMessage[];
  let contentParts: t.MessageContentComplex[];

  const configV2 = {
    configurable: { thread_id: 'or-convo-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  beforeEach(async () => {
    conversationHistory = [];
    collectedUsage = [];
    const { contentParts: cp } = createContentAggregator();
    contentParts = cp as t.MessageContentComplex[];
  });

  const onMessageDeltaSpy = jest.fn();
  const onRunStepSpy = jest.fn();

  afterAll(() => {
    onMessageDeltaSpy.mockReset();
    onRunStepSpy.mockReset();
  });

  const setupCustomHandlers = (): Record<
    string | GraphEvents,
    t.EventHandler
  > => ({
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
  });

  test(`${capitalizeFirstLetter(provider)}: simple stream + title`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'or-run-1',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions: 'You are a friendly AI assistant.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    const userMessage = 'hi';
    conversationHistory.push(new HumanMessage(userMessage));

    const finalContentParts = await run.processStream(
      { messages: conversationHistory },
      configV2
    );
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(
      (collectedUsage[0]?.input_tokens ?? 0) +
        (collectedUsage[0]?.output_tokens ?? 0)
    ).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    conversationHistory.push(...(finalMessages ?? []));

    const titleRes = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.COMPLETION,
      contentParts,
    });
    expect(titleRes.title).toBeDefined();
  });
});
