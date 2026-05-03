/* eslint-disable no-console */
import { HumanMessage } from '@langchain/core/messages';
import type { OpenAI as OpenAIClient } from 'openai';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatDeepSeek } from '@/llm/openai';
import { Providers } from '@/common';
import { Calculator } from '@/tools/Calculator';
import { Run } from '@/run';

type MessageParam = OpenAIClient.Chat.Completions.ChatCompletionMessageParam & {
  reasoning_content?: string;
};

type AssistantParam = Extract<MessageParam, { role: 'assistant' }> & {
  reasoning_content?: string;
};

type DeepSeekDelta =
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice.Delta & {
    reasoning_content?: string;
  };

type DeepSeekChunk = Omit<
  OpenAIClient.Chat.Completions.ChatCompletionChunk,
  'choices'
> & {
  choices: Array<
    Omit<OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice, 'delta'> & {
      delta: DeepSeekDelta;
    }
  >;
};

type CompletionRequest =
  | OpenAIClient.Chat.Completions.ChatCompletionCreateParamsStreaming
  | OpenAIClient.Chat.Completions.ChatCompletionCreateParamsNonStreaming;

type CapturedRequest = {
  call: number;
  messages: MessageParam[];
};

type StreamConfig = Partial<RunnableConfig> & {
  version: 'v2';
  streamMode: 'values';
  configurable: { thread_id: string };
};

const capturedRequests: CapturedRequest[] = [];

const createChunk = (
  delta: DeepSeekDelta,
  finishReason: DeepSeekChunk['choices'][number]['finish_reason'] = null
): DeepSeekChunk => ({
  id: `chatcmpl-local-${Date.now()}`,
  created: Math.floor(Date.now() / 1000),
  model: 'deepseek-v4-pro',
  object: 'chat.completion.chunk',
  choices: [
    {
      index: 0,
      delta,
      finish_reason: finishReason,
      logprobs: null,
    },
  ],
});

async function* streamForCall(call: number): AsyncGenerator<DeepSeekChunk> {
  if (call === 1) {
    yield createChunk({
      role: 'assistant',
      content: '',
      reasoning_content: 'Turn 1.1: I need the calculator for 127 * 453.',
    });
    yield createChunk({ content: 'Let me calculate that.' });
    yield createChunk(
      {
        content: '',
        tool_calls: [
          {
            index: 0,
            id: 'call_calc_1',
            type: 'function',
            function: {
              name: 'calculator',
              arguments: '{"input":"127 * 453"}',
            },
          },
        ],
      },
      'tool_calls'
    );
    return;
  }

  if (call === 2) {
    yield createChunk({
      role: 'assistant',
      content: '',
      reasoning_content:
        'Turn 1.2: The calculator returned 57531, so I can answer.',
    });
    yield createChunk({ content: '127 * 453 = 57531.' }, 'stop');
    return;
  }

  if (call === 3) {
    yield createChunk({
      role: 'assistant',
      content: '',
      reasoning_content:
        'Turn 2.1: I should explain the previous calculation briefly.',
    });
    yield createChunk(
      {
        content: 'I multiplied 127 by 453 and reported the calculator result.',
      },
      'stop'
    );
    return;
  }

  yield createChunk({
    role: 'assistant',
    content: '',
    reasoning_content: 'Turn 3.1: I should confirm the prior explanation.',
  });
  yield createChunk({ content: 'That explanation is still correct.' }, 'stop');
}

const originalCompletionWithRetry = ChatDeepSeek.prototype.completionWithRetry;

function installDeepSeekStub(): void {
  let call = 0;

  function stubbedCompletionWithRetry(
    this: ChatDeepSeek,
    request: OpenAIClient.Chat.Completions.ChatCompletionCreateParamsStreaming,
    options?: OpenAIClient.RequestOptions
  ): Promise<AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>>;
  function stubbedCompletionWithRetry(
    this: ChatDeepSeek,
    request: OpenAIClient.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
    options?: OpenAIClient.RequestOptions
  ): Promise<OpenAIClient.Chat.Completions.ChatCompletion>;
  async function stubbedCompletionWithRetry(
    this: ChatDeepSeek,
    request: CompletionRequest,
    _options?: OpenAIClient.RequestOptions
  ): Promise<
    | AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    | OpenAIClient.Chat.Completions.ChatCompletion
  > {
    if (request.stream !== true) {
      throw new Error('This diagnostic only stubs streaming DeepSeek calls.');
    }
    call += 1;
    capturedRequests.push({
      call,
      messages: request.messages as MessageParam[],
    });
    return streamForCall(call);
  }

  ChatDeepSeek.prototype.completionWithRetry = stubbedCompletionWithRetry;
}

function restoreDeepSeekStub(): void {
  ChatDeepSeek.prototype.completionWithRetry = originalCompletionWithRetry;
}

function assistantMessages(requestCall: number): AssistantParam[] {
  const request = capturedRequests.find(({ call }) => call === requestCall);
  return (request?.messages ?? []).filter(
    (message): message is AssistantParam => message.role === 'assistant'
  );
}

function hasReasoning(message: AssistantParam | undefined): boolean {
  return typeof message?.reasoning_content === 'string';
}

function hasToolCalls(message: AssistantParam | undefined): boolean {
  return (
    Array.isArray(message?.tool_calls) && (message.tool_calls?.length ?? 0) > 0
  );
}

function hasStoredReasoning(message: BaseMessage | undefined): boolean {
  return typeof message?.additional_kwargs.reasoning_content === 'string';
}

function summarizeRequest(call: number): void {
  const assistants = assistantMessages(call);
  console.log(
    `request ${call}: ${assistants.length} assistant message(s) sent`
  );
  assistants.forEach((message, index) => {
    console.log(
      `  assistant[${index}] tool_calls=${hasToolCalls(message)} reasoning_content=${hasReasoning(message)} content=${JSON.stringify(message.content)}`
    );
  });
}

async function createRun(runId: string): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      instructions: 'Use the calculator when arithmetic is requested.',
      llmConfig: {
        provider: Providers.DEEPSEEK,
        model: 'deepseek-v4-pro',
        apiKey: 'local-test-key',
        streaming: true,
        streamUsage: true,
      },
      tools: [new Calculator()],
    },
    returnContent: true,
    skipCleanup: true,
  });
}

async function runTurn(
  run: Run<t.IState>,
  messages: BaseMessage[],
  threadId: string
): Promise<BaseMessage[]> {
  const config: StreamConfig = {
    version: 'v2',
    streamMode: 'values',
    configurable: { thread_id: threadId },
  };

  await run.processStream({ messages }, config);
  return run.getRunMessages() ?? [];
}

async function main(): Promise<void> {
  installDeepSeekStub();
  try {
    const conversationHistory: BaseMessage[] = [
      new HumanMessage('What is 127 * 453?'),
    ];

    const firstRun = await createRun('deepseek-v4-local-turn-1');
    const firstRunMessages = await runTurn(
      firstRun,
      conversationHistory,
      'deepseek-v4-local'
    );
    conversationHistory.push(...firstRunMessages);

    const finalTurnOneMessage = firstRunMessages.at(-1);
    console.log(
      `turn 1 final stored reasoning=${hasStoredReasoning(finalTurnOneMessage)}`
    );

    conversationHistory.push(
      new HumanMessage('Now explain what you did in one sentence.')
    );

    const secondRun = await createRun('deepseek-v4-local-turn-2');
    const secondRunMessages = await runTurn(
      secondRun,
      conversationHistory,
      'deepseek-v4-local'
    );
    conversationHistory.push(...secondRunMessages);

    const finalTurnTwoMessage = secondRunMessages.at(-1);
    console.log(
      `turn 2 final stored reasoning=${hasStoredReasoning(finalTurnTwoMessage)}`
    );

    conversationHistory.push(
      new HumanMessage('Was that explanation correct? Answer briefly.')
    );

    const thirdRun = await createRun('deepseek-v4-local-turn-3');
    await runTurn(thirdRun, conversationHistory, 'deepseek-v4-local');

    summarizeRequest(2);
    summarizeRequest(3);
    summarizeRequest(4);

    const sameTurnToolAssistant = assistantMessages(2).find(hasToolCalls);
    const nextTurnFinalAssistant = assistantMessages(3).find(
      (message) =>
        !hasToolCalls(message) && message.content === '127 * 453 = 57531.'
    );
    const thirdTurnPreviousAssistant = assistantMessages(4).find(
      (message) =>
        !hasToolCalls(message) &&
        message.content ===
          'I multiplied 127 by 453 and reported the calculator result.'
    );

    const checks = [
      {
        name: 'same-turn assistant tool call includes reasoning_content',
        pass: hasReasoning(sameTurnToolAssistant),
      },
      {
        name: 'turn 1 final assistant stored reasoning_content',
        pass: hasStoredReasoning(finalTurnOneMessage),
      },
      {
        name: 'next-turn final assistant includes reasoning_content',
        pass: hasReasoning(nextTurnFinalAssistant),
      },
      {
        name: 'turn 2 final assistant stored reasoning_content',
        pass: hasStoredReasoning(finalTurnTwoMessage),
      },
      {
        name: 'third-turn previous assistant includes reasoning_content',
        pass: hasReasoning(thirdTurnPreviousAssistant),
      },
    ];

    checks.forEach(({ name, pass }) => {
      console.log(`${pass ? 'PASS' : 'FAIL'} ${name}`);
    });

    if (checks.some(({ pass }) => !pass)) {
      process.exitCode = 1;
    }
  } finally {
    restoreDeepSeekStub();
  }
}

main().catch((error: Error) => {
  restoreDeepSeekStub();
  console.error(error);
  process.exitCode = 1;
});
