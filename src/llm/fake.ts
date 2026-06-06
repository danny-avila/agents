import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage } from '@langchain/core/messages';

type SplitStrategy = {
  type: 'regex' | 'fixed';
  value: RegExp | number;
};

export class FakeChatModel extends FakeListChatModel {
  private splitStrategy: SplitStrategy;
  private toolCalls: ToolCall[] = [];
  private addedToolCalls: boolean = false;

  constructor({
    responses,
    sleep,
    emitCustomEvent,
    splitStrategy = { type: 'regex', value: /(?<=\s+)|(?=\s+)/ },
    toolCalls = [],
  }: {
    responses: string[];
    sleep?: number;
    emitCustomEvent?: boolean;
    splitStrategy?: SplitStrategy;
    toolCalls?: ToolCall[];
  }) {
    super({ responses, sleep, emitCustomEvent });
    this.splitStrategy = splitStrategy;
    this.toolCalls = toolCalls;
  }

  private splitText(text: string): string[] {
    if (this.splitStrategy.type === 'regex') {
      return text.split(this.splitStrategy.value as RegExp);
    } else {
      const chunkSize = this.splitStrategy.value as number;
      const chunks: string[] = [];
      for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
      }
      return chunks;
    }
  }
  _createResponseChunk(
    text: string,
    tool_call_chunks?: ToolCallChunk[]
  ): ChatGenerationChunk {
    return new ChatGenerationChunk({
      text,
      generationInfo: {},
      message: new AIMessageChunk({
        content: text,
        tool_call_chunks,
        additional_kwargs: tool_call_chunks
          ? {
            tool_calls: tool_call_chunks.map((toolCall) => ({
              index: toolCall.index ?? 0,
              id: toolCall.id ?? '',
              type: 'function',
              function: {
                name: toolCall.name ?? '',
                arguments: toolCall.args ?? '',
              },
            })),
          }
          : undefined,
      }),
    });
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const response = this._currentResponse();
    this._incrementResponse();

    if (this.emitCustomEvent) {
      await runManager?.handleCustomEvent('some_test_event', {
        someval: true,
      });
    }

    const chunks = this.splitText(response);
    for await (const chunk of chunks) {
      await this._sleepIfRequested();

      if (options.thrownErrorString != null && options.thrownErrorString) {
        throw new Error(options.thrownErrorString);
      }

      const responseChunk = super._createResponseChunk(chunk);
      yield responseChunk;
      void runManager?.handleLLMNewToken(chunk);
    }

    await this._sleepIfRequested();
    if (this.toolCalls.length > 0 && !this.addedToolCalls) {
      this.addedToolCalls = true;
      const toolCallChunks = this.toolCalls.map((toolCall) => {
        return {
          name: toolCall.name,
          args: JSON.stringify(toolCall.args),
          id: toolCall.id,
          type: 'tool_call_chunk',
        } as ToolCallChunk;
      });
      const responseChunk = this._createResponseChunk('', toolCallChunks);
      yield responseChunk;
      void runManager?.handleLLMNewToken('');
    }
  }
}

export function createFakeStreamingLLM({
  responses,
  sleep,
  splitStrategy,
  toolCalls,
}: {
  responses: string[];
  sleep?: number;
  splitStrategy?: SplitStrategy;
  toolCalls?: ToolCall[];
}): FakeChatModel {
  return new FakeChatModel({
    sleep,
    responses,
    emitCustomEvent: true,
    splitStrategy,
    toolCalls,
  });
}
