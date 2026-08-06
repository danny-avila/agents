import { ChatMistralAI } from '@langchain/mistralai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { MistralAIClientOptions } from '@/types';
import { smoothGenerationChunks } from '@/llm/stream/chunkAdapters';
import { resolveStreamDelay } from '@/llm/stream/smoother';

export class CustomChatMistralAI extends ChatMistralAI {
  _lc_stream_delay: number;

  static lc_name(): 'LibreChatMistralAI' {
    return 'LibreChatMistralAI';
  }

  constructor(fields?: MistralAIClientOptions) {
    super(fields);
    this._lc_stream_delay = resolveStreamDelay(fields?._lc_stream_delay);
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* smoothGenerationChunks({
      chunks: super._streamResponseChunks(messages, options, undefined),
      delayMs: this._lc_stream_delay,
      signal: options.signal,
      runManager,
    });
  }
}
