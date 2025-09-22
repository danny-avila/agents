import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatOllama as BaseChatOllama } from '@langchain/ollama';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type {
  ChatResponse as OllamaChatResponse,
  Message as OllamaMessage,
} from 'ollama';
import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import {
  convertOllamaMessagesToLangChain,
  convertToOllamaMessages,
} from './utils';

export class ChatOllama extends BaseChatOllama {
  static lc_name(): 'LibreChatOllama' {
    return 'LibreChatOllama';
  }
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.checkOrPullModel) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      if (!((await this.checkModelExistsOnMachine(this.model)) as boolean)) {
        await this.pull(this.model, {
          logProgress: true,
        });
      }
    }

    const params = this.invocationParams(options);
    // TODO: remove cast after SDK adds support for tool calls
    const ollamaMessages = convertToOllamaMessages(messages) as OllamaMessage[];

    const usageMetadata: UsageMetadata = {
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
    };

    const stream = await this.client.chat({
      ...params,
      messages: ollamaMessages,
      stream: true,
    });

    let lastMetadata: Omit<OllamaChatResponse, 'message'> | undefined;

    for await (const chunk of stream) {
      if (options.signal?.aborted === true) {
        this.client.abort();
      }
      const { message: responseMessage, ...rest } =
        chunk as Partial<OllamaChatResponse>;
      usageMetadata.input_tokens += rest.prompt_eval_count ?? 0;
      usageMetadata.output_tokens += rest.eval_count ?? 0;
      usageMetadata.total_tokens =
        usageMetadata.input_tokens + usageMetadata.output_tokens;
      lastMetadata = rest as Omit<OllamaChatResponse, 'message'>;
      if (!responseMessage) {
        continue;
      }
      const message = convertOllamaMessagesToLangChain(responseMessage);
      const generationChunk = new ChatGenerationChunk({
        text: responseMessage.content || '',
        message,
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        responseMessage.content || '',
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }

    // Yield the `response_metadata` as the final chunk.
    yield new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        response_metadata: lastMetadata,
        usage_metadata: usageMetadata,
      }),
    });
  }
}
