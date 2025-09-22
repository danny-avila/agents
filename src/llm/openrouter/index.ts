import type { ChatOpenAICallOptions, OpenAIClient } from '@langchain/openai';
import type {
  AIMessageChunk,
  HumanMessageChunk,
  SystemMessageChunk,
  FunctionMessageChunk,
  ToolMessageChunk,
  ChatMessageChunk,
} from '@langchain/core/messages';
import { ChatOpenAI } from '@/llm/openai';

export interface ChatOpenRouterCallOptions extends ChatOpenAICallOptions {
  include_reasoning?: boolean;
}
export class ChatOpenRouter extends ChatOpenAI {
  constructor(_fields: Partial<ChatOpenRouterCallOptions>) {
    const { include_reasoning, ...fields } = _fields;
    super({
      ...fields,
      modelKwargs: {
        include_reasoning,
      },
    });
  }
  static lc_name(): 'LibreChatOpenRouter' {
    return 'LibreChatOpenRouter';
  }
  protected override _convertOpenAIDeltaToBaseMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    rawResponse: OpenAIClient.ChatCompletionChunk,
    defaultRole?:
      | 'function'
      | 'user'
      | 'system'
      | 'developer'
      | 'assistant'
      | 'tool'
  ):
    | AIMessageChunk
    | HumanMessageChunk
    | SystemMessageChunk
    | FunctionMessageChunk
    | ToolMessageChunk
    | ChatMessageChunk {
    const messageChunk = super._convertOpenAIDeltaToBaseMessageChunk(
      delta,
      rawResponse,
      defaultRole
    );
    messageChunk.additional_kwargs.reasoning = delta.reasoning;
    return messageChunk;
  }
}
