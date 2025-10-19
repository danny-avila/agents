import type Anthropic from '@anthropic-ai/sdk';
import type { BaseMessage } from '@langchain/core/messages';
export type AnthropicMessages = Array<AnthropicMessage | BaseMessage>;
export type AnthropicMessage = Anthropic.MessageParam;
