import type { Anthropic } from '@anthropic-ai/sdk';
import { AnthropicToolChoice } from '../types.js';

export function handleToolChoice(
  toolChoice?: AnthropicToolChoice
):
  | Anthropic.Messages.ToolChoiceAuto
  | Anthropic.Messages.ToolChoiceAny
  | Anthropic.Messages.ToolChoiceTool
  | Anthropic.Messages.ToolChoiceNone
  | undefined {
  if (toolChoice == null) {
    return undefined;
  } else if (toolChoice === 'any' || toolChoice === 'required') {
    // "required" is the OpenAI-style alias for forcing tool use.
    return {
      type: 'any',
    };
  } else if (toolChoice === 'auto') {
    return {
      type: 'auto',
    };
  } else if (toolChoice === 'none') {
    return {
      type: 'none',
    };
  } else if (typeof toolChoice === 'string') {
    return {
      type: 'tool',
      name: toolChoice,
    };
  } else {
    return toolChoice;
  }
}
