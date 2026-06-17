import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import { addBedrockTailCacheControl } from '@/messages/cache';
import { convertToConverseMessages } from './message_inputs';
import { toLangChainContent } from '@/messages/langchain';

/**
 * A Bedrock `cachePoint` is a message-level ContentBlock and is NOT a valid
 * `ToolResultContentBlock`. When the single tail prompt-cache breakpoint
 * anchors on a tool result (the common agent-loop shape), the cachePoint must
 * be hoisted out of `toolResult.content` to a message-level sibling — otherwise
 * Bedrock silently drops the breakpoint (no cache write, no cache read),
 * verified live against Bedrock Converse.
 */

interface ConverseBlock {
  text?: string;
  cachePoint?: { type?: string };
  toolResult?: {
    toolUseId?: string;
    content?: Array<{ text?: string; cachePoint?: { type?: string } }>;
  };
}

function toolUserMessage(
  result: ReturnType<typeof convertToConverseMessages>
): ConverseBlock[] {
  const msg = result.converseMessages.find(
    (m) =>
      m.role === 'user' && m.content?.some((c) => 'toolResult' in c) === true
  );
  return (msg?.content ?? []) as ConverseBlock[];
}

describe('convertToConverseMessages — tool-result cachePoint hoisting', () => {
  it('hoists a cachePoint out of toolResult.content to a message-level sibling', () => {
    const toolMsg = new ToolMessage({
      tool_call_id: 't1',
      content: toLangChainContent([
        { type: 'text', text: 'result body' },
        { cachePoint: { type: 'default' } },
      ] as MessageContentComplex[]),
    });

    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('go'),
      toolMsg,
    ]);

    const content = toolUserMessage({ converseMessages, converseSystem: [] });

    // toolResult body must NOT contain the cachePoint
    const toolResult = content.find((c) => 'toolResult' in c)?.toolResult;
    expect(toolResult?.content?.some((b) => 'cachePoint' in b)).toBe(false);
    expect(toolResult?.content).toEqual([{ text: 'result body' }]);

    // cachePoint must be a sibling AFTER the toolResult block
    expect(content[content.length - 1]).toEqual({
      cachePoint: { type: 'default' },
    });
  });

  it('leaves tool results without a cachePoint untouched', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('go'),
      new ToolMessage({ tool_call_id: 't1', content: 'plain result' }),
    ]);

    const content = toolUserMessage({ converseMessages, converseSystem: [] });
    expect(content).toEqual([
      { toolResult: { toolUseId: 't1', content: [{ text: 'plain result' }] } },
    ]);
  });

  it('end-to-end: tail breakpoint on a string tool result renders as a valid sibling cachePoint', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('What is 15 * 23? Use the calculator.'),
      new AIMessage({
        content: 'Calculating.',
        tool_calls: [
          { id: 't1', name: 'calculator', args: { expression: '15 * 23' } },
        ],
      }),
      new ToolMessage({ tool_call_id: 't1', content: '345' }),
    ];

    const cached = addBedrockTailCacheControl(messages);
    const { converseMessages } = convertToConverseMessages(cached);

    const content = toolUserMessage({ converseMessages, converseSystem: [] });
    const toolResult = content.find((c) => 'toolResult' in c)?.toolResult;

    // Exactly one cachePoint, at the message level, never nested in the body.
    expect(toolResult?.content?.some((b) => 'cachePoint' in b)).toBe(false);
    expect(content.filter((c) => 'cachePoint' in c)).toHaveLength(1);
    expect(content[content.length - 1]).toEqual({
      cachePoint: { type: 'default' },
    });
  });
});
