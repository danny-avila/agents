import { AIMessage } from '@langchain/core/messages';
import { convertMessagesToResponsesInput } from '@langchain/openai';
import type { TPayload } from '@/types';
import { expandOpenAIResponsesReasoningReplay } from '@/llm/openai';
import { ContentTypes, Providers } from '@/common';
import { formatAgentMessages } from './format';

const model = 'gpt-5.4';
const encryptedContent = 'encrypted-reasoning';
const replayItems = [
  {
    type: 'reasoning' as const,
    id: 'rs_123',
    status: 'completed' as const,
    summary: [
      {
        type: 'summary_text' as const,
        text: 'Checked the constraint.',
      },
    ],
    encrypted_content: encryptedContent,
  },
  {
    type: 'reasoning' as const,
    id: 'rs_456',
    status: 'completed' as const,
    summary: [],
    encrypted_content: 'encrypted-follow-up',
  },
];
const payload: TPayload = [
  {
    role: 'assistant',
    content: [
      {
        type: ContentTypes.THINK,
        think: 'Checked the constraint.',
        provider_replay: {
          openai_responses: {
            provider: Providers.OPENAI,
            model,
            items: replayItems,
          },
        },
      },
      {
        type: ContentTypes.TEXT,
        text: 'The answer is 42.',
      },
    ],
  },
];

describe('formatAgentMessages OpenAI Responses reasoning replay', () => {
  it('reconstructs encrypted reasoning for a stateless Responses request', () => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      payload,
      { 0: 900 },
      undefined,
      undefined,
      {
        provider: Providers.OPENAI,
        model,
        useResponsesApi: true,
      }
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(
      (messages[0] as AIMessage).additional_kwargs
        .openai_responses_reasoning_replay
    ).toEqual(replayItems);

    const input = convertMessagesToResponsesInput({
      messages: expandOpenAIResponsesReasoningReplay(messages),
      model,
      zdrEnabled: true,
    });
    expect(input[0]).toMatchObject({
      type: 'reasoning',
      id: 'rs_123',
      encrypted_content: encryptedContent,
    });
    expect(input[1]).toMatchObject({
      type: 'reasoning',
      id: 'rs_456',
      encrypted_content: 'encrypted-follow-up',
    });
    expect(input[2]).toMatchObject({
      type: 'message',
      role: 'assistant',
    });
    expect(indexTokenCountMap).toEqual({ 0: 900 });
  });

  it.each([
    {
      name: 'a different model',
      options: {
        provider: Providers.OPENAI,
        model: 'gpt-5.5',
        useResponsesApi: true,
      },
    },
    {
      name: 'Chat Completions',
      options: {
        provider: Providers.OPENAI,
        model,
        useResponsesApi: false,
      },
    },
    {
      name: 'Azure OpenAI',
      options: {
        provider: Providers.AZURE,
        model,
        useResponsesApi: true,
      },
    },
    {
      name: 'a different provider',
      options: {
        provider: Providers.ANTHROPIC,
        model,
        useResponsesApi: true,
      },
    },
  ])('does not replay encrypted reasoning to $name', ({ options }) => {
    const { messages, indexTokenCountMap } = formatAgentMessages(
      payload,
      { 0: 900 },
      undefined,
      undefined,
      options
    );

    expect(JSON.stringify(messages)).not.toContain(encryptedContent);
    expect(indexTokenCountMap).toEqual({});
  });
});
