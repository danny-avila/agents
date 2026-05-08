import { HumanMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { formatAgentMessages } from './format';
import { ContentTypes, Constants } from '@/common';

/** Helper to build a skill tool_call content part */
function skillToolCall(
  id: string,
  skillName: string,
  output = 'Skill loaded.',
): Record<string, unknown> {
  return {
    type: ContentTypes.TOOL_CALL,
    tool_call: {
      id,
      name: Constants.SKILL_TOOL,
      args: JSON.stringify({ skillName }),
      output,
    },
  };
}

describe('formatAgentMessages skill body reconstruction', () => {
  const skillBodies = new Map([
    ['pdf-analyzer', '# PDF Analyzer\nAnalyze PDF files step by step.'],
    ['code-review', '# Code Review\nReview the code for issues.'],
  ]);

  describe('with discoveredTools (tools filtering active)', () => {
    const tools = new Set([Constants.SKILL_TOOL, 'web_search']);

    it('reconstructs HumanMessage after skill ToolMessage', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Analyze this PDF' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'I\'ll invoke the skill.',
              tool_call_ids: ['call_1'],
            },
            skillToolCall('call_1', 'pdf-analyzer'),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      // user, AI, ToolMessage, injected HumanMessage
      expect(messages.length).toBeGreaterThanOrEqual(4);
      const last = messages[messages.length - 1];
      expect(last).toBeInstanceOf(HumanMessage);
      expect(last.content).toBe('# PDF Analyzer\nAnalyze PDF files step by step.');
      expect((last as HumanMessage).additional_kwargs.source).toBe('skill');
      expect((last as HumanMessage).additional_kwargs.skillName).toBe('pdf-analyzer');
      expect((last as HumanMessage).additional_kwargs.isMeta).toBe(true);
    });

    it('does NOT inject body when skill tool is not in discoveredTools', () => {
      const restrictedTools = new Set(['web_search']); // skill NOT allowed
      const payload: TPayload = [
        { role: 'user', content: 'Analyze this' },
        {
          role: 'assistant',
          content: [skillToolCall('call_1', 'pdf-analyzer')],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, restrictedTools, skillBodies);

      const humanMessages = messages.filter((m) => m instanceof HumanMessage);
      // Only the user message, no injected skill body
      expect(humanMessages).toHaveLength(1);
    });

    it('does not inject when skill name is not in skills Map', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [skillToolCall('call_1', 'unknown-skill')],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const humanMessages = messages.filter((m) => m instanceof HumanMessage);
      expect(humanMessages).toHaveLength(1); // only the user message
    });
  });

  describe('without discoveredTools (no tools filtering)', () => {
    it('reconstructs HumanMessage when skills Map provided', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Review my code' },
        {
          role: 'assistant',
          content: [skillToolCall('call_1', 'code-review')],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, undefined, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(1);
      expect(injected[0].content).toBe('# Code Review\nReview the code for issues.');
    });

    it('no injection when skills Map is undefined', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [skillToolCall('call_1', 'pdf-analyzer')],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, undefined, undefined);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(0);
    });

    it('no injection when skills Map is empty', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [skillToolCall('call_1', 'pdf-analyzer')],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, undefined, new Map());

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(0);
    });
  });

  describe('extractSkillName edge cases', () => {
    const tools = new Set([Constants.SKILL_TOOL]);

    it('handles object args (not stringified)', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Go' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'call_1',
                name: Constants.SKILL_TOOL,
                args: { skillName: 'pdf-analyzer' }, // object, not string
                output: 'Loaded.',
              },
            },
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(1);
    });

    it('gracefully skips malformed JSON args', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Go' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'call_1',
                name: Constants.SKILL_TOOL,
                args: '{bad json',
                output: 'Loaded.',
              },
            },
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(0); // gracefully skipped
    });

    it('skips empty skillName', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Go' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'call_1',
                name: Constants.SKILL_TOOL,
                args: JSON.stringify({ skillName: '' }),
                output: 'Loaded.',
              },
            },
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(0);
    });
  });

  describe('deduplication', () => {
    const tools = new Set([Constants.SKILL_TOOL]);

    it('injects body only once when same skill invoked twice in one message', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Go' },
        {
          role: 'assistant',
          content: [
            skillToolCall('call_1', 'pdf-analyzer'),
            skillToolCall('call_2', 'pdf-analyzer'),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(1);
    });

    it('injects body for each distinct skill invoked', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Go' },
        {
          role: 'assistant',
          content: [
            skillToolCall('call_1', 'pdf-analyzer'),
            skillToolCall('call_2', 'code-review'),
          ],
        },
      ];

      const { messages } = formatAgentMessages(payload, undefined, tools, skillBodies);

      const injected = messages.filter(
        (m) => m instanceof HumanMessage && (m as HumanMessage).additional_kwargs?.source === 'skill',
      );
      expect(injected).toHaveLength(2);
      const names = injected.map((m) => (m as HumanMessage).additional_kwargs.skillName);
      expect(names).toContain('pdf-analyzer');
      expect(names).toContain('code-review');
    });
  });

  describe('indexTokenCountMap distribution', () => {
    const tools = new Set([Constants.SKILL_TOOL]);

    it('excludes injected HumanMessages from assistant token distribution', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Analyze this' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Invoking skill.',
              tool_call_ids: ['call_1'],
            },
            skillToolCall('call_1', 'pdf-analyzer'),
          ],
        },
      ];

      const inputTokenMap: Record<number, number | undefined> = {
        0: 100, // user message
        1: 500, // assistant message
      };

      const { messages, indexTokenCountMap } = formatAgentMessages(
        payload,
        inputTokenMap,
        tools,
        skillBodies,
      );

      // There should be messages: user, AI, ToolMessage, injected HumanMessage
      expect(messages.length).toBeGreaterThanOrEqual(4);
      const lastMsg = messages[messages.length - 1];
      expect(lastMsg).toBeInstanceOf(HumanMessage);
      expect((lastMsg as HumanMessage).additional_kwargs.source).toBe('skill');

      // Token map must be defined when input was provided
      expect(indexTokenCountMap).toBeDefined();

      // The injected HumanMessage's index should NOT be in the token map
      const injectedIndex = messages.length - 1;
      expect(indexTokenCountMap![injectedIndex]).toBeUndefined();

      // The assistant's 500 tokens should be distributed only across
      // the AI + ToolMessage, NOT the injected HumanMessage
      let assistantTotal = 0;
      for (const [idx, count] of Object.entries(indexTokenCountMap!)) {
        if (Number(idx) > 0 && Number(idx) < injectedIndex) {
          assistantTotal += count ?? 0;
        }
      }
      expect(assistantTotal).toBe(500);
    });
  });
});
