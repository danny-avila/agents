import { ContentTypes } from '@/common';
import { labelContentByAgent } from './format';
import type { MessageContentComplex, ToolCallContent } from '@/types';

/**
 * Type guard to check if content is ToolCallContent
 */
function isToolCallContent(
  content: MessageContentComplex
): content is ToolCallContent {
  return content.type === ContentTypes.TOOL_CALL && 'tool_call' in content;
}

/**
 * Type guard to check if content has text property
 */
function hasTextProperty(
  content: MessageContentComplex
): content is MessageContentComplex & { text: string } {
  return 'text' in content;
}

describe('labelContentByAgent', () => {
  describe('Basic functionality', () => {
    it('should return contentParts unchanged when no agentIdMap provided', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Hello world' },
      ];

      const result = labelContentByAgent(contentParts, undefined);

      expect(result).toEqual(contentParts);
      expect(result.length).toBe(1);
    });

    it('should return contentParts unchanged when agentIdMap is empty', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Hello world' },
      ];

      const result = labelContentByAgent(contentParts, {});

      expect(result).toEqual(contentParts);
      expect(result.length).toBe(1);
    });

    it('should handle empty contentParts array', () => {
      const contentParts: MessageContentComplex[] = [];
      const agentIdMap = {};

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result).toEqual([]);
    });
  });

  describe('Transfer-based labeling (default)', () => {
    it('should consolidate transferred agent content into transfer tool output', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: '' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'call_123',
            name: 'lc_transfer_to_specialist',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Specialist response here' },
      ];

      const agentIdMap = {
        0: 'supervisor',
        1: 'supervisor',
        2: 'specialist',
      };

      const agentNames = {
        supervisor: 'Supervisor',
        specialist: 'Specialist Agent',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, agentNames);

      // Should have 2 items: empty text + modified transfer tool call
      expect(result.length).toBe(2);
      expect(result[0].type).toBe(ContentTypes.TEXT);

      // The transfer tool call should have consolidated output
      expect(result[1].type).toBe(ContentTypes.TOOL_CALL);
      const toolCallContent = result[1] as ToolCallContent;
      expect(toolCallContent.tool_call?.output).toContain(
        '--- Transfer to Specialist Agent ---'
      );
      expect(toolCallContent.tool_call?.output).toContain('"type":"text"');
      expect(toolCallContent.tool_call?.output).toContain(
        '"text":"Specialist response here"'
      );
      expect(toolCallContent.tool_call?.output).toContain(
        '--- End of Specialist Agent response ---'
      );
    });

    it('should handle multiple content types from transferred agent', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_1',
            name: 'lc_transfer_to_analyst',
            args: '',
          },
        },
        { type: ContentTypes.THINK, think: 'Analyzing the problem...' },
        { type: ContentTypes.TEXT, text: 'Here is my analysis' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_1',
            name: 'search',
            args: '{"query":"test"}',
          },
        },
      ];

      const agentIdMap = {
        0: 'supervisor',
        1: 'analyst',
        2: 'analyst',
        3: 'analyst',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result.length).toBe(1);
      expect(isToolCallContent(result[0])).toBe(true);
      if (isToolCallContent(result[0])) {
        expect(result[0].tool_call?.output).toContain('"type":"think"');
        expect(result[0].tool_call?.output).toContain('"type":"text"');
        expect(result[0].tool_call?.output).toContain('"type":"tool_call"');
        expect(result[0].tool_call?.output).toContain(
          'Analyzing the problem...'
        );
        expect(result[0].tool_call?.output).toContain('Here is my analysis');
      }
    });

    it('should use agentId when agentNames not provided', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'call_1',
            name: 'lc_transfer_to_agent2',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Response from agent2' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent2',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(isToolCallContent(result[0])).toBe(true);
      if (isToolCallContent(result[0])) {
        expect(result[0].tool_call?.output).toContain(
          '--- Transfer to agent2 ---'
        );
        expect(result[0].tool_call?.output).toContain('agent2:');
      }
    });

    it('should handle sequential transfers (agent1 -> agent2 -> agent3)', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Starting' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_1',
            name: 'lc_transfer_to_agent2',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Agent2 response' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_2',
            name: 'lc_transfer_to_agent3',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Agent3 final response' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent1',
        2: 'agent2',
        3: 'agent2',
        4: 'agent3',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result.length).toBe(3);
      expect(result[0].type).toBe(ContentTypes.TEXT);
      expect(result[1].type).toBe(ContentTypes.TOOL_CALL);
      expect(result[2].type).toBe(ContentTypes.TOOL_CALL);

      // First transfer should have agent2 content
      if (isToolCallContent(result[1])) {
        expect(result[1].tool_call?.output).toContain('agent2');
        expect(result[1].tool_call?.output).toContain('Agent2 response');
      }

      // Second transfer should have agent3 content
      if (isToolCallContent(result[2])) {
        expect(result[2].tool_call?.output).toContain('agent3');
        expect(result[2].tool_call?.output).toContain('Agent3 final response');
      }
    });
  });

  describe('Full agent labeling (labelNonTransferContent: true)', () => {
    it('should label all agent content when labelNonTransferContent is true', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Researcher coordinating' },
        { type: ContentTypes.TEXT, text: 'FINANCIAL ANALYSIS: Revenue impact' },
        {
          type: ContentTypes.TEXT,
          text: 'TECHNICAL ANALYSIS: System requirements',
        },
        { type: ContentTypes.TEXT, text: 'Summary of all analyses' },
      ];

      const agentIdMap = {
        0: 'researcher',
        1: 'analyst1',
        2: 'analyst2',
        3: 'summarizer',
      };

      const agentNames = {
        researcher: 'Research Coordinator',
        analyst1: 'Financial Analyst',
        analyst2: 'Technical Analyst',
        summarizer: 'Synthesis Expert',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, agentNames, {
        labelNonTransferContent: true,
      });

      // Should create 4 labeled groups
      expect(result.length).toBe(4);

      // Each should be a text content part with agent labels
      expect(result[0].type).toBe(ContentTypes.TEXT);
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('--- Research Coordinator ---');
        expect(result[0].text).toContain(
          'Research Coordinator: Researcher coordinating'
        );
        expect(result[0].text).toContain('--- End of Research Coordinator ---');
      }

      expect(result[1].type).toBe(ContentTypes.TEXT);
      if (hasTextProperty(result[1])) {
        expect(result[1].text).toContain('--- Financial Analyst ---');
        expect(result[1].text).toContain(
          'Financial Analyst: FINANCIAL ANALYSIS: Revenue impact'
        );
      }

      expect(result[2].type).toBe(ContentTypes.TEXT);
      if (hasTextProperty(result[2])) {
        expect(result[2].text).toContain('--- Technical Analyst ---');
        expect(result[2].text).toContain(
          'Technical Analyst: TECHNICAL ANALYSIS: System requirements'
        );
      }

      expect(result[3].type).toBe(ContentTypes.TEXT);
      if (hasTextProperty(result[3])) {
        expect(result[3].text).toContain('--- Synthesis Expert ---');
        expect(result[3].text).toContain(
          'Synthesis Expert: Summary of all analyses'
        );
      }
    });

    it('should group consecutive content from same agent', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'First message' },
        { type: ContentTypes.TEXT, text: 'Second message' },
        { type: ContentTypes.TEXT, text: 'Third message' },
        { type: ContentTypes.TEXT, text: 'Different agent' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent1',
        2: 'agent1',
        3: 'agent2',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Should create 2 groups
      expect(result.length).toBe(2);

      // First group has all 3 messages from agent1
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('agent1: First message');
        expect(result[0].text).toContain('agent1: Second message');
        expect(result[0].text).toContain('agent1: Third message');
      }

      // Second group has agent2 message
      if (hasTextProperty(result[1])) {
        expect(result[1].text).toContain('agent2: Different agent');
      }
    });

    it('should handle thinking content in parallel labeling', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.THINK, think: 'Let me analyze this...' },
        { type: ContentTypes.TEXT, text: 'My conclusion' },
      ];

      const agentIdMap = {
        0: 'analyst',
        1: 'analyst',
      };

      const agentNames = {
        analyst: 'Expert Analyst',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, agentNames, {
        labelNonTransferContent: true,
      });

      expect(result.length).toBe(1);
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('--- Expert Analyst ---');
        expect(result[0].text).toContain('"type":"think"');
        expect(result[0].text).toContain('Let me analyze this...');
        expect(result[0].text).toContain('Expert Analyst: My conclusion');
      }
    });

    it('should skip empty text content in parallel labeling', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: '' },
        { type: ContentTypes.TEXT, text: 'Valid content' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent1',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      expect(result.length).toBe(1);
      // Should only contain the valid content, not the empty string
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('agent1: Valid content');
        expect(result[0].text).not.toContain('agent1: \n');
      }
    });
  });

  describe('Edge cases', () => {
    it('should handle content parts without agentId in map', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Message 1' },
        { type: ContentTypes.TEXT, text: 'Message 2' },
        { type: ContentTypes.TEXT, text: 'Message 3' },
      ];

      const agentIdMap = {
        0: 'agent1',
        // Missing index 1
        2: 'agent2',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Should still process and group by available agent IDs
      expect(result.length).toBeGreaterThan(0);
    });

    it('should handle transfer tool without subsequent agent content', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_1',
            name: 'lc_transfer_to_specialist',
            args: '',
          },
        },
      ];

      const agentIdMap = {
        0: 'supervisor',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      // Transfer tool should still be present, just without added content
      expect(result.length).toBe(1);
      expect(result[0].type).toBe(ContentTypes.TOOL_CALL);
    });

    it('should handle multiple transfers in sequence', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_1',
            name: 'lc_transfer_to_agent_a',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Agent A response' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_2',
            name: 'lc_transfer_to_agent_b',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Agent B response' },
      ];

      const agentIdMap = {
        0: 'supervisor',
        1: 'agent_a',
        2: 'agent_a',
        3: 'agent_b',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result.length).toBe(2);

      // Both transfers should have consolidated outputs
      if (isToolCallContent(result[0])) {
        expect(result[0].tool_call?.output).toContain('agent_a');
        expect(result[0].tool_call?.output).toContain('Agent A response');
      }

      if (isToolCallContent(result[1])) {
        expect(result[1].tool_call?.output).toContain('agent_b');
        expect(result[1].tool_call?.output).toContain('Agent B response');
      }
    });

    it('should preserve non-transfer tool calls unchanged', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Using a tool' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_1',
            name: 'search',
            args: '{"query":"test"}',
            output: 'Search results',
          },
        },
        { type: ContentTypes.TEXT, text: 'Here are the results' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent1',
        2: 'agent1',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      // All content from same agent with no transfers, should pass through
      expect(result).toEqual(contentParts);
    });
  });

  describe('Parallel patterns', () => {
    it('should label parallel analyst contributions separately', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Coordinating research' },
        { type: ContentTypes.TEXT, text: 'FINANCIAL: Budget analysis' },
        { type: ContentTypes.TEXT, text: 'TECHNICAL: System design' },
        { type: ContentTypes.TEXT, text: 'MARKET: Competitive landscape' },
        { type: ContentTypes.TEXT, text: 'Integrated summary' },
      ];

      const agentIdMap = {
        0: 'researcher',
        1: 'financial_analyst',
        2: 'technical_analyst',
        3: 'market_analyst',
        4: 'summarizer',
      };

      const agentNames = {
        researcher: 'Research Coordinator',
        financial_analyst: 'Financial Analyst',
        technical_analyst: 'Technical Analyst',
        market_analyst: 'Market Analyst',
        summarizer: 'Synthesis Expert',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, agentNames, {
        labelNonTransferContent: true,
      });

      // Should have 5 labeled groups (one per agent)
      expect(result.length).toBe(5);

      // Verify each group
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('--- Research Coordinator ---');
        expect(result[0].text).toContain('Coordinating research');
      }

      if (hasTextProperty(result[1])) {
        expect(result[1].text).toContain('--- Financial Analyst ---');
        expect(result[1].text).toContain('FINANCIAL: Budget analysis');
      }

      if (hasTextProperty(result[2])) {
        expect(result[2].text).toContain('--- Technical Analyst ---');
        expect(result[2].text).toContain('TECHNICAL: System design');
      }

      if (hasTextProperty(result[3])) {
        expect(result[3].text).toContain('--- Market Analyst ---');
        expect(result[3].text).toContain('MARKET: Competitive landscape');
      }

      if (hasTextProperty(result[4])) {
        expect(result[4].text).toContain('--- Synthesis Expert ---');
        expect(result[4].text).toContain('Integrated summary');
      }
    });

    it('should handle agent alternation in parallel mode', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'A1' },
        { type: ContentTypes.TEXT, text: 'B1' },
        { type: ContentTypes.TEXT, text: 'A2' },
        { type: ContentTypes.TEXT, text: 'B2' },
      ];

      const agentIdMap = {
        0: 'agentA',
        1: 'agentB',
        2: 'agentA',
        3: 'agentB',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Should create 4 groups (alternating agents)
      expect(result.length).toBe(4);

      if (hasTextProperty(result[0]))
        expect(result[0].text).toContain('agentA: A1');
      if (hasTextProperty(result[1]))
        expect(result[1].text).toContain('agentB: B1');
      if (hasTextProperty(result[2]))
        expect(result[2].text).toContain('agentA: A2');
      if (hasTextProperty(result[3]))
        expect(result[3].text).toContain('agentB: B2');
    });

    it('should handle tool calls in parallel labeling', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Analyzing' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_1',
            name: 'search',
            args: '{"query":"data"}',
          },
        },
        { type: ContentTypes.TEXT, text: 'Results analyzed' },
      ];

      const agentIdMap = {
        0: 'analyst',
        1: 'analyst',
        2: 'analyst',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      expect(result.length).toBe(1);
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('--- analyst ---');
        expect(result[0].text).toContain('analyst: Analyzing');
        expect(result[0].text).toContain('"type":"tool_call"');
        expect(result[0].text).toContain('"name":"search"');
        expect(result[0].text).toContain('analyst: Results analyzed');
      }
    });
  });

  describe('Mixed patterns', () => {
    it('should handle content with no transfer but mixed agents (without option)', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Agent 1 says this' },
        { type: ContentTypes.TEXT, text: 'Agent 2 says this' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent2',
      };

      // Default mode (transfer-based) should pass through non-transfer content
      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result).toEqual(contentParts);
    });

    it('should handle hybrid: transfer followed by non-transfer agents', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'transfer_1',
            name: 'lc_transfer_to_specialist',
            args: '',
          },
        },
        { type: ContentTypes.TEXT, text: 'Specialist work' },
        { type: ContentTypes.TEXT, text: 'Another agent response' },
      ];

      const agentIdMap = {
        0: 'supervisor',
        1: 'specialist',
        2: 'agent3',
      };

      const result = labelContentByAgent(contentParts, agentIdMap);

      expect(result.length).toBe(2);

      // Transfer tool should have specialist content
      if (isToolCallContent(result[0])) {
        expect(result[0].tool_call?.output).toContain('specialist');
        expect(result[0].tool_call?.output).toContain('Specialist work');
      }

      // Non-transfer content should pass through
      expect(result[1]).toEqual(contentParts[2]);
    });
  });

  describe('Real-world scenarios', () => {
    it('should handle supervisor -> legal_advisor handoff pattern', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: '' },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'call_legal',
            name: 'lc_transfer_to_legal_advisor',
            args: '',
          },
        },
        {
          type: ContentTypes.TEXT,
          text: 'GPL licensing creates obligations...',
        },
      ];

      const agentIdMap = {
        0: 'supervisor',
        1: 'supervisor',
        2: 'legal_advisor',
      };

      const agentNames = {
        supervisor: 'Supervisor',
        legal_advisor: 'Legal Advisor',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, agentNames);

      expect(result.length).toBe(2);
      if (isToolCallContent(result[1])) {
        expect(result[1].tool_call?.output).toContain(
          '--- Transfer to Legal Advisor ---'
        );
        expect(result[1].tool_call?.output).toContain('"type":"text"');
        expect(result[1].tool_call?.output).toContain(
          '"text":"GPL licensing creates obligations..."'
        );
        expect(result[1].tool_call?.output).toContain(
          '--- End of Legal Advisor response ---'
        );
      }
    });

    it('should handle fan-out to 3 analysts then fan-in to summarizer', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Coordination brief' },
        { type: ContentTypes.TEXT, text: 'Financial analysis content' },
        { type: ContentTypes.TEXT, text: 'Technical analysis content' },
        { type: ContentTypes.TEXT, text: 'Market analysis content' },
        { type: ContentTypes.TEXT, text: 'Executive summary' },
      ];

      const agentIdMap = {
        0: 'researcher',
        1: 'analyst1',
        2: 'analyst2',
        3: 'analyst3',
        4: 'summarizer',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      expect(result.length).toBe(5);

      // Each analyst's work should be clearly separated
      if (hasTextProperty(result[0]))
        expect(result[0].text).toContain('researcher:');
      if (hasTextProperty(result[1]))
        expect(result[1].text).toContain('analyst1:');
      if (hasTextProperty(result[2]))
        expect(result[2].text).toContain('analyst2:');
      if (hasTextProperty(result[3]))
        expect(result[3].text).toContain('analyst3:');
      if (hasTextProperty(result[4]))
        expect(result[4].text).toContain('summarizer:');
    });
  });

  describe('Performance and edge cases', () => {
    it('should handle large number of content parts efficiently', () => {
      const contentParts: MessageContentComplex[] = [];
      const agentIdMap: Record<number, string> = {};

      // Create 100 content parts from 10 different agents
      for (let i = 0; i < 100; i++) {
        contentParts.push({
          type: ContentTypes.TEXT,
          text: `Message ${i}`,
        });
        agentIdMap[i] = `agent${i % 10}`;
      }

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Should complete without errors
      expect(result.length).toBeGreaterThan(0);
      expect(result.length).toBeLessThanOrEqual(100);
    });

    it('should handle all content from single agent', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Part 1' },
        { type: ContentTypes.TEXT, text: 'Part 2' },
        { type: ContentTypes.TEXT, text: 'Part 3' },
      ];

      const agentIdMap = {
        0: 'single_agent',
        1: 'single_agent',
        2: 'single_agent',
      };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Should create 1 group with all content
      expect(result.length).toBe(1);
      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('single_agent: Part 1');
        expect(result[0].text).toContain('single_agent: Part 2');
        expect(result[0].text).toContain('single_agent: Part 3');
      }
    });
  });

  describe('Content type handling', () => {
    it('should properly format thinking content with JSON', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.THINK,
          think: 'I need to consider multiple factors...',
        },
      ];

      const agentIdMap = { 0: 'analyst' };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      if (hasTextProperty(result[0])) {
        const parsed = JSON.parse(
          result[0].text.split('analyst: ')[1].split('\n')[0]
        );
        expect(parsed.type).toBe('think');
        expect(parsed.think).toBe('I need to consider multiple factors...');
      }
    });

    it('should properly format tool call content with JSON', () => {
      const contentParts: MessageContentComplex[] = [
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_123',
            name: 'calculator',
            args: { expression: '2+2' },
            output: '4',
          },
        },
      ];

      const agentIdMap = { 0: 'agent1' };

      const result = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      if (hasTextProperty(result[0])) {
        expect(result[0].text).toContain('"type":"tool_call"');
        expect(result[0].text).toContain('"name":"calculator"');
        expect(result[0].text).toContain('"id":"tool_123"');
      }
    });
  });

  describe('Integration scenarios', () => {
    it('should work with formatAgentMessages pipeline', () => {
      const contentParts: MessageContentComplex[] = [
        { type: ContentTypes.TEXT, text: 'Agent 1 content' },
        { type: ContentTypes.TEXT, text: 'Agent 2 content' },
      ];

      const agentIdMap = {
        0: 'agent1',
        1: 'agent2',
      };

      const labeled = labelContentByAgent(contentParts, agentIdMap, undefined, {
        labelNonTransferContent: true,
      });

      // Labeled content should be valid for formatAgentMessages
      expect(labeled.length).toBeGreaterThan(0);
      expect(labeled.every((part) => part.type != null)).toBe(true);
    });
  });
});
