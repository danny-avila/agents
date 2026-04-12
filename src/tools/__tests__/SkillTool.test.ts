import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import { Constants } from '@/common';
import {
  SkillToolName,
  SkillToolSchema,
  SkillToolDescription,
  SkillToolDefinition,
  createSkillTool,
} from '../SkillTool';

describe('SkillTool', () => {
  describe('schema validation', () => {
    it('validates correct input with skillName only', () => {
      const result = SkillToolSchema.safeParse({ skillName: 'pdf-processor' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.skillName).toBe('pdf-processor');
        expect(result.data.args).toBeUndefined();
      }
    });

    it('validates correct input with skillName and args', () => {
      const result = SkillToolSchema.safeParse({
        skillName: 'review-pr',
        args: '123',
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.skillName).toBe('review-pr');
        expect(result.data.args).toBe('123');
      }
    });

    it('rejects missing skillName', () => {
      const result = SkillToolSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    it('rejects non-string skillName', () => {
      const result = SkillToolSchema.safeParse({ skillName: 123 });
      expect(result.success).toBe(false);
    });
  });

  describe('createSkillTool', () => {
    it('throws on direct invocation', async () => {
      const skillTool = createSkillTool();
      await expect(skillTool.invoke({ skillName: 'test' })).rejects.toThrow(
        'SkillTool requires event-driven execution mode (ON_TOOL_EXECUTE). Direct invocation is not supported.'
      );
    });

    it('has correct name', () => {
      const skillTool = createSkillTool();
      expect(skillTool.name).toBe('skill');
    });
  });

  describe('SkillToolDefinition', () => {
    it('has correct name', () => {
      expect(SkillToolDefinition.name).toBe(Constants.SKILL_TOOL);
    });

    it('has correct parameter schema', () => {
      expect(SkillToolDefinition.parameters.required).toContain('skillName');
      expect(SkillToolDefinition.parameters.properties.skillName.type).toBe(
        'string'
      );
      expect(SkillToolDefinition.parameters.properties.args.type).toBe(
        'string'
      );
    });

    it('has a description', () => {
      expect(SkillToolDefinition.description).toBe(SkillToolDescription);
      expect(SkillToolDefinition.description.length).toBeGreaterThan(0);
    });
  });

  describe('SkillToolName', () => {
    it('equals Constants.SKILL_TOOL', () => {
      expect(SkillToolName).toBe('skill');
      expect(SkillToolName).toBe(Constants.SKILL_TOOL);
    });
  });

  describe('InjectedMessage type-check', () => {
    it('constructs a valid ToolExecuteResult with injectedMessages', () => {
      const result: t.ToolExecuteResult = {
        toolCallId: 'call_1',
        content: 'Skill loaded successfully.',
        status: 'success',
        injectedMessages: [
          {
            role: 'user',
            content: '# PDF Processor Instructions\n\nFollow these steps...',
            isMeta: true,
            source: 'skill',
            skillName: 'pdf-processor',
          },
          {
            role: 'system',
            content: 'Skill files are available at /skills/pdf-processor/',
            source: 'skill',
            skillName: 'pdf-processor',
          },
        ],
      };

      expect(result.injectedMessages).toHaveLength(2);
      expect(result.injectedMessages![0].role).toBe('user');
      expect(result.injectedMessages![1].role).toBe('system');
    });
  });

  describe('ToolNode injectedMessages plumbing (event-driven)', () => {
    it('prepends injected messages before ToolMessages in output', async () => {
      const dummyTool = tool(async () => 'dummy', {
        name: 'dummy',
        description: 'dummy',
        schema: z.object({ x: z.string() }),
      }) as unknown as StructuredToolInterface;

      const toolNode = new ToolNode({
        tools: [dummyTool],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_1', 'step_1']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_1', name: 'dummy', args: { x: 'hello' } }],
      });

      const injectedMessages: t.InjectedMessage[] = [
        {
          role: 'user',
          content: 'Injected skill body content',
          isMeta: true,
          source: 'skill',
          skillName: 'test-skill',
        },
        {
          role: 'system',
          content: 'System context hint',
          source: 'system',
        },
      ];

      const mockResults: t.ToolExecuteResult[] = [
        {
          toolCallId: 'call_1',
          content: 'Tool result text',
          status: 'success',
          injectedMessages,
        },
      ];

      jest
        .spyOn(await import('@/utils/events'), 'safeDispatchCustomEvent')
        .mockImplementation(async (_event, data) => {
          const request = data as t.ToolExecuteBatchRequest;
          if (request.resolve) {
            request.resolve(mockResults);
          }
        });

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(3);

      const first = messages[0] as HumanMessage;
      expect(first).toBeInstanceOf(HumanMessage);
      expect(first.content).toBe('Injected skill body content');
      expect(first.additional_kwargs.isMeta).toBe(true);
      expect(first.additional_kwargs.source).toBe('skill');
      expect(first.additional_kwargs.skillName).toBe('test-skill');

      const second = messages[1] as SystemMessage;
      expect(second).toBeInstanceOf(SystemMessage);
      expect(second.content).toBe('System context hint');
      expect(second.additional_kwargs.source).toBe('system');

      const third = messages[2];
      expect(third._getType()).toBe('tool');

      jest.restoreAllMocks();
    });

    it('returns only ToolMessages when no injectedMessages present', async () => {
      const dummyTool = tool(async () => 'dummy', {
        name: 'dummy',
        description: 'dummy',
        schema: z.object({ x: z.string() }),
      }) as unknown as StructuredToolInterface;

      const toolNode = new ToolNode({
        tools: [dummyTool],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_2', 'step_2']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_2', name: 'dummy', args: { x: 'test' } }],
      });

      const mockResults: t.ToolExecuteResult[] = [
        {
          toolCallId: 'call_2',
          content: 'Normal result',
          status: 'success',
        },
      ];

      jest
        .spyOn(await import('@/utils/events'), 'safeDispatchCustomEvent')
        .mockImplementation(async (_event, data) => {
          const request = data as t.ToolExecuteBatchRequest;
          if (request.resolve) {
            request.resolve(mockResults);
          }
        });

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(1);
      expect(messages[0]._getType()).toBe('tool');

      jest.restoreAllMocks();
    });
  });
});
