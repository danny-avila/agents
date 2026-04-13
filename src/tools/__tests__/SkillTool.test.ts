import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import * as events from '@/utils/events';
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
  describe('schema structure', () => {
    it('has skillName as required string property', () => {
      expect(SkillToolSchema.properties.skillName.type).toBe('string');
      expect(SkillToolSchema.required).toContain('skillName');
    });

    it('has args as optional string property', () => {
      expect(SkillToolSchema.properties.args.type).toBe('string');
      expect(SkillToolSchema.required).not.toContain('args');
    });

    it('is an object type schema', () => {
      expect(SkillToolSchema.type).toBe('object');
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

    it('validates skillName is required', async () => {
      const skillTool = createSkillTool();
      await expect(skillTool.invoke({})).rejects.toThrow();
    });
  });

  describe('SkillToolDefinition', () => {
    it('has correct name', () => {
      expect(SkillToolDefinition.name).toBe(Constants.SKILL_TOOL);
    });

    it('references the same SkillToolSchema object (no duplication)', () => {
      expect(SkillToolDefinition.parameters).toBe(SkillToolSchema);
    });

    it('has a non-empty description', () => {
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

    it('accepts MessageContentComplex[] content', () => {
      const result: t.ToolExecuteResult = {
        toolCallId: 'call_1',
        content: '',
        status: 'success',
        injectedMessages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Skill instructions here' },
              { type: 'image_url', image_url: { url: 'data:image/png;...' } },
            ],
            isMeta: true,
            source: 'skill',
            skillName: 'visual-skill',
          },
        ],
      };

      expect(Array.isArray(result.injectedMessages![0].content)).toBe(true);
    });
  });

  describe('ToolNode injectedMessages plumbing (event-driven)', () => {
    const createDummyTool = (name = 'dummy'): StructuredToolInterface =>
      tool(async () => 'dummy', {
        name,
        description: 'dummy',
        schema: z.object({ x: z.string() }),
      }) as unknown as StructuredToolInterface;

    function mockEventDispatch(
      mockResults: t.ToolExecuteResult[]
    ): jest.SpyInstance {
      return jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (_event, data) => {
          const request = data as Record<string, unknown>;
          if (typeof request.resolve === 'function') {
            (request.resolve as (r: t.ToolExecuteResult[]) => void)(
              mockResults
            );
          }
        });
    }

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('appends injected messages AFTER ToolMessages in output', async () => {
      const toolNode = new ToolNode({
        tools: [createDummyTool()],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_1', 'step_1']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_1', name: 'dummy', args: { x: 'hello' } }],
      });

      mockEventDispatch([
        {
          toolCallId: 'call_1',
          content: 'Tool result text',
          status: 'success',
          injectedMessages: [
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
          ],
        },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(3);

      // ToolMessage comes FIRST (preserves AIMessage -> ToolMessage adjacency)
      expect(messages[0]._getType()).toBe('tool');

      // Injected messages come AFTER
      const second = messages[1] as HumanMessage;
      expect(second).toBeInstanceOf(HumanMessage);
      expect(second.content).toBe('Injected skill body content');
      expect(second.additional_kwargs.isMeta).toBe(true);
      expect(second.additional_kwargs.source).toBe('skill');
      expect(second.additional_kwargs.skillName).toBe('test-skill');

      // role: 'system' also becomes HumanMessage (avoids provider rejections)
      const third = messages[2] as HumanMessage;
      expect(third).toBeInstanceOf(HumanMessage);
      expect(third.content).toBe('System context hint');
      expect(third.additional_kwargs.role).toBe('system');
      expect(third.additional_kwargs.source).toBe('system');
    });

    it('returns only ToolMessages when no injectedMessages present', async () => {
      const toolNode = new ToolNode({
        tools: [createDummyTool()],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_2', 'step_2']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_2', name: 'dummy', args: { x: 'test' } }],
      });

      mockEventDispatch([
        { toolCallId: 'call_2', content: 'Normal result', status: 'success' },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(1);
      expect(messages[0]._getType()).toBe('tool');
    });

    it('passes MessageContentComplex[] content through without stringifying', async () => {
      const toolNode = new ToolNode({
        tools: [createDummyTool()],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_3', 'step_3']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_3', name: 'dummy', args: { x: 'test' } }],
      });

      const complexContent = [
        { type: 'text', text: 'Multi-part skill instructions' },
        { type: 'text', text: 'Second part of instructions' },
      ];

      mockEventDispatch([
        {
          toolCallId: 'call_3',
          content: '',
          status: 'success',
          injectedMessages: [
            {
              role: 'user' as const,
              content: complexContent,
              isMeta: true,
              source: 'skill' as const,
              skillName: 'complex-skill',
            },
          ],
        },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(2);
      // ToolMessage first
      expect(messages[0]._getType()).toBe('tool');
      // Injected message second with array content preserved (not stringified)
      const injected = messages[1] as HumanMessage;
      expect(injected).toBeInstanceOf(HumanMessage);
      expect(Array.isArray(injected.content)).toBe(true);
      expect(injected.content).toEqual(complexContent);
    });

    it('aggregates injected messages from multiple tool calls', async () => {
      const toolNode = new ToolNode({
        tools: [createDummyTool('tool_a'), createDummyTool('tool_b')],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([
          ['call_a', 'step_a'],
          ['call_b', 'step_b'],
        ]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call_a', name: 'tool_a', args: { x: 'a' } },
          { id: 'call_b', name: 'tool_b', args: { x: 'b' } },
        ],
      });

      mockEventDispatch([
        {
          toolCallId: 'call_a',
          content: 'Result A',
          status: 'success',
          injectedMessages: [
            {
              role: 'user',
              content: 'Injected from A',
              isMeta: true,
              source: 'skill',
              skillName: 'skill-a',
            },
          ],
        },
        {
          toolCallId: 'call_b',
          content: 'Result B',
          status: 'success',
          injectedMessages: [
            {
              role: 'user',
              content: 'Injected from B',
              isMeta: true,
              source: 'skill',
              skillName: 'skill-b',
            },
          ],
        },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      // 2 ToolMessages + 2 injected messages
      expect(messages).toHaveLength(4);
      // ToolMessages come first
      expect(messages[0]._getType()).toBe('tool');
      expect(messages[1]._getType()).toBe('tool');
      // Injected messages come after all ToolMessages
      expect(messages[2]).toBeInstanceOf(HumanMessage);
      expect((messages[2] as HumanMessage).content).toBe('Injected from A');
      expect(messages[3]).toBeInstanceOf(HumanMessage);
      expect((messages[3] as HumanMessage).content).toBe('Injected from B');
    });

    it('handles mixed mode: direct tools + event-driven with injected messages', async () => {
      const directTool = tool(async () => 'direct result', {
        name: 'handoff_tool',
        description: 'A direct tool',
        schema: z.object({ target: z.string() }),
      }) as unknown as StructuredToolInterface;

      const eventTool = createDummyTool('event_tool');

      const toolNode = new ToolNode({
        tools: [directTool, eventTool],
        eventDrivenMode: true,
        agentId: 'test-agent',
        directToolNames: new Set(['handoff_tool']),
        toolCallStepIds: new Map([
          ['call_direct', 'step_direct'],
          ['call_event', 'step_event'],
        ]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_direct',
            name: 'handoff_tool',
            args: { target: 'agent-2' },
          },
          { id: 'call_event', name: 'event_tool', args: { x: 'hello' } },
        ],
      });

      mockEventDispatch([
        {
          toolCallId: 'call_event',
          content: 'Event result',
          status: 'success',
          injectedMessages: [
            {
              role: 'user',
              content: 'Skill body from event tool',
              isMeta: true,
              source: 'skill',
              skillName: 'my-skill',
            },
          ],
        },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      // directOutputs first, then eventResult.toolMessages, then eventResult.injected
      expect(messages.length).toBeGreaterThanOrEqual(3);
      // Direct tool result (ToolMessage from runTool)
      expect(messages[0]._getType()).toBe('tool');
      // Event tool result (ToolMessage from dispatchToolEvents)
      expect(messages[1]._getType()).toBe('tool');
      // Injected message last
      const last = messages[messages.length - 1] as HumanMessage;
      expect(last).toBeInstanceOf(HumanMessage);
      expect(last.content).toBe('Skill body from event tool');
      expect(last.additional_kwargs.skillName).toBe('my-skill');
    });

    it('includes injected messages even when tool result has error status', async () => {
      const toolNode = new ToolNode({
        tools: [createDummyTool()],
        eventDrivenMode: true,
        agentId: 'test-agent',
        toolCallStepIds: new Map([['call_err', 'step_err']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_err', name: 'dummy', args: { x: 'fail' } }],
      });

      mockEventDispatch([
        {
          toolCallId: 'call_err',
          content: '',
          status: 'error',
          errorMessage: 'Skill not found',
          injectedMessages: [
            {
              role: 'user',
              content: 'Partial context before failure',
              isMeta: true,
              source: 'skill',
              skillName: 'broken-skill',
            },
          ],
        },
      ]);

      const result = await toolNode.invoke({ messages: [aiMsg] });
      const messages = (result as { messages: BaseMessage[] }).messages;

      expect(messages).toHaveLength(2);
      // Error ToolMessage first
      expect(messages[0]._getType()).toBe('tool');
      expect(String(messages[0].content)).toContain('Skill not found');
      // Injected message still included
      const injected = messages[1] as HumanMessage;
      expect(injected).toBeInstanceOf(HumanMessage);
      expect(injected.content).toBe('Partial context before failure');
      expect(injected.additional_kwargs.skillName).toBe('broken-skill');
    });
  });
});
