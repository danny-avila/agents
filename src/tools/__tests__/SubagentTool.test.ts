import { describe, it, expect } from '@jest/globals';
import { Constants } from '@/common';
import {
  SubagentToolName,
  SubagentToolDescription,
  SubagentToolDefinition,
  SubagentToolSchema,
  createSubagentToolDefinition,
  buildSubagentToolParams,
} from '../SubagentTool';
import type { SubagentConfig } from '@/types';

describe('SubagentTool', () => {
  describe('schema structure', () => {
    it('has description as required string property', () => {
      expect(SubagentToolSchema.properties.description.type).toBe('string');
      expect(SubagentToolSchema.required).toContain('description');
    });

    it('has subagent_type as required string property', () => {
      expect(SubagentToolSchema.properties.subagent_type.type).toBe('string');
      expect(SubagentToolSchema.required).toContain('subagent_type');
    });

    it('is an object type schema', () => {
      expect(SubagentToolSchema.type).toBe('object');
    });
  });

  describe('SubagentToolDefinition', () => {
    it('has correct name', () => {
      expect(SubagentToolDefinition.name).toBe(Constants.SUBAGENT);
    });

    it('references the same schema object', () => {
      expect(SubagentToolDefinition.parameters).toBe(SubagentToolSchema);
    });

    it('has a non-empty description', () => {
      expect(SubagentToolDefinition.description).toBe(SubagentToolDescription);
      expect(SubagentToolDefinition.description!.length).toBeGreaterThan(0);
    });
  });

  describe('SubagentToolName', () => {
    it('equals Constants.SUBAGENT', () => {
      expect(SubagentToolName).toBe('subagent');
      expect(SubagentToolName).toBe(Constants.SUBAGENT);
    });
  });

  describe('createSubagentToolDefinition', () => {
    const configs: SubagentConfig[] = [
      {
        type: 'researcher',
        name: 'Research Agent',
        description: 'Searches and summarizes information',
      },
      {
        type: 'coder',
        name: 'Coding Agent',
        description: 'Writes and reviews code',
      },
    ];

    it('populates subagent_type enum from configs', () => {
      const def = createSubagentToolDefinition(configs);
      const schema = def.parameters as Record<string, unknown>;
      const props = schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher', 'coder']);
    });

    it('includes type descriptions in tool description', () => {
      const def = createSubagentToolDefinition(configs);
      expect(def.description).toContain('"researcher" (Research Agent)');
      expect(def.description).toContain('"coder" (Coding Agent)');
      expect(def.description).toContain('Searches and summarizes information');
      expect(def.description).toContain('Writes and reviews code');
    });

    it('has correct name', () => {
      const def = createSubagentToolDefinition(configs);
      expect(def.name).toBe(Constants.SUBAGENT);
    });

    it('has required description and subagent_type fields', () => {
      const def = createSubagentToolDefinition(configs);
      const schema = def.parameters as Record<string, unknown>;
      expect(schema.required).toContain('description');
      expect(schema.required).toContain('subagent_type');
    });

    it('works with single config', () => {
      const def = createSubagentToolDefinition([configs[0]]);
      const schema = def.parameters as Record<string, unknown>;
      const props = schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher']);
    });
  });

  describe('buildSubagentToolParams', () => {
    const configs: SubagentConfig[] = [
      {
        type: 'researcher',
        name: 'Research Agent',
        description: 'Searches and summarizes information',
      },
      {
        type: 'coder',
        name: 'Coding Agent',
        description: 'Writes and reviews code',
      },
    ];

    it('returns name matching Constants.SUBAGENT', () => {
      const params = buildSubagentToolParams(configs);
      expect(params.name).toBe(Constants.SUBAGENT);
    });

    it('schema has enum populated from config types', () => {
      const params = buildSubagentToolParams(configs);
      const props = params.schema.properties as Record<
        string,
        Record<string, unknown>
      >;
      expect(props.subagent_type.enum).toEqual(['researcher', 'coder']);
    });

    it('description includes type listings', () => {
      const params = buildSubagentToolParams(configs);
      expect(params.description).toContain('"researcher" (Research Agent)');
      expect(params.description).toContain('"coder" (Coding Agent)');
    });

    it('produces same schema as createSubagentToolDefinition', () => {
      const params = buildSubagentToolParams(configs);
      const def = createSubagentToolDefinition(configs);
      expect(params.name).toBe(def.name);
      expect(params.description).toBe(def.description);
      expect(params.schema).toEqual(def.parameters);
    });
  });
});
