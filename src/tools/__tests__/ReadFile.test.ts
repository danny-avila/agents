import { describe, it, expect } from '@jest/globals';
import { Constants } from '@/common';
import {
  ReadFileToolName,
  ReadFileToolSchema,
  ReadFileToolDescription,
  ReadFileToolDefinition,
} from '../ReadFile';

describe('ReadFile', () => {
  describe('schema structure', () => {
    it('has file_path as required string property', () => {
      expect(ReadFileToolSchema.properties.file_path.type).toBe('string');
      expect(ReadFileToolSchema.required).toContain('file_path');
    });

    it('is an object type schema', () => {
      expect(ReadFileToolSchema.type).toBe('object');
    });
  });

  describe('ReadFileToolDefinition', () => {
    it('has correct name', () => {
      expect(ReadFileToolDefinition.name).toBe(Constants.READ_FILE);
      expect(ReadFileToolDefinition.name).toBe('read_file');
    });

    it('references the same ReadFileToolSchema object', () => {
      expect(ReadFileToolDefinition.parameters).toBe(ReadFileToolSchema);
    });

    it('has a non-empty description', () => {
      expect(ReadFileToolDefinition.description).toBe(ReadFileToolDescription);
      expect(ReadFileToolDefinition.description.length).toBeGreaterThan(0);
    });
  });

  describe('ReadFileToolName', () => {
    it('equals Constants.READ_FILE', () => {
      expect(ReadFileToolName).toBe('read_file');
      expect(ReadFileToolName).toBe(Constants.READ_FILE);
    });
  });
});
