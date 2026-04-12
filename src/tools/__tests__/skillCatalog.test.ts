import { describe, it, expect } from '@jest/globals';
import { formatSkillCatalog } from '../skillCatalog';
import type { SkillCatalogEntry } from '@/types';

describe('formatSkillCatalog', () => {
  it('returns empty string for empty array', () => {
    expect(formatSkillCatalog([])).toBe('');
  });

  it('formats a single skill with header', () => {
    const skills: SkillCatalogEntry[] = [
      {
        name: 'pdf-processor',
        description: 'Processes PDF files into structured data.',
      },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toBe(
      '## Available Skills\n\n- pdf-processor: Processes PDF files into structured data.'
    );
  });

  it('formats multiple skills within budget', () => {
    const skills: SkillCatalogEntry[] = [
      { name: 'pdf-processor', description: 'Processes PDF files.' },
      { name: 'review-pr', description: 'Reviews pull requests.' },
      { name: 'meeting-notes', description: 'Formats meeting transcripts.' },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toContain('## Available Skills');
    expect(result).toContain('- pdf-processor: Processes PDF files.');
    expect(result).toContain('- review-pr: Reviews pull requests.');
    expect(result).toContain('- meeting-notes: Formats meeting transcripts.');
  });

  it('caps per-entry descriptions at maxEntryChars', () => {
    const longDesc = 'A'.repeat(300);
    const skills: SkillCatalogEntry[] = [
      { name: 'long-skill', description: longDesc },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toContain('- long-skill: ' + 'A'.repeat(249) + '\u2026');
    expect(result).not.toContain('A'.repeat(300));
  });

  it('truncates descriptions proportionally when over budget', () => {
    const skills: SkillCatalogEntry[] = Array.from({ length: 20 }, (_, i) => ({
      name: `skill-${i}`,
      description: 'D'.repeat(200),
    }));
    const result = formatSkillCatalog(skills, {
      contextWindowTokens: 1000,
      budgetPercent: 0.01,
      charsPerToken: 4,
    });
    expect(result).toContain('## Available Skills');
    for (let i = 0; i < 20; i++) {
      expect(result).toContain(`skill-${i}`);
    }
    expect(result).not.toContain('D'.repeat(200));
  });

  it('falls back to names-only when extremely over budget', () => {
    const skills: SkillCatalogEntry[] = Array.from({ length: 50 }, (_, i) => ({
      name: `s${i}`,
      description: 'Very detailed description here.',
    }));
    const result = formatSkillCatalog(skills, {
      contextWindowTokens: 200,
      budgetPercent: 0.01,
      charsPerToken: 4,
    });
    expect(result).toContain('## Available Skills');
    expect(result).toContain('- s0');
    expect(result).not.toContain(':');
  });

  it('respects custom options', () => {
    const skills: SkillCatalogEntry[] = [
      { name: 'test', description: 'A'.repeat(100) },
    ];
    const result = formatSkillCatalog(skills, { maxEntryChars: 50 });
    expect(result).toContain('A'.repeat(49) + '\u2026');
    expect(result).not.toContain('A'.repeat(100));
  });

  it('includes skills with descriptions shorter than minDescLength', () => {
    const skills: SkillCatalogEntry[] = [
      { name: 'short', description: 'Hi' },
      { name: 'normal', description: 'A normal description here.' },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toContain('- short: Hi');
    expect(result).toContain('- normal: A normal description here.');
  });

  it('handles all skills with zero-length descriptions as names-only', () => {
    const skills: SkillCatalogEntry[] = [
      { name: 'alpha', description: '' },
      { name: 'beta', description: '' },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toBe('## Available Skills\n\n- alpha\n- beta');
  });

  it('has no trailing or leading whitespace', () => {
    const skills: SkillCatalogEntry[] = [
      { name: 'test', description: 'A test skill.' },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).toBe(result.trim());
    const lines = result.split('\n');
    for (const line of lines) {
      expect(line).toBe(line.trimEnd());
    }
  });

  it('ignores displayTitle in output', () => {
    const skills: SkillCatalogEntry[] = [
      {
        name: 'my-skill',
        description: 'Does stuff.',
        displayTitle: 'My Fancy Skill',
      },
    ];
    const result = formatSkillCatalog(skills);
    expect(result).not.toContain('My Fancy Skill');
    expect(result).toContain('- my-skill: Does stuff.');
  });
});
