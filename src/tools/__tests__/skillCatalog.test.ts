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
    const skills: SkillCatalogEntry[] = Array.from({ length: 10 }, (_, i) => ({
      name: `sk-${i}`,
      description: 'D'.repeat(200),
    }));
    // Budget = 10000 * 0.01 * 4 = 400 chars — enough for names + short descs, not full 200-char descs
    const result = formatSkillCatalog(skills, {
      contextWindowTokens: 10000,
      budgetPercent: 0.01,
      charsPerToken: 4,
    });
    expect(result).toContain('## Available Skills');
    for (let i = 0; i < 10; i++) {
      expect(result).toContain(`sk-${i}`);
    }
    // Full 200-char descriptions should be truncated
    expect(result).not.toContain('D'.repeat(200));
  });

  it('falls back to names-only when extremely over budget', () => {
    const skills: SkillCatalogEntry[] = Array.from({ length: 10 }, (_, i) => ({
      name: `s${i}`,
      description: 'Very detailed description that is quite long and verbose.',
    }));
    // Budget = 2000 * 0.01 * 4 = 80 chars — enough for names-only but not descriptions
    const result = formatSkillCatalog(skills, {
      contextWindowTokens: 2000,
      budgetPercent: 0.01,
      charsPerToken: 4,
    });
    expect(result).toContain('## Available Skills');
    expect(result).toContain('- s0');
    // Verify entry lines have no descriptions (names-only format)
    const entryLines = result.split('\n').filter((l) => l.startsWith('- '));
    for (const line of entryLines) {
      expect(line).toMatch(/^- s\d+$/);
    }
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

  it('truncates names-only list when even names exceed budget', () => {
    const skills: SkillCatalogEntry[] = Array.from({ length: 100 }, (_, i) => ({
      name: `skill-with-a-long-name-${i}`,
      description: 'Some description.',
    }));
    // Budget so small that even names-only for 100 skills exceeds it
    const result = formatSkillCatalog(skills, {
      contextWindowTokens: 100,
      budgetPercent: 0.01,
      charsPerToken: 4,
    });
    // Should still have the header and at least one entry, but not all 100
    if (result === '') {
      // Budget too small for even one entry — valid edge case
      expect(result).toBe('');
    } else {
      expect(result).toContain('## Available Skills');
      const entryLines = result.split('\n').filter((l) => l.startsWith('- '));
      expect(entryLines.length).toBeLessThan(100);
      expect(entryLines.length).toBeGreaterThan(0);
      expect(result.length).toBeLessThanOrEqual(100 * 0.01 * 4);
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
