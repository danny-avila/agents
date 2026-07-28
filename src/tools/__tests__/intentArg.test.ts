import { describe, it, expect } from '@jest/globals';
import type { JsonSchemaType } from '@/types';
import {
  INTENT_ARG,
  INTENT_PROPERTY,
  INTENT_DESCRIPTION,
  withIntent,
  readIntent,
  stripIntent,
  applyOutcome,
  readOutcomeFields,
  resolveToolOutcome,
} from '../intentArg';

describe('withIntent', () => {
  const base: JsonSchemaType = {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Search query' },
      limit: { type: 'number' },
    },
    required: ['query'],
  };

  it('prepends intent as the FIRST property key', () => {
    const next = withIntent(base);
    expect(Object.keys(next.properties ?? {})).toEqual(['intent', 'query', 'limit']);
    expect(next.properties?.[INTENT_ARG]).toEqual(INTENT_PROPERTY);
    /**
     * Must be a copy, not the frozen canonical instance: LangChain's
     * JSON-schema validator stamps `__absolute_uri__` onto subschemas,
     * which throws on a frozen object.
     */
    expect(next.properties?.[INTENT_ARG]).not.toBe(INTENT_PROPERTY);
    expect(Object.isFrozen(next.properties?.[INTENT_ARG])).toBe(false);
  });

  it('never mutates the input schema', () => {
    const frozen = Object.freeze<JsonSchemaType>({
      type: 'object',
      properties: Object.freeze({ query: { type: 'string' } }),
    }) as JsonSchemaType;
    const next = withIntent(frozen);
    expect(next).not.toBe(frozen);
    expect(Object.keys(frozen.properties ?? {})).toEqual(['query']);
    expect(Object.keys(next.properties ?? {})).toEqual(['intent', 'query']);
  });

  it('does not add intent to required', () => {
    const next = withIntent(base);
    expect(next.required).toEqual(['query']);
  });

  it('is idempotent when intent already exists (position preserved)', () => {
    const once = withIntent(base);
    const twice = withIntent(once);
    expect(twice).toBe(once);
    expect(Object.keys(twice.properties ?? {})[0]).toBe('intent');
  });

  it('handles a schema with no properties', () => {
    const next = withIntent({ type: 'object', properties: {} });
    expect(Object.keys(next.properties ?? {})).toEqual(['intent']);
  });

  it('handles an undefined schema', () => {
    const next = withIntent(undefined);
    expect(next.type).toBe('object');
    expect(Object.keys(next.properties ?? {})).toEqual(['intent']);
  });

  it('carries the model-facing instruction', () => {
    expect(INTENT_PROPERTY.description).toBe(INTENT_DESCRIPTION);
    expect(INTENT_DESCRIPTION).toContain('FIRST');
    expect(INTENT_DESCRIPTION).toContain('distinguish that call from its siblings');
  });
});

describe('readIntent', () => {
  it('reads from object args', () => {
    expect(readIntent({ intent: 'Searching for OAuth handling' })).toBe(
      'Searching for OAuth handling'
    );
  });

  it('reads from stringified JSON args', () => {
    expect(readIntent('{"intent":"Searching for OAuth handling","query":"oauth"}')).toBe(
      'Searching for OAuth handling'
    );
  });

  it('returns undefined for absent, empty, or non-string intents', () => {
    expect(readIntent({ query: 'oauth' })).toBeUndefined();
    expect(readIntent({ intent: '' })).toBeUndefined();
    expect(readIntent({ intent: '   ' })).toBeUndefined();
    expect(readIntent({ intent: 42 })).toBeUndefined();
    expect(readIntent(undefined)).toBeUndefined();
    expect(readIntent('not json')).toBeUndefined();
    expect(readIntent('{"intent": broken')).toBeUndefined();
  });
});

describe('stripIntent', () => {
  it('removes the key from object args', () => {
    expect(stripIntent({ intent: 'Searching', query: 'oauth' })).toEqual({ query: 'oauth' });
  });

  it('parses and strips stringified args', () => {
    expect(stripIntent('{"intent":"Searching","query":"oauth"}')).toEqual({ query: 'oauth' });
  });

  it('returns args unchanged when the key is absent', () => {
    const args = { query: 'oauth' };
    expect(stripIntent(args)).toBe(args);
    expect(stripIntent('plain string')).toBe('plain string');
    expect(stripIntent(undefined)).toBeUndefined();
  });
});

describe('applyOutcome', () => {
  const intent = 'Searching for OAuth handling';

  it('prefers a tool-supplied outcome (full replacement)', () => {
    expect(
      applyOutcome(intent, {
        outcome: 'Found 12 results for OAuth handling',
        outcome_patch: { from: 'Searching', to: 'Searched' },
      })
    ).toBe('Found 12 results for OAuth handling');
  });

  it('ignores a blank outcome', () => {
    expect(applyOutcome(intent, { outcome: '  ' })).toBe('Searched for OAuth handling');
  });

  it('applies outcome_patch to the first occurrence only, case-sensitive', () => {
    expect(
      applyOutcome('Searching for Searching patterns', {
        outcome_patch: { from: 'Searching', to: 'Searched' },
      })
    ).toBe('Searched for Searching patterns');
    expect(
      applyOutcome(intent, { outcome_patch: { from: 'searching', to: 'searched' } })
    ).toBe('Searched for OAuth handling');
  });

  it('ignores a patch whose from is empty or absent from the intent', () => {
    expect(applyOutcome(intent, { outcome_patch: { from: '', to: 'x' } })).toBe(
      'Searched for OAuth handling'
    );
    expect(applyOutcome(intent, { outcome_patch: { from: 'Grepping', to: 'Grepped' } })).toBe(
      'Searched for OAuth handling'
    );
  });

  it('mechanically transforms the leading verb', () => {
    expect(applyOutcome('Reading the callback router')).toBe('Read the callback router');
    expect(applyOutcome('Writing the zod schema')).toBe('Wrote the zod schema');
    expect(applyOutcome('Delegating the OAuth audit')).toBe('Delegated the OAuth audit');
  });

  it('preserves a lowercase leading verb', () => {
    expect(applyOutcome('searching for OAuth handling')).toBe('searched for OAuth handling');
  });

  it('leaves an unknown leading word unchanged', () => {
    expect(applyOutcome('Refactoring the zod schema')).toBe('Refactoring the zod schema');
  });

  it('handles a single-word intent', () => {
    expect(applyOutcome('Searching')).toBe('Searched');
  });

  it('returns undefined with neither intent nor outcome', () => {
    expect(applyOutcome(undefined)).toBeUndefined();
    expect(applyOutcome('')).toBeUndefined();
    expect(applyOutcome(undefined, { outcome_patch: { from: 'a', to: 'b' } })).toBeUndefined();
  });

  it('returns the outcome even without an intent', () => {
    expect(applyOutcome(undefined, { outcome: 'Found 12 results' })).toBe('Found 12 results');
  });
});

describe('resolveToolOutcome', () => {
  const args = { intent: 'Searching for OAuth handling', query: 'oauth' };

  it('returns undefined when the tool authored no outcome fields', () => {
    expect(resolveToolOutcome(args)).toBeUndefined();
    expect(resolveToolOutcome(args, {})).toBeUndefined();
    expect(resolveToolOutcome(args, null)).toBeUndefined();
  });

  it('resolves a tool-supplied outcome', () => {
    expect(resolveToolOutcome(args, { outcome: 'Found 12 results' })).toBe('Found 12 results');
  });

  it('resolves a patch against the intent read from args', () => {
    expect(
      resolveToolOutcome(args, { outcome_patch: { from: 'Searching', to: 'Searched' } })
    ).toBe('Searched for OAuth handling');
  });

  it('returns undefined for a patch with no intent in the args', () => {
    expect(
      resolveToolOutcome({ query: 'oauth' }, { outcome_patch: { from: 'a', to: 'b' } })
    ).toBeUndefined();
  });

  it('collapses the label to a bounded single line', () => {
    expect(
      resolveToolOutcome(args, { outcome: 'Found 12 results\n  for   "OAuth handling"' })
    ).toBe('Found 12 results for "OAuth handling"');
    const oversized = resolveToolOutcome(args, { outcome: `Found ${'x'.repeat(500)}` });
    expect(oversized?.length).toBe(256);
    expect(oversized?.endsWith('…')).toBe(true);
    expect(resolveToolOutcome(args, { outcome: ' \n \t ' })).toBe(
      'Searched for OAuth handling'
    );
  });
});

describe('readOutcomeFields', () => {
  it('reads valid fields from an artifact-shaped object', () => {
    expect(readOutcomeFields({ outcome: 'Found 12 results', other: 1 })).toEqual({
      outcome: 'Found 12 results',
      outcome_patch: undefined,
    });
    expect(readOutcomeFields({ outcome_patch: { from: 'a', to: 'b' } })).toEqual({
      outcome: undefined,
      outcome_patch: { from: 'a', to: 'b' },
    });
  });

  it('rejects malformed fields', () => {
    expect(readOutcomeFields(undefined)).toBeUndefined();
    expect(readOutcomeFields('outcome')).toBeUndefined();
    expect(readOutcomeFields({ outcome: 42 })).toBeUndefined();
    expect(readOutcomeFields({ outcome: '  ' })).toBeUndefined();
    expect(readOutcomeFields({ outcome_patch: { from: 'a' } })).toBeUndefined();
    expect(readOutcomeFields({ outcome_patch: 'Searched' })).toBeUndefined();
  });
});
