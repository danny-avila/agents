import { describe, it, expect } from '@jest/globals';
import {
  BashExecutionToolDescription,
  BashToolOutputReferencesGuide,
  buildBashExecutionToolDescription,
} from '../BashExecutor';

describe('buildBashExecutionToolDescription', () => {
  it('returns the base description by default', () => {
    expect(buildBashExecutionToolDescription()).toBe(
      BashExecutionToolDescription
    );
    expect(buildBashExecutionToolDescription({})).toBe(
      BashExecutionToolDescription
    );
    expect(
      buildBashExecutionToolDescription({ enableToolOutputReferences: false })
    ).toBe(BashExecutionToolDescription);
  });

  it('appends the tool-output references guide when enabled', () => {
    const composed = buildBashExecutionToolDescription({
      enableToolOutputReferences: true,
    });
    expect(composed.startsWith(BashExecutionToolDescription)).toBe(true);
    expect(composed).toContain(BashToolOutputReferencesGuide);
    expect(composed).toContain('{{tool<idx>turn<turn>}}');
  });

  it('separates base and guide with a blank line', () => {
    const composed = buildBashExecutionToolDescription({
      enableToolOutputReferences: true,
    });
    expect(composed.includes(`${BashExecutionToolDescription}\n\n`)).toBe(true);
  });
});
