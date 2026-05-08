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

  it('nudges the model toward heredoc when payloads may contain shell metacharacters', () => {
    /**
     * Real-world failure observed against ClickHouse + bash piping:
     * the model emitted `echo '{{ref}}' | wc -c` and the substituted
     * binary payload contained literal single quotes, breaking the
     * shell. The model self-corrected to a heredoc on retry. Surface
     * the heredoc pattern upfront so the round-trip isn't burned to
     * rediscover it.
     */
    expect(BashToolOutputReferencesGuide).toContain('heredoc');
    expect(BashToolOutputReferencesGuide).toContain('<< \'EOF\'');
  });

  it('separates base and guide with a blank line', () => {
    const composed = buildBashExecutionToolDescription({
      enableToolOutputReferences: true,
    });
    expect(composed.includes(`${BashExecutionToolDescription}\n\n`)).toBe(true);
  });
});
