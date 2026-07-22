import type { ActivityLabelToolEntry } from '@/types/activityLabel';
import { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT } from '@/langfuseToolOutputTracing';
import { buildActivityLabelPrompt } from '@/prompts/activityLabel';
import { resolveToolOutputTracingConfig } from '@/langfuseConfig';

const entries: ActivityLabelToolEntry[] = [
  {
    toolName: 'web_search',
    toolInput: { query: 'runtime versions' },
    toolOutput: 'PUBLIC_SEARCH_RESULTS',
    status: 'success',
  },
  {
    toolName: 'db_query',
    toolInput: { sql: 'select 1' },
    error: 'SECRET_CONNECTION_STRING_LEAK',
    status: 'error',
  },
];

describe('buildActivityLabelPrompt redaction', () => {
  it('embeds raw outputs and errors when no redaction policy resolves', () => {
    const prompt = buildActivityLabelPrompt({ entries, charLimit: 600 });
    expect(prompt).toContain('PUBLIC_SEARCH_RESULTS');
    expect(prompt).toContain('SECRET_CONNECTION_STRING_LEAK');
  });

  it('redacts every outcome when tool-output tracing is globally disabled', () => {
    const redaction = resolveToolOutputTracingConfig({
      toolOutputTracing: { enabled: false },
    });
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      redaction,
    });
    expect(prompt).not.toContain('PUBLIC_SEARCH_RESULTS');
    expect(prompt).not.toContain('SECRET_CONNECTION_STRING_LEAK');
    expect(prompt).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    /** Tool names and inputs stay — matching the span processor, which
     *  redacts output fields only. */
    expect(prompt).toContain('web_search');
    expect(prompt).toContain('runtime versions');
  });

  it('drops reasoning excerpts when any batch entry is redacted', () => {
    const redaction = resolveToolOutputTracingConfig({
      toolOutputTracing: { redactedToolNames: ['db_query'] },
    });
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      thinkingExcerpts: [
        'The db_query returned SECRET_CONNECTION_STRING_LEAK earlier',
      ],
      redaction,
    });
    expect(prompt).not.toContain('Reasoning excerpts');
    expect(prompt).not.toContain('SECRET_CONNECTION_STRING_LEAK');
  });

  it('keeps reasoning excerpts when no batch entry matches the policy', () => {
    const redaction = resolveToolOutputTracingConfig({
      toolOutputTracing: { redactedToolNames: ['unrelated_tool'] },
    });
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      thinkingExcerpts: ['Comparing versions across sources'],
      redaction,
    });
    expect(prompt).toContain('Comparing versions across sources');
  });

  it('redacts only named tools, including their error text', () => {
    const redaction = resolveToolOutputTracingConfig({
      toolOutputTracing: { redactedToolNames: ['db_query'] },
    });
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      redaction,
    });
    expect(prompt).toContain('PUBLIC_SEARCH_RESULTS');
    expect(prompt).not.toContain('SECRET_CONNECTION_STRING_LEAK');
    expect(prompt).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });
});
