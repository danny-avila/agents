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

  it('drops free-form context under ANY active policy, even with no matching entry', () => {
    /** Reasoning/intent can quote output from an EARLIER call to the
     *  redacted tool that this batch does not contain, so an active policy
     *  suppresses free-form prose regardless of this batch's entries. */
    const redaction = resolveToolOutputTracingConfig({
      toolOutputTracing: { redactedToolNames: ['unrelated_tool'] },
    });
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      thinkingExcerpts: ['Comparing versions across sources'],
      lastAssistantText: 'Checking the unrelated_tool result from before',
      redaction,
    });
    expect(prompt).not.toContain('Comparing versions across sources');
    expect(prompt).not.toContain('Intent');
    /** Non-matching entries keep their own outcomes. */
    expect(prompt).toContain('PUBLIC_SEARCH_RESULTS');
  });

  it('keeps free-form context when no redaction policy is configured', () => {
    const prompt = buildActivityLabelPrompt({
      entries,
      charLimit: 600,
      thinkingExcerpts: ['Comparing versions across sources'],
      lastAssistantText: 'Verifying each runtime',
      redaction: undefined,
    });
    expect(prompt).toContain('Comparing versions across sources');
    expect(prompt).toContain('Verifying each runtime');
  });

  it('bounds serialization of oversized structured tool output', () => {
    const huge = Array.from({ length: 50_000 }, (_, i) => ({
      id: i,
      blob: 'x'.repeat(200),
    }));
    const prompt = buildActivityLabelPrompt({
      entries: [
        {
          toolName: 'db_rows',
          toolInput: { sql: 'select *' },
          toolOutput: huge,
          status: 'success',
        },
      ],
      charLimit: 600,
    });
    /** Degrades to a shape summary instead of materializing ~10MB of JSON. */
    expect(prompt).toContain('[Array(50000)]');
    expect(prompt.length).toBeLessThan(2000);
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
