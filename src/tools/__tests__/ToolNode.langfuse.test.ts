const mockWithLangfuseToolOutputTracingConfig = jest.fn(
  (_runLangfuse: unknown, action: () => unknown, _agentLangfuse: unknown) =>
    action()
);

jest.mock('@/langfuseToolOutputTracing', () => ({
  ...jest.requireActual('@/langfuseToolOutputTracing'),
  withLangfuseToolOutputTracingConfig: mockWithLangfuseToolOutputTracingConfig,
}));

import { ToolNode } from '../ToolNode';

describe('ToolNode Langfuse redaction context', () => {
  beforeEach(() => {
    mockWithLangfuseToolOutputTracingConfig.mockClear();
  });

  it('uses a stable default run name for tracing', () => {
    const node = new ToolNode({ tools: [] });

    expect(node.name).toBe('tool_batch');
  });

  it('scopes ToolNode invocation with run and agent Langfuse config', async () => {
    const runLangfuse = {
      toolOutputTracing: { enabled: true },
    };
    const agentLangfuse = {
      toolOutputTracing: { enabled: false },
    };
    const node = new ToolNode({
      tools: [],
      runLangfuse,
      agentLangfuse,
    });

    await expect(node.invoke([])).rejects.toThrow(
      'ToolNode only accepts AIMessages'
    );

    expect(mockWithLangfuseToolOutputTracingConfig).toHaveBeenCalledWith(
      runLangfuse,
      expect.any(Function),
      agentLangfuse
    );
  });
});
