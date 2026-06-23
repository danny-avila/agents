const mockWithLangfuseRuntimeScope = jest.fn(
  (_scope: unknown, action: () => unknown) => action()
);
const mockResolvedScope = {
  langfuse: {
    toolOutputTracing: { enabled: false },
  },
};
const mockResolveLangfuseRuntimeScope = jest.fn(() => mockResolvedScope);

jest.mock('@/langfuseRuntimeScope', () => ({
  ...jest.requireActual('@/langfuseRuntimeScope'),
  resolveLangfuseRuntimeScope: mockResolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope: mockWithLangfuseRuntimeScope,
}));

import { ToolNode } from '../ToolNode';

describe('ToolNode Langfuse redaction context', () => {
  beforeEach(() => {
    mockWithLangfuseRuntimeScope.mockClear();
    mockResolveLangfuseRuntimeScope.mockClear();
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

    expect(mockResolveLangfuseRuntimeScope).toHaveBeenCalledWith({
      runLangfuse,
      langfuseOverlay: agentLangfuse,
    });
    expect(mockWithLangfuseRuntimeScope).toHaveBeenCalledWith(
      mockResolvedScope,
      expect.any(Function)
    );
  });
});
