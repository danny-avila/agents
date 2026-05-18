import type * as t from '@/types';

export function coerceRecordArgs(
  args: unknown
): Record<string, unknown> | undefined {
  if (typeof args === 'string') {
    try {
      const parsed = JSON.parse(args) as unknown;
      return coerceRecordArgs(parsed);
    } catch {
      return undefined;
    }
  }

  if (args === null || typeof args !== 'object' || Array.isArray(args)) {
    return undefined;
  }

  return args as Record<string, unknown>;
}

export function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  }

  if (value !== null && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `{${keys
      .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
      .join(',')}}`;
  }

  return JSON.stringify(value);
}

export function recordArgsEqual(
  left: Record<string, unknown>,
  right: Record<string, unknown>
): boolean {
  return stableStringify(left) === stableStringify(right);
}

export function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export type ToolExecutionPlanCall = {
  id?: string;
  name: string;
  args: unknown;
  stepId?: string;
  codeSessionContext?: t.ToolCallRequest['codeSessionContext'];
};

export type ToolExecutionRequestPlan = {
  allRequests: t.ToolCallRequest[];
  requests: t.ToolCallRequest[];
  rejectedResults: t.ToolExecuteResult[];
};

export function buildToolExecutionRequestPlan(args: {
  toolCalls: ToolExecutionPlanCall[];
  usageCount: Map<string, number>;
  invalidArgsBehavior?: 'abort' | 'error-result';
  recordTurn?: (toolName: string, turn: number, callId: string) => void;
}): ToolExecutionRequestPlan | undefined {
  const invalidArgsBehavior = args.invalidArgsBehavior ?? 'abort';
  const prepared: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
    stepId?: string;
    codeSessionContext?: t.ToolCallRequest['codeSessionContext'];
    rejectedErrorMessage?: string;
  }> = [];

  for (const toolCall of args.toolCalls) {
    if (
      toolCall.id == null ||
      toolCall.id === '' ||
      toolCall.name === ''
    ) {
      return undefined;
    }
    const coercedArgs = coerceRecordArgs(toolCall.args);
    if (coercedArgs == null) {
      if (invalidArgsBehavior === 'abort') {
        return undefined;
      }
      prepared.push({
        id: toolCall.id,
        name: toolCall.name,
        args: {},
        stepId: toolCall.stepId,
        codeSessionContext: toolCall.codeSessionContext,
        rejectedErrorMessage:
          'Invalid tool call arguments: expected a JSON object.',
      });
      continue;
    }
    prepared.push({
      id: toolCall.id,
      name: toolCall.name,
      args: coercedArgs,
      stepId: toolCall.stepId,
      codeSessionContext: toolCall.codeSessionContext,
    });
  }

  const nextUsageCount = new Map(args.usageCount);
  const allRequests = prepared.map((toolCall): t.ToolCallRequest => {
    const turn = nextUsageCount.get(toolCall.name) ?? 0;
    nextUsageCount.set(toolCall.name, turn + 1);
    const request: t.ToolCallRequest = {
      id: toolCall.id,
      name: toolCall.name,
      args: toolCall.args,
      stepId: toolCall.stepId,
      turn,
    };
    if (toolCall.codeSessionContext != null) {
      request.codeSessionContext = toolCall.codeSessionContext;
    }
    return request;
  });
  const requests = allRequests.filter(
    (_, index) => prepared[index].rejectedErrorMessage == null
  );
  const rejectedResults = prepared.flatMap((toolCall) => {
    if (toolCall.rejectedErrorMessage == null) {
      return [];
    }
    return [
      {
        toolCallId: toolCall.id,
        status: 'error' as const,
        content: '',
        errorMessage: toolCall.rejectedErrorMessage,
      },
    ];
  });

  for (const [toolName, count] of nextUsageCount) {
    args.usageCount.set(toolName, count);
  }
  for (const request of allRequests) {
    args.recordTurn?.(request.name, request.turn ?? 0, request.id);
  }

  return { allRequests, requests, rejectedResults };
}
