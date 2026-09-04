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

const INTEGER_STRING = /^-?(?:0|[1-9]\d*)$/;
const NUMBER_STRING = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/;

function coerceValueForSchema(
  value: unknown,
  schema: t.JsonSchemaType | undefined
): unknown {
  if (schema == null) {
    return value;
  }

  if (schema.type === 'integer' && typeof value === 'string') {
    if (!INTEGER_STRING.test(value)) {
      return value;
    }
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) ? parsed : value;
  }

  if (
    (schema.type === 'number' || schema.type === 'float') &&
    typeof value === 'string'
  ) {
    if (!NUMBER_STRING.test(value)) {
      return value;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) && String(parsed) === value
      ? parsed
      : value;
  }

  if (schema.type === 'boolean' && typeof value === 'string') {
    if (value === 'true') {
      return true;
    }
    if (value === 'false') {
      return false;
    }
    return value;
  }

  if (schema.type === 'array' && Array.isArray(value)) {
    return value.map((item) => coerceValueForSchema(item, schema.items));
  }

  if (
    schema.type !== 'object' ||
    value == null ||
    typeof value !== 'object' ||
    Array.isArray(value)
  ) {
    return value;
  }

  const record = value as Record<string, unknown>;
  const additionalProperties =
    typeof schema.additionalProperties === 'object'
      ? schema.additionalProperties
      : undefined;

  return Object.fromEntries(
    Object.entries(record).map(([key, entry]) => [
      key,
      coerceValueForSchema(
        entry,
        schema.properties?.[key] ?? additionalProperties
      ),
    ])
  );
}

/**
 * Applies only lossless, schema-directed repairs to model-generated arguments.
 * The host remains responsible for full schema validation and all business rules.
 */
export function coerceArgsForSchema(
  args: Record<string, unknown>,
  schema: t.JsonSchemaType | undefined
): Record<string, unknown> {
  return coerceValueForSchema(args, schema) as Record<string, unknown>;
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
  runtimeSessionHint?: string;
};

/**
 * Stateful runtime session hint for the remote sandbox: only when
 * `toolExecution.sandbox.statefulSessions` is on; explicit host hint else the
 * conversation `thread_id`. Undefined disables the wire field. Shared by the
 * direct ToolNode path and both event-driven planners so they stay in lockstep.
 */
export function resolveRuntimeSessionHint(
  toolExecution: t.ToolExecutionConfig | undefined,
  threadId: string | undefined
): string | undefined {
  const sandbox = toolExecution?.sandbox;
  if (sandbox?.statefulSessions !== true) {
    return undefined;
  }
  const explicit = sandbox.runtimeSessionHint;
  if (explicit != null && explicit !== '') {
    return explicit;
  }
  return threadId != null && threadId !== '' ? threadId : undefined;
}

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
  getToolSchema?: (toolName: string) => t.JsonSchemaType | undefined;
}): ToolExecutionRequestPlan | undefined {
  const invalidArgsBehavior = args.invalidArgsBehavior ?? 'abort';
  const prepared: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
    stepId?: string;
    codeSessionContext?: t.ToolCallRequest['codeSessionContext'];
    runtimeSessionHint?: string;
    rejectedErrorMessage?: string;
  }> = [];

  for (const toolCall of args.toolCalls) {
    if (toolCall.id == null || toolCall.id === '' || toolCall.name === '') {
      return undefined;
    }
    const recordArgs = coerceRecordArgs(toolCall.args);
    if (recordArgs == null) {
      if (invalidArgsBehavior === 'abort') {
        return undefined;
      }
      prepared.push({
        id: toolCall.id,
        name: toolCall.name,
        args: {},
        stepId: toolCall.stepId,
        codeSessionContext: toolCall.codeSessionContext,
        runtimeSessionHint: toolCall.runtimeSessionHint,
        rejectedErrorMessage:
          'Invalid tool call arguments: expected a JSON object.',
      });
      continue;
    }
    const coercedArgs = coerceArgsForSchema(
      recordArgs,
      args.getToolSchema?.(toolCall.name)
    );
    prepared.push({
      id: toolCall.id,
      name: toolCall.name,
      args: coercedArgs,
      stepId: toolCall.stepId,
      codeSessionContext: toolCall.codeSessionContext,
      runtimeSessionHint: toolCall.runtimeSessionHint,
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
    if (
      toolCall.runtimeSessionHint != null &&
      toolCall.runtimeSessionHint !== ''
    ) {
      request.runtimeSessionHint = toolCall.runtimeSessionHint;
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
