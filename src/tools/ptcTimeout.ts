import { EnvVar } from '@/common';

export const DEFAULT_CODE_API_RUN_TIMEOUT_MS = 15_000;
export const MIN_CODE_API_RUN_TIMEOUT_MS = 1_000;

type TimeoutSchema = {
  type: 'integer';
  minimum: number;
  maximum: number;
  default: number;
  description: string;
};

export type ProgrammaticToolCallingJsonSchema = {
  type: 'object';
  properties: {
    code: {
      type: 'string';
      minLength: number;
      description: string;
    };
    timeout: TimeoutSchema;
  };
  required: readonly ['code'];
};

function normalizeTimeoutMs(value: number | undefined): number | undefined {
  if (value == null || !Number.isFinite(value)) {
    return undefined;
  }

  return Math.max(MIN_CODE_API_RUN_TIMEOUT_MS, Math.floor(value));
}

function parseTimeoutMs(value: string | undefined): number | undefined {
  if (value == null || value.trim() === '') {
    return undefined;
  }

  return normalizeTimeoutMs(Number(value));
}

function formatTimeout(timeoutMs: number): string {
  return timeoutMs % 1000 === 0
    ? `${timeoutMs / 1000} seconds`
    : `${timeoutMs} milliseconds`;
}

export function resolveCodeApiRunTimeoutMs(override?: number): number {
  return (
    normalizeTimeoutMs(override) ??
    parseTimeoutMs(process.env[EnvVar.CODE_API_RUN_TIMEOUT_MS]) ??
    DEFAULT_CODE_API_RUN_TIMEOUT_MS
  );
}

export function clampCodeApiRunTimeoutMs(
  timeoutMs: number | undefined,
  maxRunTimeoutMs = resolveCodeApiRunTimeoutMs()
): number {
  const normalizedMaxRunTimeoutMs =
    normalizeTimeoutMs(maxRunTimeoutMs) ?? DEFAULT_CODE_API_RUN_TIMEOUT_MS;
  const normalizedTimeoutMs = normalizeTimeoutMs(timeoutMs);

  if (normalizedTimeoutMs == null) {
    return normalizedMaxRunTimeoutMs;
  }

  return Math.min(normalizedTimeoutMs, normalizedMaxRunTimeoutMs);
}

export function createCodeApiRunTimeoutSchema(
  maxRunTimeoutMs = resolveCodeApiRunTimeoutMs()
): TimeoutSchema {
  const normalizedMaxRunTimeoutMs =
    normalizeTimeoutMs(maxRunTimeoutMs) ?? DEFAULT_CODE_API_RUN_TIMEOUT_MS;
  const formattedTimeout = formatTimeout(normalizedMaxRunTimeoutMs);

  return {
    type: 'integer',
    minimum: MIN_CODE_API_RUN_TIMEOUT_MS,
    maximum: normalizedMaxRunTimeoutMs,
    default: normalizedMaxRunTimeoutMs,
    description:
      'Maximum wall-clock time in milliseconds for one sandbox run or replay iteration. ' +
      'This is not the total multi-round-trip task budget. ' +
      `Default: ${formattedTimeout}. Max: ${formattedTimeout}.`,
  };
}
