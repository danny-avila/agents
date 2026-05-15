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
