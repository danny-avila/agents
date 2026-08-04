export function isPresent(value: string | null | undefined): value is string {
  return value != null && value !== '';
}

export function parseBooleanEnv(
  value: string | undefined
): boolean | undefined {
  if (value == null) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true;
  }
  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false;
  }

  return undefined;
}

/**
 * Unescapes a c-escaped string
 * @param str The string to unescape
 * @returns The unescaped string
 */
const unescapeString = (string: string): string =>
  string.replace(/\\(.)/g, (_, char) => {
    switch (char) {
    case 'n':
      return '\n';
    case 't':
      return '\t';
    case 'r':
      return '\r';
    case '"':
      return '"';
    case '\'':
      return '\'';
    case '\\':
      return '\\';
    default:
      return char;
    }
  });

/**
 * Recursively unescapes all string values in an object
 * @param obj The object to unescape
 * @returns The unescaped object
 */
export function unescapeObject(obj: unknown, key?: string): unknown {
  if (typeof obj === 'string') {
    let unescaped = unescapeString(obj);
    if (key === 'filePath' && unescaped.match(/^"(.+)"$/)) {
      unescaped = unescaped.substring(1, unescaped.length - 1);
    }
    return unescaped;
  }
  if (Array.isArray(obj)) {
    return obj.map((value) =>
      unescapeObject(value, key === 'contextPaths' ? 'filePath' : '')
    );
  }
  if (typeof obj === 'object' && obj !== null) {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        key,
        unescapeObject(value, key),
      ])
    );
  }
  return obj;
}

/**
 * One signal that fires when either input fires. `AbortSignal.any` is skipped
 * when the inputs collapse to a single signal — the composite is a fresh
 * object per call, and the common cases (one channel, or the host reusing the
 * same controller for both) don't need one.
 */
export function composeAbortSignals(
  a: AbortSignal | undefined,
  b: AbortSignal | undefined
): AbortSignal | undefined {
  if (a == null || a === b) {
    return b;
  }
  if (b == null) {
    return a;
  }
  return AbortSignal.any([a, b]);
}
