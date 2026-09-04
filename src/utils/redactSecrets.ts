const SECRET_KEY_RE =
  /key|token|secret|credential|authorization|password|cookie|signature|(?:^|[_-])sig(?:$|[_-])/i;

const REDACTED_VALUE = '[REDACTED]';
const CIRCULAR_VALUE = '[CIRCULAR]';

export function isSecretKey(key: string): boolean {
  return SECRET_KEY_RE.test(key);
}

function redactUrlCredentials(value: string): string {
  if (!/^https?:\/\//i.test(value)) {
    return value;
  }
  try {
    const url = new URL(value);
    if (url.username !== '') {
      url.username = REDACTED_VALUE;
    }
    if (url.password !== '') {
      url.password = REDACTED_VALUE;
    }
    for (const key of url.searchParams.keys()) {
      if (isSecretKey(key)) {
        url.searchParams.set(key, REDACTED_VALUE);
      }
    }
    return url.toString();
  } catch {
    return value;
  }
}

/** Recursively removes credentials from structured diagnostic payloads. */
export function redactSecrets(
  value: unknown,
  seen: WeakSet<object> = new WeakSet()
): unknown {
  if (typeof value === 'string') {
    return redactUrlCredentials(value);
  }
  if (value == null || typeof value !== 'object') {
    return value;
  }
  if (seen.has(value)) {
    return CIRCULAR_VALUE;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    return value.map((entry) => redactSecrets(entry, seen));
  }

  const redacted: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    redacted[key] = isSecretKey(key)
      ? REDACTED_VALUE
      : redactSecrets(entry, seen);
  }
  return redacted;
}
