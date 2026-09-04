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

/** Credential shapes that survive key-name redaction because they live inside
 * free text — an error message, a stack frame, or a rejected request's body
 * echoing the header that was sent. Deliberately shape-based rather than
 * keyword-based: prose naming a secret ("failed to load secret signing-key") is
 * the operator's main clue and must survive. */
const AUTH_SCHEME_RE =
  /\b(Bearer|Basic|Digest|Token)\s+[A-Za-z0-9\-._~+/]{8,}={0,2}/gi;
const JWT_RE = /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]+/g;
const EMBEDDED_URL_CREDENTIALS_RE =
  /\b([a-z][a-z0-9+.-]*:\/\/)[^\s/@:]+:[^\s/@]+@/gi;
const CREDENTIAL_ASSIGNMENT_RE =
  /("?[A-Za-z0-9_-]*(?:key|token|secret|credential|password|signature|auth)[A-Za-z0-9_-]*"?\s*[=:]\s*)"?[^\s,;&"'}\])]+"?/gi;

export function redactSecretText(value: string): string {
  return value
    .replace(EMBEDDED_URL_CREDENTIALS_RE, `$1${REDACTED_VALUE}@`)
    .replace(AUTH_SCHEME_RE, `$1 ${REDACTED_VALUE}`)
    .replace(JWT_RE, REDACTED_VALUE)
    .replace(CREDENTIAL_ASSIGNMENT_RE, `$1${REDACTED_VALUE}`);
}

/** Recursively removes credentials from structured diagnostic payloads. */
export function redactSecrets(
  value: unknown,
  seen: WeakSet<object> = new WeakSet()
): unknown {
  if (typeof value === 'string') {
    return redactSecretText(redactUrlCredentials(value));
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
