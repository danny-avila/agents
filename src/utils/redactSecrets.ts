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
 * echoing the header that was sent. Every pattern is deliberately shape-based
 * rather than "a word containing `key`": prose that merely NAMES a secret
 * ("failed to load secret signing-key") is the operator's main clue and must
 * survive, and an unanchored keyword scan backtracks catastrophically on a long
 * body. Each alternation is literal and each quantifier owns a disjoint
 * character class, so matching stays linear in the length of the input.
 *
 * Whole header lines go first, because `Digest` and `Cookie` spread a
 * credential across a parameter list rather than one opaque blob; the same
 * headers reach us JSON-serialized, so a credential-bearing key redacts its
 * complete quoted value, escapes included. Scheme matching is case-sensitive
 * and requires a non-lowercase character in the value: RFC schemes are
 * capitalized and their values are opaque, while "bearer token expired" is
 * prose worth keeping. Replacements never emit a character a value pattern can
 * start with, which keeps redaction idempotent under repeated passes. */
const AUTH_HEADER_RE =
  /\b(authorization|proxy-authorization|set-cookie|cookie)([ \t]*:[ \t]*)[^\r\n]*/gi;
const AUTH_SCHEME_RE =
  /\b(Bearer|Basic|Token)\s+(?=[A-Za-z0-9\-._~+/]{0,64}[0-9A-Z_~+/=])[A-Za-z0-9\-._~+/]{3,}={0,2}/g;
const JWT_RE = /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]+/g;
const EMBEDDED_URL_CREDENTIALS_RE =
  /\b([a-z][a-z0-9+.-]*:\/\/)[^\s/@:]+(?::[^\s/@]+)?@/gi;
const CREDENTIAL_ASSIGNMENT_RE =
  /\b((?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization|credential|password|passwd|signature|secret|cookie|token|key)"?[ \t]*[=:][ \t]*)(?:"((?:[^"\\\r\n]|\\.)*)"|[^\s,;&"'}\])[]+)/gi;

export function redactSecretText(value: string): string {
  return value
    .replace(AUTH_HEADER_RE, `$1$2${REDACTED_VALUE}`)
    .replace(EMBEDDED_URL_CREDENTIALS_RE, `$1${REDACTED_VALUE}@`)
    .replace(AUTH_SCHEME_RE, `$1 ${REDACTED_VALUE}`)
    .replace(JWT_RE, REDACTED_VALUE)
    .replace(
      CREDENTIAL_ASSIGNMENT_RE,
      (_match, prefix: string, quoted?: string) =>
        quoted === undefined
          ? `${prefix}${REDACTED_VALUE}`
          : `${prefix}"${REDACTED_VALUE}"`
    );
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
