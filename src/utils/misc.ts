export function isPresent(value: string | null | undefined): value is string {
  return value != null && value !== '';
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
