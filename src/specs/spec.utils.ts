export function capitalizeFirstLetter(string: string): string {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

export function hasEnv(key: string): boolean {
  return (process.env[key]?.trim() ?? '') !== '';
}

export function hasEveryEnv(keys: readonly string[]): boolean {
  return keys.every(hasEnv);
}

export function hasAnyEnv(keys: readonly string[]): boolean {
  return keys.some(hasEnv);
}
