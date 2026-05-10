import { randomUUID } from 'crypto';

export function createSessionId(): string {
  return randomUUID();
}

export function createEntryId(): string {
  return randomUUID().replace(/-/g, '').slice(0, 12);
}

export function createRunId(): string {
  return randomUUID();
}

export function createTimestamp(): string {
  return new Date().toISOString();
}
