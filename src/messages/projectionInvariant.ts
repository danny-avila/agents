import type { BaseMessage } from '@langchain/core/messages';
import { inspectProviderMessageProvenance } from './provenance';

export type ProviderMessageProjectionInvariantMode =
  | 'off'
  | 'observe'
  | 'assert';

export type ProviderMessageProjectionIssueCode =
  | 'absent_provenance'
  | 'invalid_provenance'
  | 'unsourced_non_synthetic_part';

export interface ProviderMessageProjectionInvariantIssue {
  readonly code: ProviderMessageProjectionIssueCode;
  readonly messageIndex: number;
  readonly messageType: string;
}

export interface ProviderMessageProjectionInvariantReport {
  readonly valid: boolean;
  readonly messageCount: number;
  readonly sourceBackedMessageCount: number;
  readonly syntheticMessageCount: number;
  readonly gapMessageCount: number;
  readonly issues: readonly ProviderMessageProjectionInvariantIssue[];
}

const MAX_REPORTED_PROJECTION_ISSUES = 64;

function getMessageType(message: BaseMessage): string {
  try {
    return typeof message.type === 'string' ? message.type : 'unknown';
  } catch {
    return 'unknown';
  }
}

function appendIssue(
  issues: ProviderMessageProjectionInvariantIssue[],
  code: ProviderMessageProjectionIssueCode,
  messageIndex: number,
  messageType: string
): void {
  if (issues.length >= MAX_REPORTED_PROJECTION_ISSUES) {
    return;
  }
  issues.push(Object.freeze({ code, messageIndex, messageType }));
}

/** Resolves the opt-in provider projection invariant without enabling it for
 * unrecognized values. */
export function resolveProviderMessageProjectionInvariantMode(
  value = process.env.AGENT_MESSAGE_PROJECTION_INVARIANT
): ProviderMessageProjectionInvariantMode {
  if (value === 'observe' || value === 'assert') {
    return value;
  }
  return 'off';
}

/** Inspects provider-bound lineage without reading message content or source ids. */
export function inspectProviderMessageProjection(
  messages: readonly BaseMessage[]
): ProviderMessageProjectionInvariantReport {
  const issues: ProviderMessageProjectionInvariantIssue[] = [];
  let sourceBackedMessageCount = 0;
  let syntheticMessageCount = 0;
  let gapMessageCount = 0;

  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex];
    const messageType = getMessageType(message);
    const state = inspectProviderMessageProvenance(message);
    if (state.status !== 'valid') {
      gapMessageCount++;
      appendIssue(
        issues,
        state.status === 'absent'
          ? 'absent_provenance'
          : 'invalid_provenance',
        messageIndex,
        messageType
      );
      continue;
    }

    let hasSourceBackedPart = false;
    let hasGap = false;
    for (const part of state.provenance.parts) {
      if (part.sourceMessageId != null) {
        hasSourceBackedPart = true;
        continue;
      }
      if (part.attribution === 'synthetic') {
        continue;
      }
      hasGap = true;
      appendIssue(
        issues,
        'unsourced_non_synthetic_part',
        messageIndex,
        messageType
      );
    }

    if (hasGap) {
      gapMessageCount++;
    } else if (hasSourceBackedPart) {
      sourceBackedMessageCount++;
    } else {
      syntheticMessageCount++;
    }
  }

  return Object.freeze({
    valid: gapMessageCount === 0,
    messageCount: messages.length,
    sourceBackedMessageCount,
    syntheticMessageCount,
    gapMessageCount,
    issues: Object.freeze(issues),
  });
}

export class ProviderMessageProjectionInvariantError extends Error {
  readonly report: ProviderMessageProjectionInvariantReport;

  constructor(report: ProviderMessageProjectionInvariantReport) {
    super('Provider message projection invariant failed');
    this.name = 'ProviderMessageProjectionInvariantError';
    this.report = report;
  }
}
