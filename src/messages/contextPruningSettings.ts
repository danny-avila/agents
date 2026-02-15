/**
 * Default settings for position-based context pruning.
 *
 * These are merged with user-provided overrides so any subset can be customized.
 */

export interface ContextPruningSettings {
  /** Whether position-based pruning is enabled. Default: false (opt-in). */
  enabled: boolean;
  /** Number of recent assistant turns to protect from pruning. Default: 3 */
  keepLastAssistants: number;
  /** Age ratio (0-1) at which soft-trim fires. Default: 0.3 */
  softTrimRatio: number;
  /** Age ratio (0-1) at which hard-clear fires. Default: 0.5 */
  hardClearRatio: number;
  /** Minimum tool result size (chars) before pruning applies. Default: 50000 */
  minPrunableToolChars: number;
  softTrim: {
    /** Maximum total chars after soft-trim. Default: 4000 */
    maxChars: number;
    /** Head portion to keep. Default: 1500 */
    headChars: number;
    /** Tail portion to keep. Default: 1500 */
    tailChars: number;
  };
  hardClear: {
    /** Whether hard-clear is enabled. Default: true */
    enabled: boolean;
    /** Placeholder text for hard-cleared content. */
    placeholder: string;
  };
}

export const DEFAULT_CONTEXT_PRUNING_SETTINGS: ContextPruningSettings = {
  enabled: false,
  keepLastAssistants: 3,
  softTrimRatio: 0.3,
  hardClearRatio: 0.5,
  minPrunableToolChars: 50_000,
  softTrim: {
    maxChars: 4_000,
    headChars: 1_500,
    tailChars: 1_500,
  },
  hardClear: {
    enabled: true,
    placeholder: '[Old tool result content cleared]',
  },
};

/**
 * Merges user-provided partial overrides with the defaults.
 */
export function resolveContextPruningSettings(
  overrides?: Partial<{
    enabled?: boolean;
    keepLastAssistants?: number;
    softTrimRatio?: number;
    hardClearRatio?: number;
    minPrunableToolChars?: number;
    softTrim?: Partial<ContextPruningSettings['softTrim']>;
    hardClear?: Partial<ContextPruningSettings['hardClear']>;
  }>
): ContextPruningSettings {
  if (!overrides) {
    return { ...DEFAULT_CONTEXT_PRUNING_SETTINGS };
  }
  return {
    enabled: overrides.enabled ?? DEFAULT_CONTEXT_PRUNING_SETTINGS.enabled,
    keepLastAssistants:
      overrides.keepLastAssistants ??
      DEFAULT_CONTEXT_PRUNING_SETTINGS.keepLastAssistants,
    softTrimRatio:
      overrides.softTrimRatio ?? DEFAULT_CONTEXT_PRUNING_SETTINGS.softTrimRatio,
    hardClearRatio:
      overrides.hardClearRatio ??
      DEFAULT_CONTEXT_PRUNING_SETTINGS.hardClearRatio,
    minPrunableToolChars:
      overrides.minPrunableToolChars ??
      DEFAULT_CONTEXT_PRUNING_SETTINGS.minPrunableToolChars,
    softTrim: {
      ...DEFAULT_CONTEXT_PRUNING_SETTINGS.softTrim,
      ...overrides.softTrim,
    },
    hardClear: {
      ...DEFAULT_CONTEXT_PRUNING_SETTINGS.hardClear,
      ...overrides.hardClear,
    },
  };
}
