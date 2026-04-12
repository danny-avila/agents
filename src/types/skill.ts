// src/types/skill.ts
import type { MessageContentComplex } from './stream';

/**
 * A message injected into graph state by any tool execution handler.
 * Generic mechanism — not skill-specific. Any tool returning `injectedMessages`
 * in its `ToolExecuteResult` will have these prepended to state before the ToolMessage.
 */
export type InjectedMessage = {
  /** 'user' for skill body injection, 'system' for context hints */
  role: 'user' | 'system';
  /** Message content — string for simple text, array for complex multi-part content */
  content: string | MessageContentComplex[];
  /** When true, the message is framework-internal: not shown in UI, not counted as a user turn */
  isMeta?: boolean;
  /** Origin tag for downstream consumers (UI, pruner, compaction) */
  source?: 'skill' | 'hook' | 'system';
  /** Only set when source is 'skill', for compaction preservation */
  skillName?: string;
};

/** Minimal skill metadata for catalog assembly. The host provides these from its own data layer. */
export type SkillCatalogEntry = {
  /** Kebab-case identifier (what the model passes to SkillTool) */
  name: string;
  /** One-line description for the catalog listing */
  description: string;
  /** Optional human-readable label (UI only, not shown to model) */
  displayTitle?: string;
};

/**
 * Documentation type for the resolved skill content the host passes back through ToolExecuteResult.
 * The SDK doesn't interpret it, but hosts should follow this shape.
 */
export type SkillExecutionMeta = {
  skillName: string;
  executionMode: 'inline' | 'fork';
  filesStaged: boolean;
  /** Resolved skill directory path in the runtime, if files were staged */
  skillDir?: string;
};
