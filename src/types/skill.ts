// src/types/skill.ts

/** Minimal skill metadata for catalog assembly. The host provides these from its own data layer. */
export type SkillCatalogEntry = {
  /** Kebab-case identifier (what the model passes to SkillTool) */
  name: string;
  /** One-line description for the catalog listing */
  description: string;
  /** Optional human-readable label (UI only, not shown to model) */
  displayTitle?: string;
};
