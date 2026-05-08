// src/tools/SkillTool.ts
import { Constants } from '@/common';

export const SkillToolName = Constants.SKILL_TOOL;

export const SkillToolDescription = `Invoke a skill from the user's library. Skills provide domain-specific instructions loaded into the conversation context, and may also provide files accessible via available tools depending on the runtime environment.

WHEN TO USE:
- The user's request matches a skill listed in the "Available Skills" section of the system prompt.
- You MUST invoke the matching skill BEFORE attempting the task yourself.

WHAT HAPPENS:
- The skill's full instructions are loaded into the conversation as context.
- Files bundled with the skill may become accessible via available tools.
- Follow the skill's instructions to complete the task.

CONSTRAINTS:
- Do not invoke a skill that is already active in this conversation.
- Skill names come from the catalog only. Do not guess names.`;

/**
 * JSON Schema for the SkillTool parameters.
 * Single source of truth used by both SkillToolDefinition (LCTool registry)
 * and createSkillTool() (DynamicStructuredTool instance).
 */
export const SkillToolSchema = {
  type: 'object',
  properties: {
    skillName: {
      type: 'string',
      description:
        'The kebab-case identifier of the skill to invoke (e.g. "financial-analyzer", "meeting-notes"). Must match a name from the "Available Skills" section.',
    },
    args: {
      type: 'string',
      description: 'Optional freeform arguments string passed to the skill.',
    },
  },
  required: ['skillName'],
} as const;

export const SkillToolDefinition = {
  name: SkillToolName,
  description: SkillToolDescription,
  parameters: SkillToolSchema,
} as const;
