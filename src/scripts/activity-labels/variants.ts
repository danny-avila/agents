/* eslint-disable no-console */
/**
 * System-prompt and user-framing variants for the SDK-side activity-label
 * eval. Unlike the LibreChat harness this was ported from (LibreChat
 * #14527, scripts/activity-labels/), the user prompt here is rendered by
 * the REAL `buildActivityLabelPrompt` from `src/prompts/activityLabel.ts`,
 * so a prompt-builder change is measured directly instead of through a
 * hand-port that can drift.
 *
 * Two instruction baselines matter to this repo:
 *
 * - `sdk-default`  — ACTIVITY_LABEL_PROMPT, what a host gets when it passes
 *                    no `prompt`. LibreChat never exercises this in
 *                    production (it always sends its own instruction), so
 *                    this is the quality floor the SDK ships on its own.
 * - `host-shipped` — a pinned copy of LibreChat's ACTIVITY_INSTRUCTION
 *                    (packages/api/src/agents/activityLabels/runtime.ts at
 *                    dev@a07c0e4ae8), the string every production label
 *                    call actually receives. It is a FIXTURE: if LibreChat
 *                    changes its instruction, update this copy.
 *
 * Framing variants (`entriesHeading` / `terminal`) are HYPOTHESES about the
 * builder: the runner applies them as marker-exact substitutions on the
 * built prompt. When one wins and the builder adopts it, the variant
 * becomes the baseline and the substitution no-ops.
 */
import { ACTIVITY_LABEL_PROMPT } from '@/prompts/activityLabel';

export type Variant = {
  name: string;
  instruction: string;
  /** Feed each step's generated label into the next step's prompt. */
  usePreviousLabels: boolean;
  /** Replaces the `Tool calls:` section heading in the built prompt. */
  entriesHeading?: string;
  /** Replaces the trailing `Label:` terminal in the built prompt. */
  terminal?: string;
};

/** LibreChat's ACTIVITY_INSTRUCTION, verbatim (see module doc). */
export const HOST_SHIPPED_INSTRUCTION: string = [
  'You write the one-line header above a group of tool calls an AI agent just made.',
  'Say what the calls established or produced — the outcome, not the attempt. If they answered a question, the answer is the line.',
  'Write it like a git commit subject: past tense, verb first, leading with the most distinctive file, name, or finding.',
  'Good: "Confirmed /mnt/data resets between calls". "Traced the leak to formatAgentMessages". "Found 3 failing auth tests".',
  'Bad: "Ran 1 command". "Used bash_tool twice". "Executed ls /mnt/data". "Searched the codebase".',
  'If every call failed, say what failed and why, plainly.',
  'A "Previous headers" list may precede the batch: never restate one — if this batch continues that activity, say only what is new.',
  'Never name the tools, never count them, never echo the arguments: the cards below the header already show all three.',
  'Write 4 to 9 words, sentence case, no trailing punctuation, no quotes or markdown.',
  'Output only the line.',
].join(' ');

/** The host fallback builder's entries heading (LibreChat runtime.ts
 *  `buildPrompt`): frames the list as reference material because "without
 *  this the model tends to read the list as the thing to summarize and
 *  hands back a transcription of it". Never shipped on the SDK path;
 *  measured here before any port. */
const GUARD_HEADING =
  'What it called, and what came back (do not restate these):';

export const variants: Variant[] = [
  {
    name: 'sdk-default',
    usePreviousLabels: true,
    instruction: ACTIVITY_LABEL_PROMPT,
  },
  {
    name: 'host-shipped',
    usePreviousLabels: true,
    instruction: HOST_SHIPPED_INSTRUCTION,
  },
  {
    name: 'guard-heading',
    usePreviousLabels: true,
    instruction: HOST_SHIPPED_INSTRUCTION,
    entriesHeading: GUARD_HEADING,
  },
  {
    name: 'guard-full',
    usePreviousLabels: true,
    instruction: HOST_SHIPPED_INSTRUCTION,
    entriesHeading: GUARD_HEADING,
    terminal: 'Header:',
  },
];
