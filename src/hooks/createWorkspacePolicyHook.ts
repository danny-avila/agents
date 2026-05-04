/**
 * Workspace boundary policy as a `PreToolUse` hook.
 *
 * Local-engine file tools enforce a hard workspace boundary at the
 * tool implementation layer (`resolveWorkspacePathSafe`). This hook
 * adds a complementary, host-controlled layer on top that uses the
 * standard PreToolUse / HITL machinery to *negotiate* access to
 * paths outside the workspace — instead of just throwing.
 *
 * The host opts in by registering this hook on a `HookRegistry`; the
 * hook inspects each tool call's input, extracts the file paths it
 * mentions via per-tool extractors, and returns:
 *
 *   - `allow`               — every path is inside `workspace.root`
 *                              (or `additionalRoots`)
 *   - `deny`                — at least one path is outside, and the
 *                              configured outside-policy is `'deny'`
 *   - `ask`                 — at least one path is outside, and the
 *                              outside-policy is `'ask'` (default).
 *                              When `humanInTheLoop.enabled` is true,
 *                              the existing PreToolUse `'ask'` flow
 *                              raises a tool_approval interrupt the
 *                              host UI can render. When HITL is off,
 *                              `'ask'` collapses to `deny` (matches
 *                              the rest of the SDK's default).
 *
 * Default per-tool path extractors cover the local-engine coding
 * suite (`read_file`, `write_file`, `edit_file`, `grep_search`,
 * `glob_search`, `list_directory`, `compile_check`). The host can
 * override or extend via `pathExtractors`. Bash/code paths are not
 * extracted by default — bash command parsing is its own concern, and
 * the existing `bashAst` validator + sandbox-runtime fs allowlist are
 * the right gates for those.
 *
 * Important: this hook does NOT replace `resolveWorkspacePathSafe`.
 * Even if the hook returns `allow`, the file tool still enforces its
 * own clamp unless `workspace.allowReadOutside` /
 * `workspace.allowWriteOutside` (or the legacy
 * `allowOutsideWorkspace`) is set. The recommended composition for
 * "ask the user" semantics is:
 *
 *   workspace: {
 *     root,
 *     allowReadOutside: true,
 *     allowWriteOutside: true,
 *   },
 *   // …with the hook installed and humanInTheLoop.enabled = true.
 */

import { isAbsolute, relative, resolve } from 'path';
import type {
  HookCallback,
  PreToolUseHookInput,
  PreToolUseHookOutput,
  ToolDecision,
} from './types';

/**
 * What to do when a tool call references a path outside the workspace.
 *
 *   - `'ask'`    : default. Raise a PreToolUse `ask` (host UI prompts
 *                  via the HITL interrupt path).
 *   - `'allow'`  : let the call through (use the existing tool clamp
 *                  to actually enforce — the hook is purely advisory).
 *   - `'deny'`   : block the call with an error ToolMessage.
 */
export type OutsideAccessPolicy = 'ask' | 'allow' | 'deny';

export interface WorkspacePolicyConfig {
  /** Canonical workspace root. Required. */
  root: string;
  /** Sibling roots that count as inside-workspace. */
  additionalRoots?: readonly string[];
  /** Policy applied to read-only file tools. Defaults to `'ask'`. */
  outsideRead?: OutsideAccessPolicy;
  /** Policy applied to write-shaped file tools. Defaults to `'ask'`. */
  outsideWrite?: OutsideAccessPolicy;
  /**
   * Optional reason template surfaced in the `ask`/`deny` decision.
   * Supports `{tool}` and `{paths}` substitution.
   */
  reason?: string;
  /**
   * Per-tool path extractors. Defaults cover the local-engine coding
   * suite. Returning an empty array opts that tool out of policy.
   */
  pathExtractors?: Record<string, PathExtractor>;
}

export type PathExtractor = (
  toolInput: Record<string, unknown>
) => readonly string[];

const READ_TOOLS = new Set<string>([
  'read_file',
  'grep_search',
  'glob_search',
  'list_directory',
  'compile_check',
]);

const WRITE_TOOLS = new Set<string>(['write_file', 'edit_file']);

const DEFAULT_EXTRACTORS: Record<string, PathExtractor> = {
  read_file: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  write_file: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  edit_file: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  grep_search: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  glob_search: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  list_directory: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  compile_check: () => [],
};

function isInsideAnyRoot(absolutePath: string, roots: string[]): boolean {
  for (const root of roots) {
    if (absolutePath === root) return true;
    const rel = relative(root, absolutePath);
    if (!rel.startsWith('..') && !isAbsolute(rel)) return true;
  }
  return false;
}

function formatReason(
  template: string | undefined,
  toolName: string,
  outsidePaths: readonly string[]
): string {
  const fallback = `Tool "${toolName}" wants to touch ${outsidePaths.length} path(s) outside the workspace: ${outsidePaths.join(', ')}`;
  if (template == null) return fallback;
  return template
    .replace(/\{tool\}/g, toolName)
    .replace(/\{paths\}/g, outsidePaths.join(', '));
}

/**
 * Build a `PreToolUse` callback that enforces the workspace policy.
 * Register it on a `HookRegistry`:
 *
 * ```ts
 * registry.register('PreToolUse', {
 *   hooks: [createWorkspacePolicyHook({ root, outsideWrite: 'ask' })],
 * });
 * ```
 *
 * The hook is composable with `createToolPolicyHook` — register both;
 * `executeHooks` precedence (`deny > ask > allow`) sorts out which
 * decision wins per call.
 */
export function createWorkspacePolicyHook(
  config: WorkspacePolicyConfig
): HookCallback<'PreToolUse'> {
  const root = resolve(config.root);
  const additionalRoots = (config.additionalRoots ?? []).map((p) => resolve(p));
  const allRoots = [root, ...additionalRoots];

  const readPolicy: OutsideAccessPolicy = config.outsideRead ?? 'ask';
  const writePolicy: OutsideAccessPolicy = config.outsideWrite ?? 'ask';

  const extractors: Record<string, PathExtractor> = {
    ...DEFAULT_EXTRACTORS,
    ...(config.pathExtractors ?? {}),
  };

  return async (input: PreToolUseHookInput): Promise<PreToolUseHookOutput> => {
    const extractor = extractors[input.toolName];
    if (extractor == null) return { decision: 'allow' };

    const paths = extractor(
      (input.toolInput ?? {}) as Record<string, unknown>
    );
    if (paths.length === 0) return { decision: 'allow' };

    const outside: string[] = [];
    for (const p of paths) {
      const abs = isAbsolute(p) ? resolve(p) : resolve(root, p);
      if (!isInsideAnyRoot(abs, allRoots)) outside.push(p);
    }
    if (outside.length === 0) return { decision: 'allow' };

    const policy = WRITE_TOOLS.has(input.toolName)
      ? writePolicy
      : READ_TOOLS.has(input.toolName)
        ? readPolicy
        : writePolicy; // unknown tools — treat as write (stricter)
    if (policy === 'allow') return { decision: 'allow' };

    const decision: ToolDecision = policy === 'deny' ? 'deny' : 'ask';
    return {
      decision,
      reason: formatReason(config.reason, input.toolName, outside),
      ...(decision === 'ask'
        ? { allowedDecisions: ['approve', 'reject'] as const }
        : {}),
    };
  };
}
