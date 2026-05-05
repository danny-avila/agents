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

import { homedir } from 'os';
import { isAbsolute, relative, resolve } from 'path';
import { realpath } from 'fs/promises';
import { Constants } from '@/common';
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
  Constants.READ_FILE,
  Constants.GREP_SEARCH,
  Constants.GLOB_SEARCH,
  Constants.LIST_DIRECTORY,
  Constants.COMPILE_CHECK,
]);

const WRITE_TOOLS = new Set<string>([
  Constants.WRITE_FILE,
  Constants.EDIT_FILE,
]);

/**
 * Best-effort extractor for `compile_check` — pulls absolute and `~/`
 * path tokens out of the `command` string so the workspace boundary
 * sees them. Without this, a model could ship `command: 'cat
 * /etc/passwd'` and the policy hook would short-circuit to `allow`
 * (Codex P1 #26 — the prior `() => []` made the hook a no-op for
 * compile_check). Conservative by design:
 *
 *   - Matches `/foo`, `~/foo`, `$HOME/foo`, `${HOME}/foo` followed by
 *     non-shell-special chars. Stops at whitespace, quotes, redirect
 *     operators, pipes, semicolons.
 *   - Strips a leading `--flag=` so `--out=/etc/foo` extracts as
 *     `/etc/foo` (the path the agent's actually trying to write).
 *   - Misses relative paths (intended — those resolve under cwd
 *     anyway), and shell-substituted paths whose final form isn't
 *     visible at extract time. Hosts that need bulletproof gating
 *     should pair this with a `bash_tool`-level policy.
 */
const ABSOLUTE_PATH_TOKEN =
  /(?:^|[\s=])(?:--[^\s=]+=)?(\/[^\s'"|;&<>()`]+|~\/[^\s'"|;&<>()`]+|\$\{?HOME\}?\/[^\s'"|;&<>()`]+)/g;
function expandHomeRelative(token: string): string {
  // Expand ~/foo and $HOME/foo and ${HOME}/foo to absolute. The
  // workspace boundary check resolves non-absolute paths against the
  // workspace root, which would silently treat `~/secret` as
  // `<workspace>/~/secret` — exactly the bypass the codex flagged.
  const home = homedir();
  if (token.startsWith('~/')) return `${home}/${token.slice(2)}`;
  if (token.startsWith('${HOME}/')) return `${home}/${token.slice(8)}`;
  if (token.startsWith('$HOME/')) return `${home}/${token.slice(6)}`;
  return token;
}
function extractCompileCheckPaths(input: Record<string, unknown>): string[] {
  const command = typeof input.command === 'string' ? input.command : '';
  if (command === '') return [];
  const out: string[] = [];
  for (const match of command.matchAll(ABSOLUTE_PATH_TOKEN)) {
    out.push(expandHomeRelative(match[1]));
  }
  return out;
}

const DEFAULT_EXTRACTORS: Record<string, PathExtractor> = {
  [Constants.READ_FILE]: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  [Constants.WRITE_FILE]: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  [Constants.EDIT_FILE]: (i) =>
    typeof i.file_path === 'string' ? [i.file_path] : [],
  [Constants.GREP_SEARCH]: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  [Constants.GLOB_SEARCH]: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  [Constants.LIST_DIRECTORY]: (i) =>
    typeof i.path === 'string' && i.path !== '' ? [i.path] : [],
  [Constants.COMPILE_CHECK]: extractCompileCheckPaths,
};

function isInsideAnyRoot(absolutePath: string, roots: string[]): boolean {
  for (const root of roots) {
    if (absolutePath === root) return true;
    const rel = relative(root, absolutePath);
    if (!rel.startsWith('..') && !isAbsolute(rel)) return true;
  }
  return false;
}

/**
 * Symlink-aware variant: realpaths the candidate AND the roots before
 * comparing. Without this, a symlink inside the workspace pointing
 * outside (e.g. `workspace/link → /etc/passwd`) compares as
 * "in-workspace" lexically, but actually grants the agent reach
 * outside the boundary. Critical when this hook is the primary gate
 * (i.e. the host opted into `workspace.allowReadOutside: true` /
 * `allowWriteOutside: true` so the file tools' own clamp is off).
 *
 * Handles paths that don't yet exist (e.g. `write_file` to a brand
 * new path) by walking up to the nearest existing ancestor and
 * realpathing that, then re-attaching the unresolved suffix. Mirrors
 * `resolveWorkspacePathSafe`'s approach in LocalExecutionEngine.
 */
async function realpathOrSelf(absolutePath: string): Promise<string> {
  try {
    return await realpath(absolutePath);
  } catch {
    return absolutePath;
  }
}

async function realpathOfPathOrAncestor(
  absolutePath: string
): Promise<string> {
  let current = absolutePath;
  let suffix = '';
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  while (true) {
    try {
      const real = await realpath(current);
      return suffix === '' ? real : resolve(real, suffix);
    } catch {
      const parent = resolve(current, '..');
      if (parent === current) {
        return absolutePath;
      }
      const base = current.slice(parent.length + 1);
      suffix = suffix === '' ? base : `${base}/${suffix}`;
      current = parent;
    }
  }
}

async function isInsideAnyRootRealpath(
  absolutePath: string,
  realRoots: readonly string[]
): Promise<boolean> {
  const real = await realpathOfPathOrAncestor(absolutePath);
  return isInsideAnyRoot(real, [...realRoots]);
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
  // Relative `additionalRoots` entries are anchored to `root` so a
  // monorepo config like `additionalRoots: ['../shared']` resolves
  // to a sibling of `root`, not of process.cwd. Matches
  // `getWorkspaceRoots` in LocalExecutionEngine.
  const additionalRoots = (config.additionalRoots ?? []).map((p) =>
    isAbsolute(p) ? resolve(p) : resolve(root, p)
  );
  const allRoots = [root, ...additionalRoots];

  // Pre-realpath the roots once at construction — these are stable
  // per Run. The candidate paths get realpath'd lazily inside the
  // hook callback. Cached so the per-call cost is just one realpath.
  let realRootsPromise: Promise<string[]> | undefined;
  const getRealRoots = (): Promise<string[]> => {
    if (realRootsPromise == null) {
      realRootsPromise = Promise.all(allRoots.map(realpathOrSelf));
    }
    return realRootsPromise;
  };

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

    // Two-stage check:
    //   1. Lexical fast path — anything that's lexically inside the
    //      workspace AND doesn't get redirected by realpath stays
    //      allow-able without paying the realpath cost on every call.
    //   2. For paths that look outside lexically OR look inside but
    //      may have been routed through a symlink, realpath both the
    //      candidate and the roots and compare. This catches the
    //      `workspace/link → /etc/passwd` escape that lexical-only
    //      checks miss.
    const outside: string[] = [];
    const realRoots = await getRealRoots();
    for (const p of paths) {
      const abs = isAbsolute(p) ? resolve(p) : resolve(root, p);
      // Realpath is the source of truth — it catches both the
      // symlink-escape case (lexically-inside path that resolves
      // outside) and the alternate-mount case (lexically-outside
      // path that resolves back inside the workspace). The lexical
      // check alone gives the wrong answer for either, so we don't
      // bother computing it.
      const realInside = await isInsideAnyRootRealpath(abs, realRoots);
      if (!realInside) {
        outside.push(p);
      }
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
