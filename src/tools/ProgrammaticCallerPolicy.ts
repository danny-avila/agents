import type { ProgrammaticRuntime } from './toolIdentifiers';
import type * as t from '@/types';
import { deriveReferencedToolDefs } from './toolIdentifiers';

export type ProgrammaticInvocationParams = {
  code: string;
  tool_manifest?: string[];
  timeout?: number;
  lang?: string;
  runtime?: string;
  language?: string;
};

export function resolveProgrammaticToolDefinitions(
  context: Partial<t.ProgrammaticCache> & { tools?: t.LCTool[] }
): t.LCTool[] | undefined {
  return context.toolDefs ?? context.tools;
}

export function projectProgrammaticToolMap(
  toolMap: t.ToolMap,
  selectedToolDefs: readonly t.LCTool[]
): t.ToolMap {
  const selectedNames = new Set(
    selectedToolDefs.map((toolDef) => toolDef.name)
  );
  return new Map([...toolMap].filter(([name]) => selectedNames.has(name)));
}

export function assertDisallowedToolUsage(
  disallowedNames: ReadonlySet<string>,
  programmaticToolName: string
): void {
  if (disallowedNames.size === 0) {
    return;
  }

  const names = [...disallowedNames];
  throw new Error(
    `Tool${names.length === 1 ? '' : 's'} ${names
      .map((name) => `"${name}"`)
      .join(', ')} cannot be called from "${programmaticToolName}" because ` +
      `the${names.length === 1 ? ' tool is' : 'se tools are'} not marked for code_execution. ` +
      `Call ${names.length === 1 ? 'it' : 'them'} directly instead.`
  );
}

/**
 * Resolves the tool manifest a programmatic call runs with.
 *
 * An omitted manifest is derived from the submitted code rather than rejected.
 * Deriving can only narrow `allowedToolDefs`, so a direct-only tool still cannot
 * be reached; the caller-facing invariant is unchanged. Rejecting instead cost a
 * round trip on every call once any direct-only tool existed, and left tool-free
 * code with no valid manifest to send at all.
 *
 * An empty result is meaningful, not an error: the code calls no tools, so it
 * needs no stubs. Remote callers route that to plain `/exec`.
 */
export function selectProgrammaticTools(args: {
  code: string;
  runtime: ProgrammaticRuntime;
  requestedToolNames?: string[];
  allowedToolDefs?: t.LCTool[];
  disallowedToolDefs?: t.LCTool[];
  programmaticToolName: string;
  /**
   * Set when the runtime defines tool functions the selection does not control,
   * so stub absence cannot enforce the caller boundary. Cloudflare's Python
   * program prepends every native tool unconditionally, and an indirect
   * reference (`globals()["write_file"]`) is beyond what derivation can see —
   * so the manifest stays required rather than derived there.
   */
  runtimeExposesUnselectedTools?: boolean;
}): t.LCTool[] {
  const allowedToolDefs = args.allowedToolDefs ?? [];
  const disallowedToolDefs = args.disallowedToolDefs ?? [];
  const requestedToolNames = args.requestedToolNames;

  if (requestedToolNames == null) {
    if (disallowedToolDefs.length === 0) {
      return allowedToolDefs;
    }

    if (args.runtimeExposesUnselectedTools === true) {
      throw new Error(
        `"${args.programmaticToolName}" requires a tool_manifest when direct-only tools are configured. ` +
          'List every registered tool name used by the submitted code in the tool_manifest field.'
      );
    }

    /* Derive over the disallowed set as well. A direct-only tool the code
     * references must still be rejected: some runtimes define native tool
     * functions unconditionally, so treating an unmatched name as absent would
     * let the call reach one the caller may not use. */
    const referencedDisallowed = deriveReferencedToolDefs(
      disallowedToolDefs,
      args.code,
      args.runtime
    );
    assertDisallowedToolUsage(
      new Set(referencedDisallowed.map((toolDef) => toolDef.name)),
      args.programmaticToolName
    );

    return deriveReferencedToolDefs(allowedToolDefs, args.code, args.runtime);
  }

  const disallowedNames = new Set(
    disallowedToolDefs.map((toolDef) => toolDef.name)
  );
  const requestedDisallowedNames = new Set(
    requestedToolNames.filter((name) => disallowedNames.has(name))
  );
  assertDisallowedToolUsage(
    requestedDisallowedNames,
    args.programmaticToolName
  );

  const allowedByName = new Map(
    allowedToolDefs.map((toolDef) => [toolDef.name, toolDef])
  );
  const unknownNames = new Set(
    requestedToolNames.filter((name) => !allowedByName.has(name))
  );
  if (unknownNames.size > 0) {
    throw new Error(
      `Tool${unknownNames.size === 1 ? '' : 's'} ${[...unknownNames]
        .map((name) => `"${name}"`)
        .join(
          ', '
        )} cannot be used by "${args.programmaticToolName}" because ` +
        `${unknownNames.size === 1 ? 'it is' : 'they are'} not available for code_execution.`
    );
  }

  return [...new Set(requestedToolNames)].map(
    (name) => allowedByName.get(name) as t.LCTool
  );
}
